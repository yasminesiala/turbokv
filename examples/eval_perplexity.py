"""End-to-end: how much worse does a real trained transformer get when its KV cache
is compressed? Measured as perplexity on held-out text (lower = better).

    # tiny GPT trained by examples/tiny_gpt.py -- no internet needed
    python examples/eval_perplexity.py

    # GPT-2 from Hugging Face (pip install transformers; downloads ~500 MB)
    python examples/eval_perplexity.py --hf gpt2 --text some_long_file.txt

For the tiny GPT every attention layer is replaced by attention over the
compressed cache (including NAT). For HF models a forward hook rewrites K and V
inside the fused QKV projection (NAT variants are skipped there).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

from turbokv.methods import compress_kv, logit_noise_var, nat_logits  # noqa: E402

# (label, spec, use NAT)
DEFAULT_METHODS = [
    ("FP16", "fp16", False),
    *[(f"{name} {b}-bit", spec.format(b=b), nat)
      for b in (4, 3, 2)
      for name, spec, nat in [
          ("int per-token", "int{b}", False),
          ("KIVI-style", "kivi{b}", False),
          ("TurboQuant-prod", "qjl{b}/mse{b}", False),
          ("TurboQuant-mse", "mse{b}/mse{b}", False),
          ("TurboKV", "cal{b}/cal{b}", False),
          ("TurboKV + NAT (experimental)", "cal{b}/cal{b}", True),
      ]],
]


def compressed_attention(spec: str, nat: bool, bits_log: list, seed: int = 0):
    """Returns attn_fn(q, k, v) that attends over the compressed cache, causally."""

    def attn_fn(q, k, v):
        B, H, T, hd = q.shape
        qn, kn, vn = (t.detach().float().numpy() for t in (q, k, v))
        out = np.empty_like(qn)
        mask = np.tril(np.ones((T, T), bool))
        scale = 1 / math.sqrt(hd)
        for b in range(B):
            for h in range(H):
                r = compress_kv(kn[b, h], vn[b, h], spec, seed=1000 * seed + h)
                bits_log.append(r.bits)
                s = qn[b, h] @ r.K.T * scale
                if nat and r.k_resid_sq is not None:
                    s = nat_logits(s, logit_noise_var(qn[b, h], r.K, r.k_resid_sq, scale), mask)
                else:
                    s = np.where(mask, s, -np.inf)
                s -= s.max(-1, keepdims=True)
                p = np.exp(s)
                p /= p.sum(-1, keepdims=True)
                out[b, h] = p @ r.V
        return torch.from_numpy(out).to(q.dtype)

    return attn_fn


class HFKVHook:
    """Rewrites K and V inside a fused QKV projection's output (GPT-2 style)."""

    def __init__(self, n_head, spec, bits_log):
        self.n_head, self.spec, self.bits = n_head, spec, bits_log

    def __call__(self, module, inp, out):
        B, T, C3 = out.shape
        C = C3 // 3
        hd = C // self.n_head
        q, k, v = out.detach().float().split(C, dim=2)
        k = k.reshape(B, T, self.n_head, hd).numpy().copy()
        v = v.reshape(B, T, self.n_head, hd).numpy().copy()
        for b in range(B):
            for h in range(self.n_head):
                r = compress_kv(k[b, :, h], v[b, :, h], self.spec, seed=h)
                k[b, :, h], v[b, :, h] = r.K, r.V
                self.bits.append(r.bits)
        k = torch.from_numpy(k).reshape(B, T, C)
        v = torch.from_numpy(v).reshape(B, T, C)
        return torch.cat([q, k, v], dim=2).to(out.dtype)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", default=None, help="GPT-2 family model id, e.g. gpt2")
    ap.add_argument("--text", default=None, help="held-out text file (default: stdlib tail)")
    ap.add_argument("--seqs", type=int, default=32)
    ap.add_argument("--ctx", type=int, default=256)
    ap.add_argument("--json", default=None, help="write results here")
    ap.add_argument("--seed", type=int, default=0, help="rotation seed")
    a = ap.parse_args()

    if a.hf:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(a.hf)
        model = AutoModelForCausalLM.from_pretrained(a.hf, torch_dtype=torch.float32).eval()
        text = open(a.text, encoding="utf-8", errors="ignore").read()
        ids = tok(text, return_tensors="np")["input_ids"][0]
        logits = lambda x: model(x).logits  # noqa: E731
        name = a.hf
    else:
        from tiny_gpt import TinyGPT, load_corpus

        ck = torch.load(os.path.join(HERE, "tiny_gpt.pt"), weights_only=False)
        model = TinyGPT(**ck["cfg"])
        model.load_state_dict(ck["state"])
        model.eval()
        if a.text:
            ids = np.frombuffer(open(a.text, "rb").read(), np.uint8)
        else:
            data = load_corpus()
            ids = data[ck["val_start"]:]  # held-out tail, never trained on
        logits = model
        name = "tiny-GPT (examples/tiny_gpt.py)"

    need = a.seqs * a.ctx
    if len(ids) < need:
        raise SystemExit(f"text too short: {len(ids)} tokens, need {need}")
    # spread the evaluation windows across the whole held-out text
    starts = np.linspace(0, len(ids) - a.ctx - 1, a.seqs).astype(int)
    x = torch.from_numpy(np.stack([ids[s : s + a.ctx] for s in starts]).astype(np.int64))

    print(f"model: {name} | {a.seqs} windows x {a.ctx} tokens\n")
    print(f"{'method':<28}{'bits':>6}{'compress':>10}{'ppl':>10}{'Δppl':>9}")
    results, base = [], None
    for label, spec, nat in DEFAULT_METHODS:
        if a.hf and nat:
            continue
        bits_log: list = []
        handles = []
        if a.hf:
            hook = HFKVHook(model.config.n_head, spec, bits_log)
            handles = [blk.attn.c_attn.register_forward_hook(hook) for blk in model.transformer.h]
        else:
            fn = None if spec == "fp16" else compressed_attention(spec, nat, bits_log, a.seed)
            for blk in model.blocks:
                blk.attn_fn = fn
        t0 = time.time()
        nll = 0.0
        for i in range(0, a.seqs, 8):
            xb = x[i : i + 8]
            out = logits(xb[:, :-1])
            nll += F.cross_entropy(out.flatten(0, 1).float(), xb[:, 1:].flatten(), reduction="sum").item()
        for h in handles:
            h.remove()
        ppl = math.exp(nll / (a.seqs * (a.ctx - 1)))
        base = base or ppl
        bits = float(np.mean(bits_log)) if bits_log else 16.0
        results.append(dict(label=label, spec=spec, nat=nat, bits=bits, ppl=ppl))
        print(f"{label:<28}{bits:>6.2f}{16 / bits:>9.1f}x{ppl:>10.4f}{ppl - base:>+9.4f}"
              f"   ({time.time() - t0:.0f}s)", flush=True)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(dict(model=name, results=results), f, indent=1)


if __name__ == "__main__":
    main()
