"""A small byte-level GPT you can train on a laptop CPU in a few minutes.

Default training data is your own Python standard library source code (always
on disk, no downloads). Gives TurboKV a *real* trained transformer to compress.

    python examples/tiny_gpt.py --steps 1500        # -> examples/tiny_gpt.pt
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import sysconfig
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.h = h
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)  # <- TurboKV hooks this output
        self.proj = nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        # optional replacement attention: attn_fn(q, k, v) -> output, all (B, H, T, hd).
        # examples/eval_perplexity.py uses it to run attention on a compressed KV cache.
        self.attn_fn = None

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(C, dim=2)
        q, k, v = (t.view(B, T, self.h, C // self.h).transpose(1, 2) for t in (q, k, v))
        if self.attn_fn is not None:
            a = self.attn_fn(q, k, v)
        else:
            a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(a.transpose(1, 2).reshape(B, T, C))
        return x + self.mlp(self.ln2(x))


class TinyGPT(nn.Module):
    def __init__(self, vocab=256, d=192, heads=3, layers=4, ctx=512):
        super().__init__()
        self.cfg = dict(vocab=vocab, d=d, heads=heads, layers=layers, ctx=ctx)
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(ctx, d)
        self.blocks = nn.ModuleList(Block(d, heads) for _ in range(layers))
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.tok.weight

    def forward(self, idx):
        x = self.tok(idx) + self.pos(torch.arange(idx.shape[1], device=idx.device))
        for b in self.blocks:
            x = b(x)
        return self.head(self.ln(x))


def load_corpus(root: str | None = None, max_bytes: int = 12_000_000) -> np.ndarray:
    root = root or sysconfig.get_paths()["stdlib"]
    files = sorted(glob.glob(os.path.join(root, "**", "*.py"), recursive=True))
    files = [f for f in files if "test" not in f.lower() and "site-packages" not in f]
    buf = bytearray()
    for f in files:
        with open(f, "rb") as fh:
            buf += fh.read() + b"\n"
        if len(buf) > max_bytes:
            break
    return np.frombuffer(bytes(buf), dtype=np.uint8)


def batches(data, bs, ctx, rng):
    ix = rng.integers(0, len(data) - ctx - 1, bs)
    x = np.stack([data[i : i + ctx] for i in ix]).astype(np.int64)
    y = np.stack([data[i + 1 : i + ctx + 1] for i in ix]).astype(np.int64)
    return torch.from_numpy(x), torch.from_numpy(y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--ctx", type=int, default=256)
    ap.add_argument("--data", default=None, help="folder of .py files (default: stdlib)")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "tiny_gpt.pt"))
    a = ap.parse_args()

    torch.manual_seed(0)
    data = load_corpus(a.data)
    split = int(0.95 * len(data))
    train = data[:split]
    print(f"corpus: {len(data) / 1e6:.1f} MB of Python source")
    model = TinyGPT()
    print(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.05, betas=(0.9, 0.95))
    rng = np.random.default_rng(0)
    t0 = time.time()
    for step in range(a.steps + 1):
        lr = 2e-3 * min(1, step / 100) * 0.5 * (1 + math.cos(math.pi * step / a.steps))
        for g in opt.param_groups:
            g["lr"] = max(lr, 1e-5)
        x, y = batches(train, a.bs, a.ctx, rng)
        loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 100 == 0:
            print(f"step {step:5d}  loss {loss.item():.3f}  ({loss.item() / math.log(2):.2f} bits/byte)"
                  f"  {time.time() - t0:.0f}s", flush=True)
    torch.save({"cfg": model.cfg, "state": model.state_dict(), "val_start": split}, a.out)
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
