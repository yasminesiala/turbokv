"""Head-to-head on a synthetic KV cache with realistic pathologies.

The cache reproduces what makes real LLM caches hard to quantize: a few
"outlier channels" with 10-20x the energy of the rest, non-zero channel means,
and retrieval-style queries that attend sharply to specific earlier tokens.

    python benchmarks/bench_attention.py            # table
    python benchmarks/bench_attention.py --plot     # + benchmarks/pareto.png
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from turbokv.methods import compress_kv, logit_noise_var, nat_logits  # noqa: E402

METHODS = [
    # (label, spec, noise_aware, family)
    ("FP16", "fp16", False, "fp16"),
    *[(f"int{b} per-token", f"int{b}", False, "int") for b in (4, 3, 2)],
    *[(f"KIVI-style int{b}", f"kivi{b}", False, "kivi") for b in (4, 3, 2)],
    *[(f"TurboQuant-prod K{b}V{b}", f"qjl{b}/mse{b}", False, "qjl") for b in (4, 3, 2)],
    *[(f"TurboQuant-mse K{b}V{b}", f"mse{b}/mse{b}", False, "mse") for b in (4, 3, 2)],
    *[(f"TurboKV K{b}V{b} (ours)", f"cal{b}/cal{b}", False, "cal") for b in (4, 3, 2)],
    *[(f"TurboKV+NAT K{b}V{b}", f"cal{b}/cal{b}", True, "calna") for b in (4, 3, 2)],
]


def make_cache(n=4096, d=128, nq=256, outliers=6, seed=0):
    rng = np.random.default_rng(seed)
    ch_scale = np.ones(d, np.float32)
    out_idx = rng.choice(d, outliers, replace=False)
    ch_scale[out_idx] = rng.uniform(10, 20, outliers)
    ch_mean = rng.normal(0, 0.5, d).astype(np.float32)
    ch_mean[out_idx] *= 8
    K = rng.standard_normal((n, d)).astype(np.float32) * ch_scale + ch_mean
    vscale = np.where(np.isin(np.arange(d), out_idx[:2]), 4.0, 1.0).astype(np.float32)
    V = rng.standard_normal((n, d)).astype(np.float32) * vscale
    targets = rng.integers(0, n, nq)
    Kc = K - K.mean(0)
    Q = 0.35 * Kc[targets] / ch_scale + rng.standard_normal((nq, d)).astype(np.float32) * 0.5
    return Q.astype(np.float32), K, V


def softmax(s):
    s = s - s.max(1, keepdims=True)
    p = np.exp(s)
    return p / p.sum(1, keepdims=True)


def evaluate(Q, K, V, spec, noise_aware=False, seed=0):
    d = Q.shape[1]
    r = compress_kv(K, V, spec, seed=seed)
    s_ref = Q @ K.T / np.sqrt(d)
    p_ref = softmax(s_ref)
    o_ref = p_ref @ V
    s = Q @ r.K.T / np.sqrt(d)
    if noise_aware:
        s = nat_logits(s, logit_noise_var(Q, r.K, r.k_resid_sq, 1 / np.sqrt(d)))
    p = softmax(s)
    o = p @ r.V
    kl = np.sum(p_ref * (np.log(p_ref + 1e-30) - np.log(p + 1e-30)), axis=1).mean()
    # logit "temperature": slope of estimated vs true logits (1.0 = no shrinkage)
    sc, rc = s - s.mean(1, keepdims=True), s_ref - s_ref.mean(1, keepdims=True)
    return dict(bits=r.bits, out_err=float(np.linalg.norm(o - o_ref) / np.linalg.norm(o_ref)),
                kl=float(kl), top1=float(np.mean(p.argmax(1) == p_ref.argmax(1))),
                slope=float(np.sum(sc * rc) / np.sum(rc * rc)))


def run(n=4096, seeds=(0, 1, 2)):
    rows = []
    for label, spec, na, fam in METHODS:
        ms = [evaluate(*make_cache(n=n, seed=s), spec, na, seed=s) for s in seeds]
        avg = {k: float(np.mean([m[k] for m in ms])) for k in ms[0]}
        rows.append(dict(name=label, family=fam, **avg))
    return rows


def print_table(rows):
    hdr = (f"{'method':<26}{'bits':>6}{'compress':>10}{'out err':>10}{'attn KL':>10}"
           f"{'top-1':>8}{'logit slope':>13}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['name']:<26}{r['bits']:>6.2f}{16 / r['bits']:>9.1f}x{r['out_err']:>10.4f}"
              f"{r['kl']:>10.4f}{r['top1'] * 100:>7.1f}%{r['slope']:>13.3f}")


STYLE = {
    "int": ("#9a9994", "s", "int-b per-token"),
    "kivi": ("#eb6834", "^", "KIVI-style (per-channel keys)"),
    "qjl": ("#6250d6", "D", "TurboQuant-prod (QJL keys)"),
    "mse": ("#1baf7a", "v", "TurboQuant-mse"),
    "cal": ("#2a78d6", "o", "TurboKV: calibrated K+V (ours)"),
    "calna": ("#52514e", "*", "TurboKV + NAT (experimental)"),
}


def plot(rows, path, metric="kl", ylabel="attention KL divergence (log)"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.8, 4.8), dpi=160)
    for fam, (c, mk, label) in STYLE.items():
        pts = sorted((r["bits"], r[metric]) for r in rows if r["family"] == fam)
        if pts:
            xs, ys = zip(*pts)
            ours = fam in ("cal", "calna")
            ax.plot(xs, ys, marker=mk, color=c, label=label, lw=2.2 if ours else 1.2,
                    ms=9 if mk == "*" else 6, alpha=1 if ours else 0.85)
    ax.set_yscale("log")
    ax.set_xlabel("bits per cached number (including scales / norms)")
    ax.set_ylabel(ylabel)
    ax.set_title("KV-cache compression: error vs. memory  (lower-left is better)")
    ax.grid(alpha=0.3, which="both")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(path)
    print(f"saved {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4096, help="cached tokens")
    ap.add_argument("--plot", action="store_true")
    a = ap.parse_args()
    rows = run(n=a.n)
    print_table(rows)
    if a.plot:
        plot(rows, os.path.join(os.path.dirname(__file__), "..", "figures", "synthetic_attention.png"))
