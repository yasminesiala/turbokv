"""Regenerate every figure in the README.

    python benchmarks/make_figures.py
needs results/tiny_gpt_seed{0,1,2}.json (from examples/eval_perplexity.py --seed N --json ...)
"""

from __future__ import annotations

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
from turbokv import TurboQuantCal, TurboQuantMSE, TurboQuantProd, variance_factor  # noqa: E402
from turbokv.lloyd_max import gaussian_distortion  # noqa: E402

FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)

INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE, ORANGE, AQUA, VIOLET, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#9a9994"

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "axes.titlecolor": INK, "figure.facecolor": SURF, "axes.facecolor": SURF,
    "savefig.facecolor": SURF, "axes.titleweight": "bold", "axes.titlelocation": "left",
})


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color=GRID, lw=0.8, which="major")
    ax.set_axisbelow(True)


# --------------------------------------------------------------- 1. hero figure


def hero():
    runs = [json.load(open(os.path.join(ROOT, f"results/tiny_gpt_seed{s}.json")))["results"]
            for s in range(3)]
    base = runs[0][0]["ppl"]
    series = [  # (label prefix in results, legend name, color, marker, emphasis)
        ("int per-token", "int per-token", GRAY, "s", False),
        ("KIVI-style", "KIVI-style", ORANGE, "^", False),
        ("TurboQuant-mse", "TurboQuant (MSE)", AQUA, "v", False),
        ("TurboQuant-prod", "TurboQuant (prod / QJL)", VIOLET, "D", False),
        ("TurboKV (ours)", "TurboKV (ours)", BLUE, "o", True),
    ]
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=170)
    for prefix, name, c, mk, ours in series:
        idx = [i for i, r in enumerate(runs[0]) if r["label"].startswith(prefix + " ")]
        bits = np.array([runs[0][i]["bits"] for i in idx])
        d = np.array([[run[i]["ppl"] - base for i in idx] for run in runs])
        mean, sd = d.mean(0), d.std(0)
        o = np.argsort(bits)
        ax.errorbar(bits[o], mean[o], yerr=sd[o], color=c, marker=mk, ms=8 if ours else 6,
                    lw=2.6 if ours else 1.6, capsize=3, zorder=5 if ours else 3,
                    mec=SURF, mew=1.2, label=name)
        if ours:  # the legend names the rest; one direct label avoids collisions
            x, y = bits[o][0], mean[o][0]
            ax.annotate(f"{name}\n{16 / x:.1f}× smaller, about half the damage", (x, y),
                        xytext=(-10, 0), textcoords="offset points", ha="right", va="center",
                        fontsize=9, color=INK, fontweight="bold")
    ax.set_yscale("log")
    ax.set_xlim(0.9, 4.7)
    ax.set_xlabel("bits per cached number  (16 = uncompressed)")
    ax.set_ylabel("perplexity increase vs. FP16  (log, lower is better)")
    ax.set_title("Damage to a real trained GPT when its KV cache is compressed", pad=12)
    ax.text(0, 1.0, "tiny GPT trained on Python source, 64 held-out windows, mean ± sd over 3 rotation seeds",
            transform=ax.transAxes, fontsize=8.5, color=INK2, va="bottom")
    _style(ax)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "hero_perplexity.png"))
    plt.close(fig)


# ------------------------------------------------- 2. the shrinkage theorem, live


def shrinkage(seeds=300, d=128):
    rng = np.random.default_rng(0)
    k = rng.standard_normal((1, d)).astype(np.float32)
    k[0, :4] *= 12
    q = rng.standard_normal((40, d)).astype(np.float32)
    q[:20] += 0.25 * k
    true = (q @ k.T)[:, 0]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.9), dpi=170, sharey=True)
    for ax, (cls, name, c) in zip(axes, [(TurboQuantMSE, "TurboQuant (MSE)", AQUA),
                                         (TurboQuantCal, "TurboKV calibrated", BLUE)]):
        est = np.array([cls(d, 2, seed=s).inner_products(q, cls(d, 2, seed=s).quantize(k))[:, 0]
                        for s in range(seeds)]).mean(0)
        lim = np.abs(true).max() * 1.1
        ax.plot([-lim, lim], [-lim, lim], color=INK2, lw=1, ls="--", label="perfect (slope 1)")
        if cls is TurboQuantMSE:
            sl = 1 - gaussian_distortion(2)
            ax.plot([-lim, lim], [-lim * sl, lim * sl], color=AQUA, lw=1.2,
                    label=f"theory: slope 1 − D₂ = {sl:.3f}")
        ax.scatter(true, est, s=26, color=c, edgecolor=SURF, lw=0.8, zorder=4,
                   label="average over 300 random rotations")
        fit = np.sum(est * true) / np.sum(true**2)
        ax.set_title(f"{name}: measured slope {fit:.3f}", fontsize=10)
        ax.set_xlabel("true  ⟨q, k⟩")
        _style(ax)
        ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    axes[0].set_ylabel("average estimate")
    fig.suptitle("Theorem 1: MSE quantization shrinks every dot product by exactly (1 − D_b)",
                 x=0.01, ha="left", fontweight="bold", color=INK, fontsize=10.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "shrinkage.png"))
    plt.close(fig)


# ---------------------------------------------- 3. variance: calibrated vs QJL


def variance(seeds=200, d=128):
    rng = np.random.default_rng(1)
    k = rng.standard_normal((1, d)).astype(np.float32)
    k /= np.linalg.norm(k)
    q = rng.standard_normal((32, d)).astype(np.float32)
    q -= (q @ k.T) * k  # q ⟂ k: the case the closed form describes exactly
    bits = [1, 2, 3, 4]
    meas = {"qjl": [], "cal": []}
    for b in bits:
        for key, cls in (("qjl", TurboQuantProd), ("cal", TurboQuantCal)):
            E = np.array([cls(d, b, seed=s).inner_products(q, cls(d, b, seed=s).quantize(k))[:, 0]
                          for s in range(seeds)])
            meas[key].append((E.var(0) * d / np.sum(q**2, 1)).mean())
    fig, ax = plt.subplots(figsize=(7.6, 4.2), dpi=170)
    x = np.arange(len(bits))
    w = 0.36
    for off, key, name, c in ((-w / 2 - 0.01, "qjl", "TurboQuant prod (QJL)", VIOLET),
                              (w / 2 + 0.01, "cal", "TurboKV calibrated (ours)", BLUE)):
        ax.bar(x + off, meas[key], w, color=c, label=name + " — measured", zorder=3)
        th = [variance_factor(key, b) for b in bits]
        ax.scatter(x + off, th, marker="_", s=500, color=INK, lw=1.6, zorder=4)
    ax.scatter([], [], marker="_", s=200, color=INK, label="closed-form theory")
    for i in x:
        ratio = meas["qjl"][i] / meas["cal"][i]
        ax.text(i, meas["qjl"][i] * 1.25, f"{ratio:.1f}× less noise", ha="center", fontsize=8.5,
                color=INK, fontweight="bold")
    ax.set_yscale("log")
    ax.set_xticks(x, [f"{b} bit{'s' if b > 1 else ''}" for b in bits])
    ax.set_ylabel("variance of ⟨q, k̂⟩   (units of ‖q‖²/d, log)")
    ax.set_title("Theorem 2: same bits, both unbiased — calibration has far less noise", pad=10)
    _style(ax)
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "variance.png"))
    plt.close(fig)
    return meas


if __name__ == "__main__":
    hero()
    shrinkage()
    m = variance()
    print("variance (measured):", {k: [round(v, 4) for v in vs] for k, vs in m.items()})
    print("saved figures/ hero_perplexity.png shrinkage.png variance.png")
