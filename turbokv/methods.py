"""One interface for every KV compressor, used by the benchmarks and model evals.

    compress_kv(K, V, "cal4/cal3")  ->  KVResult(K_hat, V_hat, bits, k_resid_sq)

K, V: (tokens, head_dim) float32 for one attention head. Feeding K_hat / V_hat
into ordinary attention reproduces exactly what the compressed cache computes
(for QJL keys, <q, K_hat> is the QJL estimate; for Cal keys, the calibrated one).

Spec grammar: "<key>/<value>", each one of
    fp16 | int{b} (per-token) | kivi{b} (per-channel keys) | mse{b} | qjl{b} | cal{b}
A bare "<x>{b}{b'}" shorthand such as "cal43" means "cal4/cal3".
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from .baselines import fp16, per_channel_int, per_token_int
from .quantizer import TurboQuantCal, TurboQuantMSE, TurboQuantProd

_Q: dict = {}
_CLS = {"mse": TurboQuantMSE, "cal": TurboQuantCal, "qjl": TurboQuantProd}


@dataclass
class KVResult:
    K: np.ndarray
    V: np.ndarray
    bits: float  # average bits per cached number, including all side info
    k_resid_sq: np.ndarray | None  # per-key ||w||^2 estimate (cal keys only)


def parse(spec: str) -> tuple[str, str]:
    if "/" in spec:
        k, v = spec.split("/")
        return k, v
    m = re.fullmatch(r"(mse|cal|qjl)(\d)(\d)", spec)
    if m:
        return m[1] + m[2], m[1] + m[3]
    return spec, spec  # fp16, int4, kivi3 ...


def _quantizer(kind, d, b, seed):
    key = (kind, d, b, seed)
    if key not in _Q:
        _Q[key] = _CLS[kind](d, b, seed=seed)
    return _Q[key]


def _one(X, spec, seed, is_key, calib):
    """Compress one matrix. Returns (X_hat, nbytes, resid_sq|None)."""
    n, d = X.shape
    if spec == "fp16":
        Xh, nb = fp16(X)
        return Xh, nb, None
    kind, b = re.fullmatch(r"([a-z]+)(\d)", spec).groups()
    b = int(b)
    if kind == "int":
        return (*per_token_int(X, b), None)
    if kind == "kivi":
        return (*(per_channel_int(X, b) if is_key else per_token_int(X, b)), None)
    # Rotation-based quantizers. Subtract a fixed per-channel mean first: for keys
    # this only shifts every logit of a query by the same constant (softmax ignores
    # it); for values the attention weights sum to 1 so the mean is added back
    # exactly. Mean is estimated on the first `calib` tokens, stored in fp16.
    mu = X[:calib].mean(0).astype(np.float16).astype(np.float32)
    q = _quantizer(kind, d, b, seed)
    code = q.quantize(X - mu)
    Xh = q.dequantize(code) + mu
    resid = q.residual_sq(code) if kind == "cal" else None
    return Xh.astype(np.float32), code.nbytes + 2 * d, resid


def compress_kv(K: np.ndarray, V: np.ndarray, spec: str, seed: int = 0, calib: int = 64) -> KVResult:
    ks, vs = parse(spec)
    n, d = K.shape
    Kh, kb, resid = _one(K, ks, seed, True, calib)
    Vh, vb, _ = _one(V, vs, seed + 7, False, calib)
    return KVResult(Kh, Vh, (kb + vb) * 8 / (2 * n * d), resid)


def logit_noise_var(Q: np.ndarray, K_hat: np.ndarray, resid_sq: np.ndarray, scale: float):
    """sigma_ij^2: variance of the error in the calibrated logit for (query i, key j).

    k_hat = k + w with w ⟂ k and pointing in a random direction, so the logit
    error scale·<q, w> has variance scale^2 · ||q_perp||^2 · ||w||^2 / (d - 1),
    where q_perp is the part of q orthogonal to k (the only part w can touch).
    """
    d = Q.shape[-1]
    q2 = np.sum(Q * Q, -1)[:, None]
    k2 = np.maximum(np.sum(K_hat * K_hat, -1), 1e-12)[None, :]
    qk = Q @ K_hat.T
    qperp2 = np.maximum(q2 - qk * qk / k2, 0.0)
    return scale * scale * qperp2 * resid_sq[None, :] / (d - 1)


def nat_logits(s: np.ndarray, sig2: np.ndarray, mask: np.ndarray | None = None,
               iters: int = 1, a_min: float = 0.3):
    """Noise-Adaptive Temperature: rescale each query's logits by the KL-optimal a*.

    If s_hat = s + eps (independent, variance sigma_j^2), expanding
    E[KL(softmax(s) || softmax(a·s_hat))] to second order gives
        (1-a)^2 · A  +  ... (a-independent)  ...  and the minimizer
        a* = 1 - A / V,   A = sum_j p_j (1 - p_j) sigma_j^2,
                          V = sum_j p_j (s_hat_j - mean_p s_hat)^2
    (V is the observed attention-weighted logit variance, which equals the true
    one plus A on average). p is unknown, so we plug in softmax of the current
    logits; one iteration is enough. Derivation: README, "What didn't work".
    """
    if mask is not None:
        s = np.where(mask, s, -np.inf)
    t = s
    for _ in range(iters):
        z = t - t.max(-1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(-1, keepdims=True)
        sf = np.where(np.isfinite(s), s, 0.0)
        A = np.sum(p * (1 - p) * sig2, -1, keepdims=True)
        mu = np.sum(p * sf, -1, keepdims=True)
        V = np.sum(p * (sf - mu) ** 2, -1, keepdims=True)
        a = np.clip(1.0 - A / np.maximum(V, 1e-9), a_min, 1.0)
        t = a * s
    return t
