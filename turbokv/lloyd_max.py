"""Lloyd-Max: the MSE-optimal scalar quantizer for a standard Gaussian.

After the random rotation every coordinate is ~ N(0, 1) (once rescaled), so
instead of uniform int-b buckets we use the 2^b levels that *minimize* the
expected squared error for exactly that distribution. Lloyd's algorithm
alternates two optimality conditions until they agree:

  1. each decision threshold is the midpoint of its two neighbouring levels
  2. each level is the average (conditional mean) of N(0,1) inside its cell:
         E[X | a < X < b] = (phi(a) - phi(b)) / (Phi(b) - Phi(a))

Resulting distortion D_b = E[(X - Q(X))^2] for X ~ N(0,1):
  1 bit: 0.3634 (= 1 - 2/pi)   2 bits: 0.1175   3 bits: 0.03454   4 bits: 0.009497

Condition 2 has a consequence that this whole repo leans on (the "centroid
property"): the error X - Q(X) averages to zero inside every cell, so it is
uncorrelated with Q(X). Hence E[X·Q(X)] = E[Q(X)^2] = 1 - D_b.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

_SQRT2 = math.sqrt(2.0)
_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)
_erf = np.vectorize(math.erf)


def _pdf(x: np.ndarray) -> np.ndarray:
    return _INV_SQRT_2PI * np.exp(-0.5 * x * x)


def _cdf(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + _erf(x / _SQRT2))


def _inv_cdf(p: float) -> float:
    lo, hi = -10.0, 10.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if 0.5 * (1 + math.erf(mid / _SQRT2)) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


@lru_cache(maxsize=None)
def gaussian_codebook(bits: int, iters: int = 500) -> tuple[np.ndarray, np.ndarray]:
    """Return (levels, thresholds) for the optimal `bits`-bit N(0,1) quantizer."""
    if not 1 <= bits <= 8:
        raise ValueError("bits must be in [1, 8]")
    k = 1 << bits
    levels = np.array([_inv_cdf((i + 0.5) / k) for i in range(k)])  # good starting guess
    for _ in range(iters):
        t = 0.5 * (levels[1:] + levels[:-1])
        a = np.concatenate(([-np.inf], t))
        b = np.concatenate((t, [np.inf]))
        new = (_pdf(a) - _pdf(b)) / np.maximum(_cdf(b) - _cdf(a), 1e-300)
        done = np.max(np.abs(new - levels)) < 1e-12
        levels = new
        if done:
            break
    thresholds = 0.5 * (levels[1:] + levels[:-1])
    return levels.astype(np.float32), thresholds.astype(np.float32)


@lru_cache(maxsize=None)
def gaussian_distortion(bits: int) -> float:
    """D_b: expected squared error of the codebook on N(0,1) (numerical integral)."""
    levels, thr = gaussian_codebook(bits)
    x = np.linspace(-10, 10, 400_001)
    idx = np.searchsorted(thr, x)
    return float(np.sum((x - levels[idx]) ** 2 * _pdf(x)) * (x[1] - x[0]))
