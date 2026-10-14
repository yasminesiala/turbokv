"""Fast Walsh-Hadamard transform and a randomized orthogonal rotation built on it.

Why this matters: a vector from a real transformer's key cache is *spiky* --
a handful of "outlier channels" carry 10-20x more energy than the rest, which
wrecks any scalar quantizer (the scale gets set by the outliers and every
other channel rounds to zero).

Multiply by H·D (H = normalized Hadamard matrix, D = random ±1 diagonal) and
the energy gets smeared evenly across all d coordinates. For a unit vector,
each rotated coordinate is then ~ N(0, 1/d) almost independently of the input,
so ONE fixed codebook is near-optimal for every vector. And because H·D is
orthogonal, inner products are preserved exactly: <Rq, Rk> = <q, k>.

Cost: O(d log d) per vector, no d x d matrix ever materialized.
"""

from __future__ import annotations

import numpy as np


def next_pow2(n: int) -> int:
    return 1 << (int(n) - 1).bit_length()


def fwht(x: np.ndarray) -> np.ndarray:
    """Orthonormal fast Walsh-Hadamard transform along the last axis.

    The last axis must be a power of two. Self-inverse: fwht(fwht(x)) == x.
    """
    x = np.asarray(x, dtype=np.float32)
    d = x.shape[-1]
    if d & (d - 1):
        raise ValueError(f"last dim must be a power of 2, got {d}")
    lead = x.shape[:-1]
    y = x.reshape(-1, d).copy()
    h = 1
    while h < d:
        y = y.reshape(-1, d // (2 * h), 2, h)
        a, b = y[:, :, 0, :], y[:, :, 1, :]
        y = np.stack((a + b, a - b), axis=2)
        h *= 2
    return (y.reshape(*lead, d) / np.sqrt(d)).astype(np.float32)


class RandomRotation:
    """R = H · diag(s2) · H · diag(s1): a random orthogonal map in O(d log d).

    Two rounds of (random signs, Hadamard) make R behave much more like a
    uniformly random rotation than a single round, which matters for the
    shrinkage theorem in `quantizer.py` (it is exact for a uniform rotation).
    """

    def __init__(self, dim: int, seed: int = 0, rounds: int = 2):
        self.dim = dim
        self.pdim = next_pow2(dim)
        rng = np.random.default_rng(seed)
        self.signs = [rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=self.pdim)
                      for _ in range(rounds)]

    def _pad(self, x: np.ndarray) -> np.ndarray:
        if self.pdim == self.dim:
            return x
        pad = [(0, 0)] * (x.ndim - 1) + [(0, self.pdim - self.dim)]
        return np.pad(x, pad)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """(..., dim) -> (..., pdim)"""
        y = self._pad(np.asarray(x, np.float32))
        for s in self.signs:
            y = fwht(y * s)
        return y

    def inverse(self, y: np.ndarray) -> np.ndarray:
        """(..., pdim) -> (..., dim)"""
        y = np.asarray(y, np.float32)
        for s in reversed(self.signs):
            y = fwht(y) * s
        return y[..., : self.dim]
