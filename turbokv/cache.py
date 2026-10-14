"""A compressed KV cache for one attention head; attention runs on the compressed data.

    cache = TurboKVCache(head_dim=128, key_bits=4, value_bits=3)
    cache.append(k, v)            # (tokens, 128) each, any number of times
    out = cache.attend(q)         # (queries, 128)

Tricks that keep it fast:
  * scores: rotate the query once (O(d log d)) and dot it against codebook
    levels in the rotated basis -- cached keys are never inverse-transformed.
  * outputs: softmax(QK^T) V = R^T [softmax(QK^T) (R V)], so values are mixed
    in the rotated basis and only ONE inverse transform runs per query.
  * keys AND values use the calibrated estimator: unbiased scores and unbiased
    outputs, for zero extra bits. NAT (noise-adaptive temperature) is available
    via noise_aware=True but is experimental: it helps diffuse heads and hurts
    sharp ones (see README, "What didn't work").
"""

from __future__ import annotations

import numpy as np

from .methods import logit_noise_var, nat_logits
from .quantizer import MSECode, TurboQuantCal


def _cat(codes: list[MSECode]) -> MSECode:
    return MSECode(np.concatenate([c.packed for c in codes]),
                   np.concatenate([c.scale for c in codes]), codes[0].bits)


class TurboKVCache:
    def __init__(self, head_dim: int, key_bits: int = 4, value_bits: int = 3,
                 noise_aware: bool = False, seed: int = 0):
        self.head_dim = head_dim
        self.kq = TurboQuantCal(head_dim, key_bits, seed=seed)
        self.vq = TurboQuantCal(head_dim, value_bits, seed=seed + 7)
        self.noise_aware = noise_aware
        self._k: list[MSECode] = []
        self._v: list[MSECode] = []

    def __len__(self) -> int:
        return sum(len(c) for c in self._k)

    @property
    def nbytes(self) -> int:
        return sum(c.nbytes for c in self._k) + sum(c.nbytes for c in self._v)

    def append(self, k: np.ndarray, v: np.ndarray) -> None:
        self._k.append(self.kq.quantize(k))
        self._v.append(self.vq.quantize(v))
        if len(self._k) > 64:  # keep the chunk list short
            self._k, self._v = [_cat(self._k)], [_cat(self._v)]

    def attend(self, q: np.ndarray) -> np.ndarray:
        """softmax(q K^T / sqrt(d)) V for queries q: (nq, d) -> (nq, d)."""
        K, V = _cat(self._k), _cat(self._v)
        d = self.head_dim
        qr = self.kq.rot.forward(np.asarray(q, np.float32).reshape(-1, d))
        Kr = self.kq.dequantize_rotated(K)  # (n, pdim), rotated basis
        s = (qr @ Kr.T) / np.sqrt(d)
        if self.noise_aware:  # rotation preserves norms/dots, so the rotated basis is fine
            sig2 = logit_noise_var(qr, Kr, self.kq.residual_sq(K), 1 / np.sqrt(d))
            s = nat_logits(s, sig2)
        s -= s.max(axis=1, keepdims=True)
        p = np.exp(s)
        p /= p.sum(axis=1, keepdims=True)
        return self.vq.rot.inverse(p @ self.vq.dequantize_rotated(V))
