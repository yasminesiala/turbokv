"""The quantizers TurboKV has to beat. Each returns (reconstruction, bytes_used)."""

from __future__ import annotations

import numpy as np


def _minmax(x: np.ndarray, bits: int, axis: int) -> np.ndarray:
    lo = x.min(axis=axis, keepdims=True)
    hi = x.max(axis=axis, keepdims=True)
    scale = (hi - lo) / ((1 << bits) - 1)
    scale = np.where(scale == 0, 1.0, scale)
    q = np.clip(np.round((x - lo) / scale), 0, (1 << bits) - 1)
    return (q * scale + lo).astype(np.float32)


def per_token_int(x: np.ndarray, bits: int):
    """Standard asymmetric int-b, one fp16 (scale, zero) pair per vector."""
    n, d = x.shape
    return _minmax(x, bits, axis=1), n * d * bits / 8 + n * 4


def per_channel_int(x: np.ndarray, bits: int, group: int = 128):
    """KIVI-style: keys quantized per channel over groups of `group` tokens.
    Handles outlier channels well, but needs a buffer of tokens per group."""
    n, d = x.shape
    out = np.empty_like(x, dtype=np.float32)
    for s in range(0, n, group):
        out[s : s + group] = _minmax(x[s : s + group], bits, axis=0)
    groups = -(-n // group)
    return out, n * d * bits / 8 + groups * d * 4


def fp16(x: np.ndarray):
    return x.astype(np.float16).astype(np.float32), x.size * 2
