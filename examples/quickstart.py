"""30-second tour.   python examples/quickstart.py"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from turbokv import TurboKVCache  # noqa: E402
from turbokv.baselines import per_token_int  # noqa: E402

rng = np.random.default_rng(0)
d, n = 128, 8192

# a nasty KV cache: a few "outlier channels" 15x bigger than the rest
K = rng.standard_normal((n, d)).astype(np.float32)
K[:, :4] *= 15
V = rng.standard_normal((n, d)).astype(np.float32)
# queries that "look for" specific earlier tokens, like retrieval heads do
targets = rng.integers(0, n, 64)
q = (0.12 * K[targets] / K.std(0) ** 0.5 + 0.5 * rng.standard_normal((64, d))).astype(np.float32)

cache = TurboKVCache(head_dim=d, key_bits=3, value_bits=3)
for i in range(0, n, 512):  # stream tokens in, like generation does
    cache.append(K[i : i + 512], V[i : i + 512])

s = q @ K.T / np.sqrt(d)
p = np.exp(s - s.max(1, keepdims=True))
exact = (p / p.sum(1, keepdims=True)) @ V
approx = cache.attend(q)

fp16_bytes = 2 * n * d * 2
print(f"cached tokens:     {len(cache)}")
print(f"fp16 cache:        {fp16_bytes / 1e6:.2f} MB")
print(f"TurboKV cache:     {cache.nbytes / 1e6:.2f} MB  ({fp16_bytes / cache.nbytes:.1f}x smaller)")

# baseline at a similar size: standard 3-bit per-token integer quantization
Ki, _ = per_token_int(K, 3)
Vi, _ = per_token_int(V, 3)
si = q @ Ki.T / np.sqrt(d)
pi = np.exp(si - si.max(1, keepdims=True))
baseline = (pi / pi.sum(1, keepdims=True)) @ Vi


def err(x):
    return np.linalg.norm(x - exact) / np.linalg.norm(exact)


print(f"attention output error, TurboKV 3-bit:    {err(approx):.1%}")
print(f"attention output error, plain int3:       {err(baseline):.1%}")
