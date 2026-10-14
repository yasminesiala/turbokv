"""Vector quantizers for KV caches (NumPy only).

    TurboQuantMSE   values; minimizes ||x - x_hat||^2           (Zandieh et al. 2025)
    TurboQuantProd  keys; (b-1)-bit MSE + 1-bit QJL residual     (Zandieh et al. 2025)
    TurboQuantCal   keys; b-bit MSE + one *calibrated* scale     (TurboKV)

----------------------------------------------------------------------------
The new idea (TurboQuantCal): unbiased inner products for zero extra bits.
----------------------------------------------------------------------------
Shrinkage theorem. Rotate a unit vector y = R x with R a uniformly random
rotation, quantize each coordinate with the Lloyd-Max codebook, rotate back.
Then, averaged over R,

        E[ x_hat ] = (1 - D_b) · x                     (exactly)

so every inner product <q, x_hat> is shrunk by the factor (1 - D_b) on
average: 0.64 at 1 bit, 0.88 at 2 bits, 0.965 at 3 bits. In attention that
divides every logit by the same number, i.e. it *raises the softmax
temperature* and blurs attention.

  Proof sketch: (a) symmetry: for any rotation U that fixes x, U·x_hat has the
  same distribution as x_hat (because R·U is just as random as R), so the
  average of x_hat is fixed by every such U and must point along x: E[x_hat] = c·x.
  (b) size: c = E[<x, x_hat>] = sum_i E[y_i Q(y_i)] = sum_i E[Q(y_i)^2] by the
  centroid property, which is 1 - D_b.  (Proof sketch: README, "The math")

TurboQuant spends 1 bit/coordinate on a random sign sketch (QJL) of the
residual to undo this bias. We instead store ONE number per vector -- in
place of the norm that every scheme stores anyway --

        alpha = ||x||^2 / <x, R^T c>          (c = decoded codebook levels)

and reconstruct k_hat = alpha · R^T c. Now <x, k_hat> = ||x||^2 exactly, i.e.
k_hat = x + w with the error w orthogonal to x, and by the same reflection
symmetry E[w] = 0: the estimator <q, k_hat> is unbiased *for every vector*,
not just on average. (The per-vector scale is the estimator used by RaBitQ,
Gao & Long 2024; here it is paired with optimal multi-bit Lloyd-Max codes.)

Variance comparison, unit vectors, d dims, in units of ||q||^2 / d:
                     1 bit     2 bits    3 bits    4 bits
  QJL (TurboQuant)   1.571     0.571     0.185     0.054     = (pi/2) D_{b-1}
  calibrated         0.571     0.133     0.036     0.0096    = D_b / (1 - D_b)
  -> 2.8x to 5.7x lower variance at the same bit budget.
The QJL bit buys unbiasedness at the cost of one bit of precision; the
calibrated scale gets unbiasedness for free and keeps all b bits for the code.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .hadamard import RandomRotation
from .lloyd_max import gaussian_codebook

# ----------------------------------------------------------------- bit packing


def pack_bits(codes: np.ndarray, bits: int) -> np.ndarray:
    """(n, d) uint8 codes in [0, 2^bits) -> (n, ceil(d*bits/8)) uint8."""
    n, d = codes.shape
    planes = (codes[..., None] >> np.arange(bits, dtype=np.uint8)) & 1
    return np.packbits(planes.reshape(n, d * bits), axis=-1)


def unpack_bits(packed: np.ndarray, bits: int, d: int) -> np.ndarray:
    n = packed.shape[0]
    planes = np.unpackbits(packed, axis=-1, count=d * bits).reshape(n, d, bits)
    return (planes << np.arange(bits, dtype=np.uint8)).sum(-1).astype(np.uint8)


# ------------------------------------------------------------------ containers


@dataclass
class MSECode:
    packed: np.ndarray  # (n, d*bits/8) uint8, bit-packed codebook indices
    scale: np.ndarray  # (n,) float16: the norm (MSE) or calibrated alpha (Cal)
    bits: int

    @property
    def nbytes(self) -> int:
        return self.packed.nbytes + self.scale.nbytes

    def __len__(self) -> int:
        return self.scale.shape[0]


@dataclass
class ProdCode:
    mse: MSECode | None  # None when bits == 1 (pure QJL)
    signs: np.ndarray  # (n, m/8) uint8 packed QJL sign bits
    gammas: np.ndarray  # (n,) float16 residual norms
    norms: np.ndarray  # (n,) float16 vector norms

    @property
    def nbytes(self) -> int:
        inner = self.mse.packed.nbytes if self.mse is not None else 0
        return inner + self.signs.nbytes + self.gammas.nbytes + self.norms.nbytes

    def __len__(self) -> int:
        return self.norms.shape[0]


# -------------------------------------------------------------- base: rotation


class _Rotated:
    def __init__(self, dim: int, bits: int, seed: int):
        self.dim, self.bits = dim, bits
        self.rot = RandomRotation(dim, seed)
        self.pdim = self.rot.pdim
        self._sqrt_p = math.sqrt(self.pdim)  # unit-vector coords ~ N(0, 1/pdim)
        if bits > 0:
            self.levels, self.thresholds = gaussian_codebook(bits)

    def _unit_rotated(self, x):
        x = np.asarray(x, np.float32).reshape(-1, self.dim)
        n = np.linalg.norm(x, axis=-1)
        return self.rot.forward(x / np.maximum(n, 1e-12)[:, None]), n

    def _encode(self, y):
        return np.searchsorted(self.thresholds, y * self._sqrt_p).astype(np.uint8)

    def _decode(self, idx):
        return self.levels[idx] / self._sqrt_p

    def _codes(self, code: MSECode):
        return self._decode(unpack_bits(code.packed, self.bits, self.pdim))


# ------------------------------------------------------------------ quantizers


class TurboQuantMSE(_Rotated):
    """Rotate -> Lloyd-Max per coordinate -> store codes + norm. MSE-optimal, biased."""

    def __init__(self, dim: int, bits: int, seed: int = 0):
        super().__init__(dim, bits, seed)

    def quantize(self, x: np.ndarray) -> MSECode:
        y, n = self._unit_rotated(x)
        return MSECode(pack_bits(self._encode(y), self.bits), n.astype(np.float16), self.bits)

    def dequantize_rotated(self, code: MSECode) -> np.ndarray:
        """Reconstruction in the ROTATED basis (n, pdim); skips the inverse transform."""
        return self._codes(code) * code.scale.astype(np.float32)[:, None]

    def dequantize(self, code: MSECode) -> np.ndarray:
        return self.rot.inverse(self.dequantize_rotated(code))

    def inner_products(self, q: np.ndarray, code: MSECode) -> np.ndarray:
        qr = self.rot.forward(np.asarray(q, np.float32).reshape(-1, self.dim))
        return qr @ self.dequantize_rotated(code).T


class TurboQuantCal(TurboQuantMSE):
    """Same codes as TurboQuantMSE, but the stored scale is calibrated so that
    <x, k_hat> = ||x||^2 exactly. Unbiased inner products, zero extra bits."""

    def __init__(self, dim: int, bits: int, seed: int = 0):
        super().__init__(dim, bits, seed)
        from .lloyd_max import gaussian_distortion

        self.D = gaussian_distortion(bits)

    def quantize(self, x: np.ndarray) -> MSECode:
        y, n = self._unit_rotated(x)
        idx = self._encode(y)
        c = self._decode(idx)
        dot = np.sum(y * c, axis=-1)  # = <x/||x||, R^T c>, always > 0
        alpha = n / np.maximum(dot, 1e-12)
        return MSECode(pack_bits(idx, self.bits), alpha.astype(np.float16), self.bits)

    def residual_sq(self, code: MSECode) -> np.ndarray:
        """Estimated ||w||^2 = ||k_hat - x||^2 from the code alone (no extra storage).

        k_hat = x + w with w ⟂ x (Pythagoras): ||k_hat||^2 = ||x||^2 + ||w||^2.
        On average ||x||^2 = (1 - D_b) ||k_hat||^2, hence ||w||^2 ≈ D_b ||k_hat||^2.
        (Using the known constant D_b is far less noisy than estimating it per vector:
        correlation with the true value ~0.95.)
        """
        c2 = np.sum(self._codes(code) ** 2, axis=-1)
        a = code.scale.astype(np.float32)
        return self.D * a * a * c2


class TurboQuantProd(_Rotated):
    """TurboQuant_prod: (b-1)-bit MSE code + 1-bit QJL sketch of the residual.

    QJL: s = sign(S r) with S ~ N(0,1)^{m x d}. For Gaussian g,
    E[<g,q> sign(<g,r>)] = sqrt(2/pi) <q,r>/||r||, so
    <q,r> ≈ sqrt(pi/2)/m · ||r|| · <S q, s> is unbiased.
    """

    def __init__(self, dim: int, bits: int, seed: int = 0, sketch_dim: int | None = None):
        if bits < 1:
            raise ValueError("bits must be >= 1")
        super().__init__(dim, bits - 1, seed)
        self.total_bits = bits
        self.m = sketch_dim or self.pdim
        rng = np.random.default_rng(seed + 1)
        self.S = rng.standard_normal((self.m, self.pdim)).astype(np.float32)
        self._c = math.sqrt(math.pi / 2.0) / self.m

    def quantize(self, x: np.ndarray) -> ProdCode:
        y, n = self._unit_rotated(x)
        if self.bits > 0:
            idx = self._encode(y)
            r = y - self._decode(idx)
            mse = MSECode(pack_bits(idx, self.bits), np.zeros(0, np.float16), self.bits)
        else:
            r, mse = y, None
        return ProdCode(
            mse=mse,
            signs=np.packbits((r @ self.S.T) >= 0, axis=-1),
            gammas=np.linalg.norm(r, axis=-1).astype(np.float16),
            norms=n.astype(np.float16),
        )

    def _signs(self, code: ProdCode) -> np.ndarray:
        return np.unpackbits(code.signs, axis=-1, count=self.m).astype(np.float32) * 2.0 - 1.0

    def dequantize_rotated(self, code: ProdCode) -> np.ndarray:
        y = self._c * (self._signs(code) @ self.S) * code.gammas.astype(np.float32)[:, None]
        if code.mse is not None:
            y = y + self._codes(code.mse)
        return y * code.norms.astype(np.float32)[:, None]

    def dequantize(self, code: ProdCode) -> np.ndarray:
        """Unbiased reconstruction: <q, dequantize(code)> is the QJL estimate."""
        return self.rot.inverse(self.dequantize_rotated(code))

    def inner_products(self, q: np.ndarray, code: ProdCode) -> np.ndarray:
        qr = self.rot.forward(np.asarray(q, np.float32).reshape(-1, self.dim))
        return qr @ self.dequantize_rotated(code).T


# ----------------------------------------------------------- theory constants


def variance_factor(method: str, bits: int) -> float:
    """Predicted Var[<q,k_hat>] for unit k, in units of ||q||^2/d (see module doc)."""
    from .lloyd_max import gaussian_distortion as D

    if method == "qjl":
        return (math.pi / 2) * (D(bits - 1) if bits > 1 else 1.0)
    if method == "cal":
        return D(bits) / (1 - D(bits))
    if method == "mse":  # variance only; MSE additionally has bias -D_b<q,k>
        return D(bits) * (1 - D(bits))
    raise ValueError(method)
