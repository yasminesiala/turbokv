import numpy as np
import pytest

from turbokv import (RandomRotation, TurboKVCache, TurboQuantCal, TurboQuantMSE, TurboQuantProd,
                     compress_kv, fwht, gaussian_codebook, gaussian_distortion, logit_noise_var,
                     nat_logits)
from turbokv.quantizer import pack_bits, unpack_bits

rng = np.random.default_rng(0)


# ---------------------------------------------------------------- building blocks


def test_fwht_orthonormal_and_self_inverse():
    x = rng.standard_normal((7, 64)).astype(np.float32)
    y = fwht(x)
    assert np.allclose(np.linalg.norm(y, axis=1), np.linalg.norm(x, axis=1), rtol=1e-5)
    assert np.allclose(fwht(y), x, atol=1e-5)


@pytest.mark.parametrize("d", [64, 100, 128])
def test_rotation_preserves_inner_products(d):
    R = RandomRotation(d, seed=3)
    a, b = rng.standard_normal((2, d)).astype(np.float32)
    assert np.isclose(R.forward(a) @ R.forward(b), a @ b, rtol=1e-4)
    assert np.allclose(R.inverse(R.forward(a)), a, atol=1e-5)


def test_rotation_flattens_spikes():
    d = 256
    x = np.zeros((300, d), np.float32)
    x[np.arange(300), np.arange(300) % d] = 1.0  # all energy in one coordinate
    y = RandomRotation(d, seed=1, rounds=1).forward(x) * np.sqrt(d)
    assert np.allclose(np.abs(y), 1.0, atol=1e-4)  # perfectly spread out


@pytest.mark.parametrize("bits,expected", [(1, 0.3634), (2, 0.1175), (3, 0.03454), (4, 0.009497)])
def test_lloyd_max_matches_known_optimum(bits, expected):
    assert gaussian_distortion(bits) == pytest.approx(expected, rel=2e-3)
    levels, _ = gaussian_codebook(bits)
    assert np.allclose(levels, -levels[::-1], atol=1e-5)


def test_one_bit_codebook_is_sqrt_2_over_pi():
    levels, _ = gaussian_codebook(1)
    assert levels[1] == pytest.approx(np.sqrt(2 / np.pi), rel=1e-5)


@pytest.mark.parametrize("bits", [1, 2, 3, 4, 5])
def test_bit_packing_roundtrip(bits):
    codes = rng.integers(0, 1 << bits, (9, 128)).astype(np.uint8)
    packed = pack_bits(codes, bits)
    assert packed.shape[1] == 128 * bits // 8
    assert np.array_equal(unpack_bits(packed, bits, 128), codes)


@pytest.mark.parametrize("bits", [2, 3, 4])
def test_mse_error_matches_theory(bits):
    d = 128
    x = rng.standard_normal((4000, d)).astype(np.float32)
    q = TurboQuantMSE(d, bits, seed=0)
    err = np.sum((q.dequantize(q.quantize(x)) - x) ** 2, 1) / np.sum(x**2, 1)
    assert err.mean() == pytest.approx(gaussian_distortion(bits), rel=0.1)


# ------------------------------------------------------------- the three results


def _estimates(cls, bits, q, k, seeds=300):
    return np.array([cls(q.shape[1], bits, seed=s).inner_products(q, cls(q.shape[1], bits, seed=s)
                     .quantize(k))[:, 0] for s in range(seeds)])


@pytest.mark.parametrize("bits", [1, 2, 3])
def test_theorem1_mse_shrinks_by_exactly_one_minus_D(bits):
    d = 128
    k = rng.standard_normal((1, d)).astype(np.float32)
    k[0, :3] *= 12  # spiky on purpose
    q = rng.standard_normal((16, d)).astype(np.float32)
    q[:8] += 0.3 * k
    E = _estimates(TurboQuantMSE, bits, q, k)
    true = (q @ k.T)[:, 0]
    slope = np.sum(E.mean(0) * true) / np.sum(true**2)
    assert slope == pytest.approx(1 - gaussian_distortion(bits), abs=0.01)


@pytest.mark.parametrize("bits", [1, 2, 3])
def test_calibrated_is_exact_along_k_and_unbiased(bits):
    d = 128
    k = rng.standard_normal((5, d)).astype(np.float32)
    cq = TurboQuantCal(d, bits, seed=3)
    kh = cq.dequantize(cq.quantize(k))
    assert np.allclose(np.sum(k * kh, 1), np.sum(k * k, 1), rtol=2e-3)  # <k, k_hat> = ||k||^2
    q = (k[:1] + rng.standard_normal((1, d))).astype(np.float32)
    E = _estimates(TurboQuantCal, bits, q, k[:1])[:, 0]
    assert abs(E.mean() - (q @ k[0]).item()) < 4 * E.std() / np.sqrt(len(E))


@pytest.mark.parametrize("bits", [1, 2, 3, 4])
def test_theorem2_calibrated_beats_qjl_variance(bits):
    d = 128
    k = rng.standard_normal((1, d)).astype(np.float32)
    q = rng.standard_normal((8, d)).astype(np.float32)
    v_cal = _estimates(TurboQuantCal, bits, q, k, 200).var(0).mean()
    v_qjl = _estimates(TurboQuantProd, bits, q, k, 200).var(0).mean()
    assert v_qjl / v_cal > 2.0


def test_residual_estimate_from_code_only():
    d = 128
    k = rng.standard_normal((500, d)).astype(np.float32) * rng.uniform(0.5, 3, (500, 1))
    cq = TurboQuantCal(d, 3, seed=0)
    code = cq.quantize(k)
    true = np.sum((cq.dequantize(code) - k) ** 2, 1)
    assert np.corrcoef(true, cq.residual_sq(code))[0, 1] > 0.9
    assert cq.residual_sq(code).mean() == pytest.approx(true.mean(), rel=0.05)


def test_theorem3_nat_reduces_attention_kl():
    d, n = 128, 2048
    K = rng.standard_normal((n, d)).astype(np.float32)
    V = rng.standard_normal((n, d)).astype(np.float32)
    Q = (0.2 * K[rng.integers(0, n, 64)] + 0.5 * rng.standard_normal((64, d))).astype(np.float32)

    def sm(s):
        p = np.exp(s - s.max(1, keepdims=True))
        return p / p.sum(1, keepdims=True)

    pr = sm(Q @ K.T / np.sqrt(d))
    r = compress_kv(K, V, "cal2/fp16")
    s = Q @ r.K.T / np.sqrt(d)
    t = nat_logits(s, logit_noise_var(Q, r.K, r.k_resid_sq, 1 / np.sqrt(d)))

    def kl(p):
        return np.sum(pr * (np.log(pr + 1e-30) - np.log(p + 1e-30)), 1).mean()

    assert kl(sm(t)) < 0.95 * kl(sm(s))


def test_nat_is_identity_without_noise():
    s = rng.standard_normal((4, 50))
    assert np.allclose(nat_logits(s, np.zeros_like(s)), s)


# ------------------------------------------------------------------ integration


def test_cache_attention_close_to_exact():
    d, n = 64, 512
    K = rng.standard_normal((n, d)).astype(np.float32)
    V = rng.standard_normal((n, d)).astype(np.float32)
    Q = (K[rng.integers(0, n, 8)] * 0.5).astype(np.float32)
    cache = TurboKVCache(d, key_bits=4, value_bits=4)
    for i in range(0, n, 64):
        cache.append(K[i : i + 64], V[i : i + 64])
    assert len(cache) == n
    s = Q @ K.T / np.sqrt(d)
    p = np.exp(s - s.max(1, keepdims=True))
    ref = (p / p.sum(1, keepdims=True)) @ V
    out = cache.attend(Q)
    assert np.linalg.norm(out - ref) / np.linalg.norm(ref) < 0.25
    assert cache.nbytes < 0.3 * (2 * n * d * 2)  # >3.3x smaller than fp16


@pytest.mark.parametrize("spec", ["fp16", "int4", "kivi3", "mse43", "qjl33", "cal42", "cal4/kivi2"])
def test_compress_kv_specs(spec):
    K = rng.standard_normal((200, 64)).astype(np.float32)
    V = rng.standard_normal((200, 64)).astype(np.float32)
    r = compress_kv(K, V, spec)
    assert r.K.shape == K.shape and r.V.shape == V.shape and 1.5 < r.bits <= 16
