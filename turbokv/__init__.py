"""TurboKV: near-optimal KV-cache compression with unbiased attention for zero extra bits."""

from .cache import TurboKVCache
from .hadamard import RandomRotation, fwht
from .lloyd_max import gaussian_codebook, gaussian_distortion
from .methods import compress_kv, logit_noise_var, nat_logits
from .quantizer import TurboQuantCal, TurboQuantMSE, TurboQuantProd, variance_factor

__all__ = [
    "TurboKVCache",
    "TurboQuantCal",
    "TurboQuantMSE",
    "TurboQuantProd",
    "RandomRotation",
    "fwht",
    "gaussian_codebook",
    "gaussian_distortion",
    "compress_kv",
    "logit_noise_var",
    "nat_logits",
    "variance_factor",
]
__version__ = "0.2.0"
