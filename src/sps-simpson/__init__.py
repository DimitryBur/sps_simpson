"""SPS-Simpson: Fast SVD approximation for LLM matrices."""

from .core import sps_simpson_svd, randomized_svd_fast, get_error
from .version import __version__

__all__ = [
    "sps_simpson_svd",
    "randomized_svd_fast", 
    "get_error",
    "__version__",
]