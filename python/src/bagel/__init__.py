"""Bayesian online changepoint detection for linear models.

We specify feature, prior, and transformation functions, then use ``fit``
for a series or ``Detector.update`` for arriving observations. For unknown
observation variance, we use ``variance="unknown", distance="approximate"``.
We read change probabilities, locations, and posterior estimates from the result.
"""
from .detector import Detector
from .api import (
    Result, analyse, fit,
    bagel_kv_exact, bagel_kv_approximate, bagel_uv_exact, bagel_uv_approximate,
    sequential_bagel_kv_exact, sequential_bagel_kv_approximate,
    sequential_bagel_uv_exact, sequential_bagel_uv_approximate,
)

__version__ = "1.8.2"
__all__ = [
    "Detector", "Result", "analyse", "fit",
    "bagel_kv_exact", "bagel_kv_approximate", "bagel_uv_exact", "bagel_uv_approximate",
    "sequential_bagel_kv_exact", "sequential_bagel_kv_approximate",
    "sequential_bagel_uv_exact", "sequential_bagel_uv_approximate",
]
