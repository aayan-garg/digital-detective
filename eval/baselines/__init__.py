"""Baseline RCA implementations for Digital Detective Stage 0 & Stage 1 evaluation.
"""

from .random_ranker import rank_with_random
from .simple_rca import rank_with_simple_rca

__all__ = [
    "rank_with_random",
    "rank_with_simple_rca",
]
