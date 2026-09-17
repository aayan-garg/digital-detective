"""Deterministic pseudo-random baseline ranker.

Shuffles candidate entities deterministically based on case_id seed.
Enforces closed candidate universe with zero omissions.
"""

from __future__ import annotations

import hashlib
import random
from typing import Sequence

from ..models import RankedEntity


def rank_with_random(
    candidate_universe: Sequence[str],
    case_id: str = "",
    seed_override: int | None = None,
) -> tuple[RankedEntity, ...]:
    """Deterministically rank candidate entities using pseudo-random permutation.

    Parameters
    ----------
    candidate_universe:
        Closed sequence of candidate service names.
    case_id:
        Identifier of the case used to derive a deterministic seed.
    seed_override:
        Optional explicit random seed.

    Returns
    -------
    tuple[RankedEntity, ...]
        Deterministically ordered pseudo-random ranking.
    """
    candidates = sorted(set(candidate_universe))
    if seed_override is not None:
        rng = random.Random(seed_override)
    else:
        seed_int = int(hashlib.sha256(case_id.encode("utf-8")).hexdigest()[:8], 16)
        rng = random.Random(seed_int)

    shuffled = list(candidates)
    rng.shuffle(shuffled)

    num_candidates = len(shuffled)
    ranked: list[RankedEntity] = []
    for rank_idx, ent in enumerate(shuffled, start=1):
        # Assign fractional score decreasing with rank
        score = (num_candidates - rank_idx + 1) / float(num_candidates) if num_candidates > 0 else 0.0
        ranked.append(RankedEntity(entity=ent, score=score, rank=rank_idx))

    return tuple(ranked)
