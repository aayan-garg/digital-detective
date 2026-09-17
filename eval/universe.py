"""Deterministic closed candidate universe resolver for Digital Detective evaluations.

Ensures that every method evaluated for the same case ranks over exactly the same
candidate-service universe. No method may silently drop candidates because evidence
is missing or unobserved.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

# Canonical 11 microservices for Google Cloud Microservices Demo / Online Boutique
ONLINE_BOUTIQUE_SERVICES: tuple[str, ...] = (
    "adservice",
    "cartservice",
    "checkoutservice",
    "currencyservice",
    "emailservice",
    "frontend",
    "paymentservice",
    "productcatalogservice",
    "recommendationservice",
    "redis",
    "shippingservice",
)

# Canonical microservices for Weaveworks Sock Shop
SOCK_SHOP_SERVICES: tuple[str, ...] = (
    "carts",
    "catalogue",
    "front-end",
    "orders",
    "payment",
    "queue-master",
    "shipping",
    "user",
)

CANONICAL_UNIVERSES: Mapping[str, tuple[str, ...]] = {
    "ob": ONLINE_BOUTIQUE_SERVICES,
    "re1-ob": ONLINE_BOUTIQUE_SERVICES,
    "re2-ob": ONLINE_BOUTIQUE_SERVICES,
    "re3-ob": ONLINE_BOUTIQUE_SERVICES,
    "online-boutique": ONLINE_BOUTIQUE_SERVICES,
    "ss": SOCK_SHOP_SERVICES,
    "re1-ss": SOCK_SHOP_SERVICES,
    "re2-ss": SOCK_SHOP_SERVICES,
    "re3-ss": SOCK_SHOP_SERVICES,
    "sock-shop": SOCK_SHOP_SERVICES,
}

# Documented exclusion policy:
# External client traffic generators, load drivers, and non-service nodes
EXCLUDED_ENTITIES: frozenset[str] = frozenset(
    {
        "frontend-external",
        "loadgenerator",
        "client",
        "locust",
    }
)

# Standard service name canonicalization aliases
DEFAULT_SERVICE_ALIASES: Mapping[str, str] = {
    "frontendservice": "frontend",
}


def normalize_service_name(name: str, aliases: Mapping[str, str] | None = None) -> str:
    """Normalize service name using configured aliases."""
    alias_map = DEFAULT_SERVICE_ALIASES if aliases is None else aliases
    return alias_map.get(name, name)


def resolve_candidate_universe(
    system_or_dataset: str,
    case: Any | None = None,
    custom_universe: Sequence[str] | None = None,
    exclusion_set: frozenset[str] | None = None,
) -> tuple[str, ...]:
    """Resolve a closed, deterministic candidate service universe for a case.

    Parameters
    ----------
    system_or_dataset:
        System identifier (e.g. 'ob', 'online-boutique', 'ss') or dataset name (e.g. 'RE2-OB').
    case:
        Optional TelemetryCase or object with metrics/graph used to discover services
        if the system is not in CANONICAL_UNIVERSES.
    custom_universe:
        Explicit sequence of candidate service names overriding standard universes.
    exclusion_set:
        Optional set of entities to exclude. Defaults to EXCLUDED_ENTITIES.

    Returns
    -------
    tuple[str, ...]
        Deterministically sorted tuple of candidate entity names.
    """
    if custom_universe is not None:
        return tuple(sorted(set(custom_universe)))

    key = system_or_dataset.strip().lower()
    if key in CANONICAL_UNIVERSES:
        return CANONICAL_UNIVERSES[key]

    exclusions = EXCLUDED_ENTITIES if exclusion_set is None else exclusion_set

    # If case is provided, extract candidate entities from metrics or topology
    discovered: set[str] = set()
    if case is not None:
        if hasattr(case, "metrics") and getattr(case.metrics, "series", None):
            for col in case.metrics.series.keys():
                if "_" in col:
                    prefix = col.split("_", 1)[0]
                    norm = normalize_service_name(prefix)
                    if norm not in exclusions:
                        discovered.add(norm)
        if hasattr(case, "graph") and getattr(case.graph, "entities", None):
            for ent in case.graph.entities.keys():
                norm = normalize_service_name(ent)
                if norm not in exclusions:
                    discovered.add(norm)

    if discovered:
        return tuple(sorted(discovered))

    raise ValueError(
        f"Unable to resolve candidate universe for system/dataset: {system_or_dataset!r}. "
        "No canonical universe registered and case telemetry provided no discoverable services."
    )
