"""Deterministic benchmark manifest generator and loader.

Supports:
- Grouping repeated service/fault combinations so repetitions do not cross partitions (no leakage).
- Partitioning into 'dev', 'val', and 'held_out'.
- Explicitly named smoke-test manifests labeled 'SMOKE / REGRESSION ONLY'.
- Immutable JSON serialization and loading.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class ManifestCase:
    """Entry identifying a single benchmark case within a manifest."""

    case_id: str
    dataset: str
    system: str
    fault: str
    root_cause_service: str
    repetition: int
    partition: str  # "dev" | "val" | "held_out" | "smoke"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> ManifestCase:
        return cls(
            case_id=str(d["case_id"]),
            dataset=str(d["dataset"]),
            system=str(d["system"]),
            fault=str(d["fault"]),
            root_cause_service=str(d["root_cause_service"]),
            repetition=int(d["repetition"]),
            partition=str(d["partition"]),
        )


@dataclass(frozen=True)
class BenchmarkManifest:
    """An immutable manifest defining experimental evaluation splits."""

    manifest_id: str
    description: str
    created_at: str
    is_smoke_test: bool
    cases: tuple[ManifestCase, ...]

    def get_partition(self, partition: str) -> tuple[ManifestCase, ...]:
        """Return cases belonging to a specific partition."""
        return tuple(c for c in self.cases if c.partition == partition)

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_id": self.manifest_id,
            "description": self.description,
            "created_at": self.created_at,
            "is_smoke_test": self.is_smoke_test,
            "total_cases": len(self.cases),
            "cases": [c.to_dict() for c in self.cases],
        }

    def save(self, path: Path | str) -> None:
        """Serialize manifest to a JSON file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path | str) -> BenchmarkManifest:
        """Load manifest from a JSON file."""
        p = Path(path)
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(
            manifest_id=data["manifest_id"],
            description=data["description"],
            created_at=data.get("created_at", ""),
            is_smoke_test=bool(data.get("is_smoke_test", False)),
            cases=tuple(ManifestCase.from_dict(c) for c in data["cases"]),
        )


def create_smoke_re2_ob_rep1_manifest(
    case_rows: Sequence[Mapping[str, Any]],
    manifest_id: str = "re2_ob_rep1_smoke_v1",
) -> BenchmarkManifest:
    """Generate the standard 30-case RE2-OB repetition-1 smoke-test manifest.

    Explicitly flagged as SMOKE / REGRESSION ONLY.
    """
    cases: list[ManifestCase] = []
    # Filter for RE2-OB rep 1
    for r in case_rows:
        dset = str(r.get("dataset") or r.get("suite", "") + "-" + r.get("system", "").upper())
        rep = int(r.get("repetition") or 1)
        system = str(r.get("system") or "ob")
        if (dset in {"RE2-OB", "RE2-ob"} or (r.get("suite") == "RE2" and system == "ob")) and rep == 1:
            cases.append(
                ManifestCase(
                    case_id=str(r.get("case") or r.get("case_id")),
                    dataset="RE2-OB",
                    system=system,
                    fault=str(r.get("fault")),
                    root_cause_service=str(r.get("root_cause_service")),
                    repetition=rep,
                    partition="smoke",
                )
            )

    cases.sort(key=lambda c: c.case_id)
    return BenchmarkManifest(
        manifest_id=manifest_id,
        description="RE2-OB repetition-1, 30 cases — SMOKE / REGRESSION ONLY",
        created_at="2026-09-17T00:00:00Z",
        is_smoke_test=True,
        cases=tuple(cases),
    )


def partition_cases_leak_free(
    case_rows: Sequence[Mapping[str, Any]],
    manifest_id: str,
    description: str,
    dev_ratio: float = 0.50,
    val_ratio: float = 0.25,
    seed: int = 42,
) -> BenchmarkManifest:
    """Generate an experimental partition ensuring repetitions do not cross partitions.

    Group key: (system, root_cause_service, fault).
    All repetitions for a given (system, service, fault) are assigned to the same partition.
    """
    groups: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    for r in case_rows:
        system = str(r.get("system") or "")
        service = str(r.get("root_cause_service") or "")
        fault = str(r.get("fault") or "")
        groups.setdefault((system, service, fault), []).append(r)

    sorted_group_keys = sorted(groups.keys())
    manifest_cases: list[ManifestCase] = []

    for group_key in sorted_group_keys:
        # Deterministic partition assignment via hash
        key_str = f"{group_key[0]}_{group_key[1]}_{group_key[2]}_{seed}"
        h_val = int(hashlib.sha256(key_str.encode("utf-8")).hexdigest()[:8], 16) / float(0xFFFFFFFF)

        if h_val < dev_ratio:
            part = "dev"
        elif h_val < (dev_ratio + val_ratio):
            part = "val"
        else:
            part = "held_out"

        for r in groups[group_key]:
            cid = str(r.get("case") or r.get("case_id"))
            dset = str(r.get("dataset") or "")
            rep = int(r.get("repetition") or 1)
            manifest_cases.append(
                ManifestCase(
                    case_id=cid,
                    dataset=dset,
                    system=group_key[0],
                    fault=group_key[2],
                    root_cause_service=group_key[1],
                    repetition=rep,
                    partition=part,
                )
            )

    manifest_cases.sort(key=lambda c: c.case_id)
    return BenchmarkManifest(
        manifest_id=manifest_id,
        description=description,
        created_at="2026-09-17T00:00:00Z",
        is_smoke_test=False,
        cases=tuple(manifest_cases),
    )
