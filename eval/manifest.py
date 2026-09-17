"""Deterministic benchmark manifest generator, loader, and validator.

Supports three explicit manifest types:
1. SMOKE_REGRESSION:
   - Rapid regression verification of frozen research baselines.
   - Example: 30-case RE2-OB repetition-1 manifest.
   - Explicitly flagged as SMOKE / REGRESSION ONLY with partition='smoke'.
2. SCIENTIFIC_BENCHMARK:
   - Evaluates fixed, deterministic methods across full declared populations (RE2, RE2-OB, RE2-SS, RE2-TT).
   - Complete unpartitioned population (partition='all').
   - Contains modality availability summary (metrics, logs, traces) based on native telemetry.
3. TUNING_SPLIT:
   - Leakage-controlled experimental partitions ('dev', 'val', 'held_out').
   - Enforces grouping rule: scenario_family = (suite, system, root_cause_service, fault).
   - All repetitions belonging to the same scenario family must remain in the same partition.
   - Explicitly designated as project-defined experimental tuning split (not official RCAEval split).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

VALID_MANIFEST_TYPES = {"SMOKE_REGRESSION", "SCIENTIFIC_BENCHMARK", "TUNING_SPLIT"}


def infer_manifest_type(is_smoke_test: bool, partitions: set[str]) -> str:
    """Infer manifest type from smoke status and case partitions."""
    if is_smoke_test:
        return "SMOKE_REGRESSION"
    if partitions.issubset({"dev", "val", "held_out"}):
        return "TUNING_SPLIT"
    return "SCIENTIFIC_BENCHMARK"


@dataclass(frozen=True)
class ManifestCase:
    """Entry identifying a single benchmark case within a manifest."""

    case_id: str
    dataset: str
    system: str
    fault: str
    root_cause_service: str
    repetition: int
    partition: str  # "dev" | "val" | "held_out" | "smoke" | "all"
    suite: str = ""

    def __post_init__(self) -> None:
        if not self.suite:
            derived_suite = self.dataset.split("-")[0] if "-" in self.dataset else self.dataset
            object.__setattr__(self, "suite", derived_suite)

    @property
    def scenario_family(self) -> tuple[str, str, str, str]:
        """Scenario family identity: (suite, system, root_cause_service, fault)."""
        return (self.suite, self.system, self.root_cause_service, self.fault)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> ManifestCase:
        suite = str(d.get("suite") or "")
        dataset = str(d["dataset"])
        if not suite:
            suite = dataset.split("-")[0] if "-" in dataset else dataset
        return cls(
            case_id=str(d["case_id"]),
            dataset=dataset,
            system=str(d["system"]),
            fault=str(d["fault"]),
            root_cause_service=str(d["root_cause_service"]),
            repetition=int(d["repetition"]),
            partition=str(d["partition"]),
            suite=suite,
        )


@dataclass(frozen=True)
class BenchmarkManifest:
    """An immutable manifest defining experimental evaluation splits."""

    manifest_id: str
    description: str
    created_at: str
    is_smoke_test: bool
    cases: tuple[ManifestCase, ...]
    manifest_type: str = "SMOKE_REGRESSION"
    dataset: str = ""
    modality_summary: dict[str, int] = field(default_factory=dict)
    partition_statistics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        parts = {c.partition for c in self.cases}
        inferred = infer_manifest_type(self.is_smoke_test, parts)
        if self.manifest_type not in VALID_MANIFEST_TYPES or (self.manifest_type != inferred and not self.is_smoke_test):
            object.__setattr__(self, "manifest_type", inferred)
        elif self.is_smoke_test and self.manifest_type != "SMOKE_REGRESSION":
            object.__setattr__(self, "manifest_type", "SMOKE_REGRESSION")
        self.validate()

    def compute_hash(self) -> str:
        """Compute deterministic SHA-256 hash of manifest cases and attributes.

        Contributing components:
        1. Header: manifest_id, manifest_type, is_smoke_test, total_cases
        2. Sorted Case Records (ordered deterministically by case_id):
           case_id, dataset, system, fault, root_cause_service, repetition, partition
        """
        hasher = hashlib.sha256()
        header = f"{self.manifest_id}:{self.manifest_type}:{self.is_smoke_test}:{len(self.cases)}\n"
        hasher.update(header.encode("utf-8"))
        for c in sorted(self.cases, key=lambda x: x.case_id):
            line = f"{c.case_id}|{c.dataset}|{c.system}|{c.fault}|{c.root_cause_service}|{c.repetition}|{c.partition}\n"
            hasher.update(line.encode("utf-8"))
        return hasher.hexdigest()

    def validate(self) -> None:
        """Validate manifest integrity constraints:
        1. Duplicate case IDs are rejected across all manifest types.
        2. SMOKE_REGRESSION:
           - is_smoke_test must be True.
           - description must explicitly indicate smoke/regression status.
           - all cases must have partition='smoke'.
        3. SCIENTIFIC_BENCHMARK:
           - is_smoke_test must be False.
           - description cannot indicate smoke/regression.
           - cases cannot have partition='smoke' (must be 'all', 'benchmark', or 'evaluation').
        4. TUNING_SPLIT:
           - is_smoke_test must be False.
           - description cannot indicate smoke/regression.
           - cases must have valid partition in {'dev', 'val', 'held_out'}.
           - Repetition-family leak freedom: For each scenario_family
             (suite, system, root_cause_service, fault), the cardinality of
             assigned partitions must be EXACTLY 1.
        """
        seen_case_ids: set[str] = set()
        for c in self.cases:
            if c.case_id in seen_case_ids:
                raise ValueError(f"Duplicate case_id {c.case_id!r} in manifest {self.manifest_id!r}.")
            seen_case_ids.add(c.case_id)

        if self.manifest_type not in VALID_MANIFEST_TYPES:
            raise ValueError(
                f"Invalid manifest_type {self.manifest_type!r}. Must be one of {VALID_MANIFEST_TYPES}."
            )

        desc_lower = self.description.lower()
        if self.manifest_type == "SMOKE_REGRESSION":
            if not self.is_smoke_test:
                raise ValueError("SMOKE_REGRESSION manifest must have is_smoke_test=True.")
            if "smoke" not in desc_lower and "regression" not in desc_lower:
                raise ValueError(
                    f"Smoke manifest description must explicitly indicate smoke/regression status: {self.description!r}"
                )
            for c in self.cases:
                if c.partition != "smoke":
                    raise ValueError(
                        f"Smoke manifest case {c.case_id!r} has partition {c.partition!r}. All smoke cases must have partition='smoke'."
                    )
        else:
            if self.is_smoke_test:
                raise ValueError(f"{self.manifest_type} manifest cannot have is_smoke_test=True.")
            if "smoke" in desc_lower or "regression only" in desc_lower:
                raise ValueError(
                    f"{self.manifest_type} manifest cannot be described as smoke/regression: {self.description!r}"
                )

            if self.manifest_type == "SCIENTIFIC_BENCHMARK":
                valid_parts = {"all", "benchmark", "evaluation"}
                for c in self.cases:
                    if c.partition == "smoke":
                        raise ValueError(
                            f"Scientific benchmark cannot contain case {c.case_id!r} with partition='smoke'."
                        )
                    if c.partition not in valid_parts:
                        raise ValueError(
                            f"Scientific benchmark case {c.case_id!r} has invalid partition {c.partition!r}. "
                            f"Must be one of {valid_parts}."
                        )
            elif self.manifest_type == "TUNING_SPLIT":
                valid_partitions = {"dev", "val", "held_out"}
                family_partitions: dict[tuple[str, str, str, str], set[str]] = {}
                for c in self.cases:
                    if c.partition == "smoke":
                        raise ValueError(
                            f"Tuning split manifest cannot contain case {c.case_id!r} with partition='smoke'."
                        )
                    if not c.partition or c.partition not in valid_partitions:
                        raise ValueError(
                            f"Tuning split case {c.case_id!r} has invalid partition {c.partition!r}. "
                            f"Must be one of {valid_partitions}."
                        )
                    family_partitions.setdefault(c.scenario_family, set()).add(c.partition)

                for family, parts in family_partitions.items():
                    if len(parts) != 1:
                        raise ValueError(
                            f"Repetition-family leakage detected in tuning manifest: scenario family {family} is split across "
                            f"partitions {parts}. Cardinality must be exactly 1."
                        )

    def get_partition(self, partition: str) -> tuple[ManifestCase, ...]:
        """Return cases belonging to a specific partition."""
        return tuple(c for c in self.cases if c.partition == partition)

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_id": self.manifest_id,
            "manifest_type": self.manifest_type,
            "manifest_hash": self.compute_hash(),
            "description": self.description,
            "created_at": self.created_at,
            "is_smoke_test": self.is_smoke_test,
            "dataset": self.dataset,
            "total_cases": len(self.cases),
            "modality_summary": dict(self.modality_summary),
            "partition_statistics": dict(self.partition_statistics),
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
        """Load manifest from a JSON file with integrity checks."""
        p = Path(path)
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        total_cases = data.get("total_cases")
        cases_raw = data.get("cases", [])
        if total_cases is not None and total_cases != len(cases_raw):
            raise ValueError(
                f"Manifest case count mismatch: total_cases={total_cases} but found {len(cases_raw)} cases in {path}."
            )
        is_smoke = bool(data.get("is_smoke_test", False))
        cases = tuple(ManifestCase.from_dict(c) for c in cases_raw)
        m_type = data.get("manifest_type") or infer_manifest_type(is_smoke, {c.partition for c in cases})

        manifest = cls(
            manifest_id=data["manifest_id"],
            description=data["description"],
            created_at=data.get("created_at", ""),
            is_smoke_test=is_smoke,
            manifest_type=m_type,
            dataset=data.get("dataset", ""),
            modality_summary=data.get("modality_summary", {}),
            partition_statistics=data.get("partition_statistics", {}),
            cases=cases,
        )
        if "manifest_hash" in data and data["manifest_hash"]:
            computed = manifest.compute_hash()
            if data["manifest_hash"] != computed:
                raise ValueError(
                    f"Manifest hash mismatch for {path}: expected {data['manifest_hash']}, computed {computed}."
                )
        return manifest


def compute_modality_summary(
    all_case_rows: Sequence[Mapping[str, Any]],
    selected_cases: Sequence[ManifestCase],
) -> dict[str, int]:
    """Compute modality availability counts from raw case metadata."""
    case_ids = {c.case_id for c in selected_cases}
    row_map = {str(r.get("case") or r.get("case_id")): r for r in all_case_rows}

    metrics_avail = 0
    logs_avail = 0
    traces_avail = 0
    all_avail = 0

    for cid in case_ids:
        r = row_map.get(cid, {})
        has_metrics = int(r.get("n_metrics", 0)) > 0
        has_logs = bool(r.get("has_logs", False) or int(r.get("n_logs", 0)) > 0)
        has_traces = bool(r.get("has_traces", False) or int(r.get("n_traces", 0)) > 0)

        metrics_avail += int(has_metrics)
        logs_avail += int(has_logs)
        traces_avail += int(has_traces)
        if has_metrics and has_logs and has_traces:
            all_avail += 1

    total = len(selected_cases)
    return {
        "total_cases": total,
        "metrics_available": metrics_avail,
        "logs_available": logs_avail,
        "traces_available": traces_avail,
        "all_modalities_available": all_avail,
        "missing_metrics": total - metrics_avail,
        "missing_logs": total - logs_avail,
        "missing_traces": total - traces_avail,
    }


def filter_cases_by_dataset(
    case_rows: Sequence[Mapping[str, Any]],
    dataset: str,
) -> list[Mapping[str, Any]]:
    """Filter case rows matching a given dataset or suite name ('RE2', 'RE2-OB', etc.)."""
    target = dataset.upper()
    filtered: list[Mapping[str, Any]] = []
    for r in case_rows:
        row_suite = str(r.get("suite") or "")
        row_dset = str(r.get("dataset") or "")
        row_sys = str(r.get("system") or "").lower()

        if target == "RE2":
            match = (row_suite == "RE2" or row_dset.startswith("RE2"))
        elif target == "RE2-OB":
            match = (row_dset == "RE2-OB" or (row_suite == "RE2" and row_sys == "ob"))
        elif target == "RE2-SS":
            match = (row_dset == "RE2-SS" or (row_suite == "RE2" and row_sys == "ss"))
        elif target == "RE2-TT":
            match = (row_dset == "RE2-TT" or (row_suite == "RE2" and row_sys == "tt"))
        else:
            match = (row_dset.upper() == target)

        if match:
            filtered.append(r)
    return filtered


def create_smoke_re2_ob_rep1_manifest(
    case_rows: Sequence[Mapping[str, Any]],
    manifest_id: str = "re2_ob_rep1_smoke_v1",
) -> BenchmarkManifest:
    """Generate the standard 30-case RE2-OB repetition-1 smoke-test manifest.

    Explicitly flagged as SMOKE / REGRESSION ONLY.
    """
    cases: list[ManifestCase] = []
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
    mod_summary = compute_modality_summary(case_rows, cases)
    return BenchmarkManifest(
        manifest_id=manifest_id,
        manifest_type="SMOKE_REGRESSION",
        description="RE2-OB repetition-1, 30 cases — SMOKE / REGRESSION ONLY",
        created_at="2026-09-17T00:00:00Z",
        is_smoke_test=True,
        dataset="RE2-OB",
        modality_summary=mod_summary,
        cases=tuple(cases),
    )


def create_scientific_manifest(
    case_rows: Sequence[Mapping[str, Any]],
    dataset: str,
    manifest_id: str | None = None,
    description: str | None = None,
) -> BenchmarkManifest:
    """Create a complete, unpartitioned scientific benchmark manifest for a dataset or suite.

    Supported datasets: 'RE2', 'RE2-OB', 'RE2-SS', 'RE2-TT'.
    All cases in the designated population are included with partition='all'.
    """
    matched_rows = filter_cases_by_dataset(case_rows, dataset)
    if not matched_rows:
        raise ValueError(f"No cases found matching dataset/suite {dataset!r} in provided case_rows.")

    cases: list[ManifestCase] = []
    for r in matched_rows:
        cid = str(r.get("case") or r.get("case_id"))
        row_dset = str(r.get("dataset") or "")
        row_sys = str(r.get("system") or "").lower()
        dset_name = row_dset if row_dset else f"RE2-{row_sys.upper()}"
        cases.append(
            ManifestCase(
                case_id=cid,
                dataset=dset_name,
                system=row_sys,
                fault=str(r.get("fault")),
                root_cause_service=str(r.get("root_cause_service")),
                repetition=int(r.get("repetition") or 1),
                partition="all",
            )
        )

    cases.sort(key=lambda c: c.case_id)
    mid = manifest_id or f"{dataset.lower().replace('-', '_')}_scientific_v1"
    desc = description or f"Full scientific benchmark population for {dataset} ({len(cases)} cases)"
    mod_summary = compute_modality_summary(case_rows, cases)

    return BenchmarkManifest(
        manifest_id=mid,
        manifest_type="SCIENTIFIC_BENCHMARK",
        description=desc,
        created_at="2026-09-17T00:00:00Z",
        is_smoke_test=False,
        dataset=dataset,
        modality_summary=mod_summary,
        cases=tuple(cases),
    )


def create_scientific_re2_manifest(
    case_rows: Sequence[Mapping[str, Any]],
    manifest_id: str = "re2_scientific_v1",
) -> BenchmarkManifest:
    """Create complete scientific benchmark manifest for RE2 (270 cases across OB, SS, TT)."""
    return create_scientific_manifest(case_rows, dataset="RE2", manifest_id=manifest_id)


def create_scientific_re2_ob_manifest(
    case_rows: Sequence[Mapping[str, Any]],
    manifest_id: str = "re2_ob_scientific_v1",
) -> BenchmarkManifest:
    """Create complete scientific benchmark manifest for RE2-OB (90 cases)."""
    return create_scientific_manifest(case_rows, dataset="RE2-OB", manifest_id=manifest_id)


def create_scientific_re2_ss_manifest(
    case_rows: Sequence[Mapping[str, Any]],
    manifest_id: str = "re2_ss_scientific_v1",
) -> BenchmarkManifest:
    """Create complete scientific benchmark manifest for RE2-SS (90 cases)."""
    return create_scientific_manifest(case_rows, dataset="RE2-SS", manifest_id=manifest_id)


def create_scientific_re2_tt_manifest(
    case_rows: Sequence[Mapping[str, Any]],
    manifest_id: str = "re2_tt_scientific_v1",
) -> BenchmarkManifest:
    """Create complete scientific benchmark manifest for RE2-TT (90 cases)."""
    return create_scientific_manifest(case_rows, dataset="RE2-TT", manifest_id=manifest_id)


def partition_cases_leak_free(
    case_rows: Sequence[Mapping[str, Any]],
    manifest_id: str,
    description: str,
    dev_ratio: float = 0.50,
    val_ratio: float = 0.25,
    seed: int = 42,
    dataset: str = "",
) -> BenchmarkManifest:
    """Generate a leak-free tuning manifest ensuring repetitions do not cross partitions.

    Grouping identity: scenario_family = (suite, system, root_cause_service, fault).
    All repetitions for a given scenario family are assigned to the same partition.

    NOTE: RCAEval does not supply an official Digital Detective train/validation/test split;
    any split created here is a project-defined experimental tuning split.
    """
    if dev_ratio < 0 or val_ratio < 0 or (dev_ratio + val_ratio) > 1.0:
        raise ValueError(
            f"Invalid split ratios: dev_ratio={dev_ratio}, val_ratio={val_ratio}. Sum must be <= 1.0."
        )

    groups: dict[tuple[str, str, str, str], list[Mapping[str, Any]]] = {}
    seen_cids: set[str] = set()

    for r in case_rows:
        cid = str(r.get("case") or r.get("case_id"))
        if cid in seen_cids:
            raise ValueError(f"Duplicate case_id {cid!r} in case_rows input.")
        seen_cids.add(cid)

        suite = str(
            r.get("suite")
            or (str(r.get("dataset", "")).split("-")[0] if "-" in str(r.get("dataset", "")) else r.get("dataset", ""))
        )
        system = str(r.get("system") or "").lower()
        service = str(r.get("root_cause_service") or "")
        fault = str(r.get("fault") or "")
        scenario_family = (suite, system, service, fault)
        groups.setdefault(scenario_family, []).append(r)

    sorted_group_keys = sorted(groups.keys())
    n_groups = len(sorted_group_keys)

    # Deterministic shuffling of groups using seed
    rng = random.Random(seed)
    shuffled_groups = list(sorted_group_keys)
    rng.shuffle(shuffled_groups)

    n_dev = int(round(n_groups * dev_ratio))
    n_val = int(round(n_groups * val_ratio))

    dev_groups = set(shuffled_groups[:n_dev])
    val_groups = set(shuffled_groups[n_dev : n_dev + n_val])
    held_out_groups = set(shuffled_groups[n_dev + n_val:])

    partition_group_sets = {
        "dev": dev_groups,
        "val": val_groups,
        "held_out": held_out_groups,
    }

    manifest_cases: list[ManifestCase] = []
    partition_case_counts = {"dev": 0, "val": 0, "held_out": 0}

    for group_key in sorted_group_keys:
        if group_key in dev_groups:
            part = "dev"
        elif group_key in val_groups:
            part = "val"
        else:
            part = "held_out"

        for r in groups[group_key]:
            cid = str(r.get("case") or r.get("case_id"))
            dset = str(r.get("dataset") or (f"{group_key[0]}-{group_key[1].upper()}" if group_key[0] else ""))
            rep = int(r.get("repetition") or 1)
            manifest_cases.append(
                ManifestCase(
                    case_id=cid,
                    dataset=dset,
                    system=group_key[1],
                    fault=group_key[3],
                    root_cause_service=group_key[2],
                    repetition=rep,
                    partition=part,
                    suite=group_key[0],
                )
            )
            partition_case_counts[part] += 1

    manifest_cases.sort(key=lambda c: c.case_id)
    total_cases = len(manifest_cases)

    part_stats: dict[str, Any] = {
        "total_groups": n_groups,
        "total_cases": total_cases,
        "seed": seed,
        "configured_ratios": {
            "dev": dev_ratio,
            "val": val_ratio,
            "held_out": round(1.0 - dev_ratio - val_ratio, 4),
        },
    }
    for p_name, g_set in partition_group_sets.items():
        c_count = partition_case_counts[p_name]
        part_stats[p_name] = {
            "groups": len(g_set),
            "cases": c_count,
            "group_ratio": round(len(g_set) / n_groups) if n_groups == 0 else round(len(g_set) / n_groups, 4),
            "case_ratio": round(c_count / total_cases) if total_cases == 0 else round(c_count / total_cases, 4),
        }

    mod_summary = compute_modality_summary(case_rows, manifest_cases)

    return BenchmarkManifest(
        manifest_id=manifest_id,
        manifest_type="TUNING_SPLIT",
        description=description,
        created_at="2026-09-17T00:00:00Z",
        is_smoke_test=False,
        dataset=dataset,
        modality_summary=mod_summary,
        partition_statistics=part_stats,
        cases=tuple(manifest_cases),
    )


def main() -> None:
    """CLI for generating and inspecting benchmark manifests."""
    import argparse
    import os
    import pyarrow.parquet as pq

    parser = argparse.ArgumentParser(description="Digital Detective Manifest Tool")
    parser.add_argument(
        "--dataset-root",
        default=os.path.expanduser("~/.cache/rcaeval_validation"),
        help="Path to RCAEval validation cache",
    )
    parser.add_argument(
        "--manifest-type",
        choices=["smoke", "scientific", "tuning"],
        default="scientific",
        help="Type of manifest to generate ('smoke', 'scientific', 'tuning')",
    )
    parser.add_argument(
        "--dataset",
        default="re2-ob",
        help="Dataset or suite name ('re2', 're2-ob', 're2-ss', 're2-tt')",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for tuning splits (default: 42)",
    )
    parser.add_argument(
        "--dev-ratio",
        type=float,
        default=0.50,
        help="Development split ratio for tuning manifests (default: 0.50)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.25,
        help="Validation split ratio for tuning manifests (default: 0.25)",
    )
    parser.add_argument(
        "--output",
        "--manifest-output",
        default=None,
        help="Path to write generated manifest JSON",
    )
    parser.add_argument(
        "--inspect",
        default=None,
        help="Path to existing manifest JSON to inspect",
    )

    args = parser.parse_args()

    if args.inspect:
        manifest = BenchmarkManifest.load(args.inspect)
        print("=" * 60)
        print("MANIFEST INSPECTION")
        print("=" * 60)
        print(f"ID:           {manifest.manifest_id}")
        print(f"Type:         {manifest.manifest_type}")
        print(f"Description:  {manifest.description}")
        print(f"Hash:         {manifest.compute_hash()}")
        print(f"Dataset:      {manifest.dataset}")
        print(f"Total Cases:  {len(manifest.cases)}")
        if manifest.modality_summary:
            print("Modality Summary:")
            for k, v in manifest.modality_summary.items():
                print(f"  {k}: {v}")
        if manifest.partition_statistics:
            print("Partition Statistics:")
            for p_name, p_stat in manifest.partition_statistics.items():
                if isinstance(p_stat, dict):
                    print(f"  {p_name}: {p_stat}")
        return

    root_path = Path(args.dataset_root)
    cases_file = root_path / "cases.parquet"
    if not cases_file.is_file():
        raise FileNotFoundError(f"cases.parquet not found at {cases_file}")
    cases_table = pq.read_table(cases_file)
    case_rows = cases_table.to_pylist()

    dset_norm = args.dataset.strip()

    if args.manifest_type == "smoke":
        manifest = create_smoke_re2_ob_rep1_manifest(case_rows)
    elif args.manifest_type == "scientific":
        manifest = create_scientific_manifest(case_rows, dataset=dset_norm)
    elif args.manifest_type == "tuning":
        filtered_rows = filter_cases_by_dataset(case_rows, dset_norm)
        manifest_id = f"{dset_norm.lower().replace('-', '_')}_tuning_seed{args.seed}_v1"
        desc = (
            f"Experimental tuning split for {dset_norm.upper()} "
            f"(seed={args.seed}, dev={args.dev_ratio}, val={args.val_ratio}) "
            "— PROJECT-DEFINED TUNING SPLIT ONLY"
        )
        manifest = partition_cases_leak_free(
            filtered_rows,
            manifest_id=manifest_id,
            description=desc,
            dev_ratio=args.dev_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed,
            dataset=dset_norm.upper(),
        )

    print("=" * 60)
    print(f"GENERATED MANIFEST: {manifest.manifest_id}")
    print("=" * 60)
    print(f"Manifest Type:  {manifest.manifest_type}")
    print(f"Hash:           {manifest.compute_hash()}")
    print(f"Dataset:        {manifest.dataset}")
    print(f"Total Cases:    {len(manifest.cases)}")
    if manifest.modality_summary:
        print("Modality Summary:")
        for k, v in manifest.modality_summary.items():
            print(f"  {k}: {v}")
    if manifest.partition_statistics:
        print("Partition Statistics:")
        for p_name in ("dev", "val", "held_out"):
            if p_name in manifest.partition_statistics:
                st = manifest.partition_statistics[p_name]
                print(f"  {p_name}: {st['cases']} cases ({st['groups']} groups, {st['case_ratio']:.1%})")

    if args.output:
        manifest.save(args.output)
        print(f"\nManifest saved to: {args.output}")


if __name__ == "__main__":
    main()

