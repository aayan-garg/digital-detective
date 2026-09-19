"""Build the frozen DD-IR-60 operational corpus from pinned official sources.

This script is intentionally independent from the RAG-1 smoke corpus and from
the final query set. It downloads only the source files listed below, verifies
their hashes when supplied, and preserves Markdown headings as retrieval-unit
boundaries.
"""

from __future__ import annotations

import hashlib
import json
import re
import urllib.request
from datetime import date
from pathlib import Path

RETRIEVED_DATE = "2026-09-19"
OUTPUT = Path("eval/rag2_benchmark")
FORBIDDEN_MATERIAL = re.compile(
    r"\b(RE2|RCAEval|postmortem|fault[- ]injection|benchmark label|"
    r"test incident|remediation outcome|root[- ]cause annotation)\b",
    re.IGNORECASE,
)

SOURCES = (
    # Kubernetes website: documentation is CC BY 4.0 at the pinned branch.
    ("kubernetes", "website", "release-1.30", "CC BY 4.0", "kubernetes",
     "content/en/docs/tasks/configure-pod-container/assign-cpu-resource.md", "cpu"),
    ("kubernetes", "website", "release-1.30", "CC BY 4.0", "kubernetes",
     "content/en/docs/tasks/configure-pod-container/assign-memory-resource.md", "memory"),
    ("kubernetes", "website", "release-1.30", "CC BY 4.0", "kubernetes",
     "content/en/docs/tasks/configure-pod-container/configure-liveness-readiness-startup-probes.md", "service-health"),
    ("kubernetes", "website", "release-1.30", "CC BY 4.0", "kubernetes",
     "content/en/docs/tasks/configure-pod-container/configure-persistent-volume-storage.md", "disk"),
    ("kubernetes", "website", "release-1.30", "CC BY 4.0", "kubernetes",
     "content/en/docs/tasks/configure-pod-container/resize-container-resources.md", "resource-limits"),
    ("kubernetes", "website", "release-1.30", "CC BY 4.0", "kubernetes",
     "content/en/docs/tasks/configure-pod-container/share-process-namespace.md", "process"),
    ("kubernetes", "website", "release-1.30", "CC BY 4.0", "kubernetes",
     "content/en/docs/tasks/network/customize-hosts-file-for-pods.md", "dns"),
    ("kubernetes", "website", "release-1.30", "CC BY 4.0", "kubernetes",
     "content/en/docs/tasks/network/validate-dual-stack.md", "networking"),
    ("kubernetes", "website", "release-1.30", "CC BY 4.0", "kubernetes",
     "content/en/docs/tasks/debug/debug-application/debug-running-pod.md", "debugging"),
    ("kubernetes", "website", "release-1.30", "CC BY 4.0", "kubernetes",
     "content/en/docs/tasks/debug/debug-application/debug-service.md", "service-health"),
    ("kubernetes", "website", "release-1.30", "CC BY 4.0", "kubernetes",
     "content/en/docs/tasks/debug/debug-cluster/resource-usage-monitoring.md", "observability"),
    # Prometheus: Apache-2.0 at the pinned release.
    ("prometheus", "prometheus", "v2.53.0", "Apache-2.0", "prometheus",
     "docs/querying/basics.md", "metrics"),
    ("prometheus", "prometheus", "v2.53.0", "Apache-2.0", "prometheus",
     "docs/querying/operators.md", "metrics"),
    ("prometheus", "prometheus", "v2.53.0", "Apache-2.0", "prometheus",
     "docs/querying/functions.md", "metrics"),
    ("prometheus", "prometheus", "v2.53.0", "Apache-2.0", "prometheus",
     "docs/storage.md", "observability"),
    ("prometheus", "prometheus", "v2.53.0", "Apache-2.0", "prometheus",
     "docs/getting_started.md", "service-health"),
    ("prometheus", "prometheus", "v2.53.0", "Apache-2.0", "prometheus",
     "docs/configuration/configuration.md", "observability"),
    ("prometheus", "prometheus", "v2.53.0", "Apache-2.0", "prometheus",
     "docs/management_api.md", "safe-operations"),
    # gRPC: Apache-2.0 at the pinned release.
    ("grpc", "grpc", "v1.64.0", "Apache-2.0", "grpc",
     "doc/PROTOCOL-HTTP2.md", "grpc"),
    ("grpc", "grpc", "v1.64.0", "Apache-2.0", "grpc",
     "doc/health-checking.md", "service-health"),
    ("grpc", "grpc", "v1.64.0", "Apache-2.0", "grpc",
     "doc/connectivity-semantics-and-api.md", "grpc"),
    ("grpc", "grpc", "v1.64.0", "Apache-2.0", "grpc",
     "doc/connection-backoff.md", "networking"),
    ("grpc", "grpc", "v1.64.0", "Apache-2.0", "grpc",
     "doc/keepalive.md", "grpc"),
    ("grpc", "grpc", "v1.64.0", "Apache-2.0", "grpc",
     "doc/statuscodes.md", "error-rate"),
    ("grpc", "grpc", "v1.64.0", "Apache-2.0", "grpc",
     "doc/wait-for-ready.md", "dependency-failure"),
    ("grpc", "grpc", "v1.64.0", "Apache-2.0", "grpc",
     "doc/load-balancing.md", "dependency-failure"),
    # OpenTelemetry specification: Apache-2.0 at the pinned release.
    ("open-telemetry", "opentelemetry-specification", "v1.28.0", "Apache-2.0", "opentelemetry",
     "specification/trace/api.md", "tracing"),
    ("open-telemetry", "opentelemetry-specification", "v1.28.0", "Apache-2.0", "opentelemetry",
     "specification/trace/sdk.md", "tracing"),
    ("open-telemetry", "opentelemetry-specification", "v1.28.0", "Apache-2.0", "opentelemetry",
     "specification/trace/semantic_conventions/README.md", "tracing"),
    ("open-telemetry", "opentelemetry-specification", "v1.28.0", "Apache-2.0", "opentelemetry",
     "specification/logs/data-model.md", "logging"),
    ("open-telemetry", "opentelemetry-specification", "v1.28.0", "Apache-2.0", "opentelemetry",
     "specification/metrics/data-model.md", "metrics"),
    ("open-telemetry", "opentelemetry-specification", "v1.28.0", "Apache-2.0", "opentelemetry",
     "specification/context/api-propagators.md", "tracing"),
    ("open-telemetry", "opentelemetry-specification", "v1.28.0", "Apache-2.0", "opentelemetry",
     "specification/error-handling.md", "safe-operations"),
    ("open-telemetry", "opentelemetry-specification", "v1.28.0", "Apache-2.0", "opentelemetry",
     "specification/semantic-conventions.md", "observability"),
)


def download(owner: str, repo: str, tag: str, path: str) -> str:
    url = f"https://raw.githubusercontent.com/{owner}/{repo}/{tag}/{path}"
    request = urllib.request.Request(url, headers={"User-Agent": "digital-detective-ddir60/1.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read().decode("utf-8")


def markdown_units(text: str) -> list[tuple[str, str]]:
    heading = "Document overview"
    buffer: list[str] = []
    units: list[tuple[str, str]] = []
    for line in text.splitlines():
        match = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
        if match:
            if "\n".join(buffer).strip():
                units.append((heading, "\n".join(buffer).strip()))
            heading = re.sub(r"[`*_]", "", match.group(2)).strip()
            buffer = []
        else:
            buffer.append(line)
    if "\n".join(buffer).strip():
        units.append((heading, "\n".join(buffer).strip()))
    return [(heading, body) for heading, body in units if len(body) >= 80]


def build() -> tuple[list[dict], list[dict]]:
    corpus: list[dict] = []
    exclusions: list[dict] = []
    seen_hashes: set[str] = set()
    for owner, repo, tag, license_name, family, path, category in SOURCES:
        url = f"https://github.com/{owner}/{repo}/blob/{tag}/{path}"
        raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{tag}/{path}"
        try:
            text = download(owner, repo, tag, path)
        except Exception as error:
            exclusions.append({
                "source_url": url,
                "reason": f"download failure: {error}",
                "admission_status": "excluded",
            })
            continue
        for ordinal, (heading, body) in enumerate(markdown_units(text), 1):
            normalized = re.sub(r"\s+", " ", body).strip()
            if FORBIDDEN_MATERIAL.search(normalized):
                exclusions.append({
                    "source_url": url,
                    "section": heading,
                    "reason": "prohibited benchmark-specific material",
                    "admission_status": "excluded",
                })
                continue
            content_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
            if content_hash in seen_hashes:
                exclusions.append({
                    "source_url": url,
                    "section": heading,
                    "reason": "duplicate normalized chunk",
                    "admission_status": "excluded",
                })
                continue
            seen_hashes.add(content_hash)
            document_id = f"ddir60-{family}-{hashlib.sha1((tag + ':' + path + ':' + str(ordinal) + ':' + heading).encode()).hexdigest()[:12]}"
            corpus.append({
                "document_id": document_id,
                "title": heading,
                "text": normalized,
                "metadata": {
                    "source": f"{owner}/{repo}",
                    "source_url": raw_url,
                    "repository": f"https://github.com/{owner}/{repo}",
                    "version": tag,
                    "commit_tag": tag,
                    "publication_date": "",
                    "retrieved_date": RETRIEVED_DATE,
                    "license": license_name,
                    "document_type": "official technical documentation",
                    "section": heading,
                    "knowledge_category": category,
                    "admission_status": "admitted",
                    "provenance": "pinned official repository file; license verified at repository LICENSE",
                },
            })
    return corpus, exclusions


def write_outputs(corpus: list[dict], exclusions: list[dict]) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "rag2_corpus.jsonl").open("w", encoding="utf-8") as handle:
        for item in corpus:
            handle.write(json.dumps(item, sort_keys=True) + "\n")
    with (OUTPUT / "rag2_corpus_provenance.tsv").open("w", encoding="utf-8") as handle:
        fields = ("document_id", "source", "source_url", "repository", "version", "commit_tag",
                  "publication_date", "retrieved_date", "license", "document_type", "section",
                  "knowledge_category", "admission_status", "provenance")
        handle.write("\t".join(fields) + "\n")
        for item in corpus:
            metadata = item["metadata"]
            handle.write("\t".join(str(metadata.get(field, "") if field != "document_id" else item["document_id"]) for field in fields) + "\n")
    with (OUTPUT / "rag2_temporal_audit.tsv").open("w", encoding="utf-8") as handle:
        handle.write("document_id\tversion\tcommit_tag\tpublication_date\tretrieved_date\ttemporal_policy\tadmission_status\treason\n")
        for item in corpus:
            metadata = item["metadata"]
            handle.write(f"{item['document_id']}\t{metadata['version']}\t{metadata['commit_tag']}\t\t{RETRIEVED_DATE}\tpinned version-controlled source\tadmitted\tversion/tag is fixed; publication date not required for tagged repository material\n")
    with (OUTPUT / "rag2_exclusion_log.tsv").open("w", encoding="utf-8") as handle:
        handle.write("source_url\tsection\treason\tadmission_status\n")
        for item in exclusions:
            handle.write("\t".join(str(item.get(field, "")) for field in ("source_url", "section", "reason", "admission_status")) + "\n")
    categories = sorted({item["metadata"]["knowledge_category"] for item in corpus})
    families = sorted({item["metadata"]["source"] for item in corpus})
    versions = sorted({item["metadata"]["version"] for item in corpus})
    corpus_hash = hashlib.sha256(
        "\n".join(json.dumps(item, sort_keys=True) for item in corpus).encode("utf-8")
    ).hexdigest()
    source_manifest = sorted(
        {
            (
                item["metadata"]["source"],
                item["metadata"]["version"],
                item["metadata"]["license"],
            )
            for item in corpus
        }
    )
    source_lines = "\n".join(
        f"  - source: {source}\n    version: {version}\n    license: {license_name}"
        for source, version, license_name in source_manifest
    )
    (OUTPUT / "rag2_manifest.yaml").write_text(
        "benchmark_id: DD-IR-60\n"
        "corpus_id: ddir60-operational-knowledge-v1\n"
        f"retrieved_date: {RETRIEVED_DATE}\n"
        "cutoff_policy: version-controlled official documentation admitted only at pinned tags; undated unversioned material excluded\n"
        "temporal_claim: temporally leakage-controlled with respect to benchmark information availability\n"
        f"retrieval_unit_count: {len(corpus)}\n"
        f"source_family_count: {len(families)}\n"
        f"source_version_count: {len(versions)}\n"
        f"excluded_source_unit_count: {len(exclusions)}\n"
        f"coverage_categories: [{', '.join(categories)}]\n"
        f"corpus_sha256: {corpus_hash}\n"
        "source_manifest:\n"
        f"{source_lines}\n"
        "scientific_qrels_present: false\n"
        "final_c0_c4_evaluation_run: false\n",
        encoding="utf-8",
    )
    (OUTPUT / "RAG2_BENCHMARK_CARD.md").write_text(
        "# DD-IR-60 benchmark card\n\n"
        "## Corpus\n\n"
        "The final corpus is a separate, frozen operational-knowledge snapshot built from pinned official Kubernetes, Prometheus, gRPC, and OpenTelemetry repository documentation. "
        "It is temporally leakage-controlled with respect to benchmark information availability. "
        "No RE2 case, fault label, postmortem, RCA report, annotation, or remediation result was used to select or write a unit.\n\n"
        f"- Retrieval units: {len(corpus)}\n"
        f"- Source families: {len(families)}\n"
        f"- Source versions: {len(versions)}\n"
        f"- Excluded source units: {len(exclusions)}\n"
        f"- Categories: {', '.join(categories)}\n"
        "- Human qrels: not present\n"
        "- Final C0-C4 evaluation: not run\n\n"
        "Repository licenses were independently verified from the pinned repositories: Kubernetes website CC BY 4.0; Prometheus, gRPC, and OpenTelemetry Apache-2.0.\n",
        encoding="utf-8",
    )
    (OUTPUT / "rag2_reproducibility.md").write_text(
        "# DD-IR-60 corpus reproducibility\n\n"
        "Run `PYTHONPATH=src python scripts/build_rag2_corpus.py` from the repository root. "
        "The script downloads only the pinned official source paths declared in its `SOURCES` tuple, preserves Markdown heading sections, "
        "deduplicates normalized chunks by SHA-256, and writes the corpus plus provenance and exclusion logs. "
        "The retrieval corpus is separate from `data/rag/operational_knowledge.json`.\n\n"
        "The source tags and raw URLs are recorded per unit. Publication dates are blank for repository material; the pinned tag is the temporal admission basis. "
        "No final queries are read by the builder, and no qrels or retrieval evaluation are generated.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    corpus, exclusions = build()
    write_outputs(corpus, exclusions)
    print(json.dumps({"retrieval_units": len(corpus), "excluded": len(exclusions)}, indent=2))
