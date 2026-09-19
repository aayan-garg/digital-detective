"""Run local Ollama preliminary relevance judgments for the blinded DD-IR-60 pool."""

from __future__ import annotations

import csv
import json
import random
import re
import time
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("eval/rag2_benchmark")
MODEL = "qwen3:4b"
ENDPOINT = "http://localhost:11434/api/chat"
PROMPT_VERSION = "rag2-llm-relevance-v1"
BATCH_SIZE = 10
RETRIES = 2
SEED = 60


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def query_map() -> dict[str, str]:
    source = [
        json.loads(line)
        for line in (ROOT / "rag2_queries.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return {f"q-{index:03d}": item["query"] for index, item in enumerate(sorted(source, key=lambda x: x["query_id"]), 1)}


def prompt_for(batch: list[dict[str, str]], queries: dict[str, str]) -> str:
    """Return the user-turn instruction content for /api/chat.

    The ChatML wrapper and system turn are handled by Ollama's chat endpoint,
    which also applies think=False correctly (unlike /api/generate with raw=True).
    """
    items = [
        {
            "annotation_document_id": row["annotation_document_id"],
            "observable_query": queries[row["query_id"]],
            "document_title": row["document_title"],
            "document_text": row["document_text"][:250],
        }
        for row in batch
    ]
    return (
        "You are a blinded operational-knowledge relevance judge. "
        "Do not identify a root cause. Judge only operational usefulness for investigating, "
        "understanding, verifying, or safely responding to the observable condition.\n\n"
        "Grades are exactly: 0 = not operationally relevant; 1 = contextually relevant; "
        "2 = directly operationally relevant. Return strict JSON array with one object per input item, "
        "each containing exactly annotation_document_id, grade (integer 0, 1, or 2), and justification (brief, under 10 words).\n\n"
        "INPUT ITEMS:\n" + json.dumps(items, ensure_ascii=False)
    )


def call_ollama(prompt: str) -> tuple[list[dict], float]:
    """Call /api/chat with think=False to suppress Qwen3 chain-of-thought.

    /api/generate with raw=True ignores the think parameter, causing the model
    to emit unbounded <think>...</think> blocks that exceed both num_predict and
    the request timeout.  /api/chat correctly injects the no-think control token
    before generation.  Annotation semantics are identical.
    """
    payload = json.dumps({
        "model": MODEL,
        "stream": False,
        # think=False disables Qwen3 chain-of-thought via the /api/chat endpoint.
        # This is the reliable mechanism: /api/generate+raw=True ignores this flag.
        "think": False,
        "messages": [
            {"role": "system", "content": "You are a precise, concise JSON relevance judge."},
            {"role": "user", "content": prompt},
        ],
        "options": {
            "temperature": 0,
            "seed": SEED,
            "num_predict": 600,
        },
    }).encode("utf-8")
    started = time.perf_counter()
    request = urllib.request.Request(
        ENDPOINT,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        body = json.loads(response.read().decode("utf-8"))
    # /api/chat returns body["message"]["content"]
    raw_response = body.get("message", {}).get("content", "").strip()
    # Strip optional markdown code fences
    content = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_response).strip()
    # Locate the JSON array; the model may prefix with a short explanation
    bracket = content.find("[")
    if bracket != -1:
        content = content[bracket:]
    if not content.startswith("["):
        content = "[" + content
    if not content.endswith("]"):
        content += "]"
    parsed = json.loads(content)
    if isinstance(parsed, dict):
        parsed = parsed.get("annotations", parsed.get("results", parsed.get("items", [])))
    if not isinstance(parsed, list):
        raise ValueError("Ollama response was not a JSON array.")
    return parsed, time.perf_counter() - started


def validate(result: list[dict], expected: set[str]) -> dict[str, dict]:
    if len(result) != len(expected):
        raise ValueError(f"Expected {len(expected)} annotations, received {len(result)}.")
    validated = {}
    for item in result:
        identifier = item.get("annotation_document_id")
        grade = item.get("grade")
        if identifier not in expected or identifier in validated:
            raise ValueError(f"Missing, duplicate, or unknown annotation_document_id: {identifier}")
        try:
            int_grade = int(grade)
            if int_grade not in (0, 1, 2):
                raise ValueError
        except Exception:
            raise ValueError(f"Invalid grade for {identifier}: {grade!r}")
        validated[identifier] = {
            "grade": int_grade,
            "justification": str(item.get("justification", "")).strip()[:1000],
        }
    if set(validated) != expected:
        raise ValueError("Response IDs did not match batch IDs.")
    return validated


def main() -> None:
    pool = rows(ROOT / "rag2_pool.tsv")
    queries = query_map()
    if len(pool) != 1251 or len(queries) != 60:
        raise SystemExit("Frozen pool/query counts do not match expected DD-IR-60 inputs.")
    output = ROOT / "rag2_llm_annotation_v1.jsonl"
    records = []
    already_done = set()
    if output.is_file():
        for line in output.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                if rec.get("success"):
                    already_done.add(rec["annotation_document_id"])
                    records.append(rec)
        print(f"Resuming annotation: {len(already_done)}/{len(pool)} already completed.", flush=True)

    failures = []
    for start in range(0, len(pool), BATCH_SIZE):
        batch = pool[start:start + BATCH_SIZE]
        batch_needed = [row for row in batch if row["annotation_document_id"] not in already_done]
        if not batch_needed:
            continue
        batch = batch_needed
        expected = {row["annotation_document_id"] for row in batch}
        prompt = prompt_for(batch, queries)
        batch_ok = False
        last_error = ""
        for attempt in range(RETRIES + 1):
            try:
                result, latency = call_ollama(prompt)
                validated = validate(result, expected)
                timestamp = datetime.now(timezone.utc).isoformat()
                for row in batch:
                    item = validated[row["annotation_document_id"]]
                    records.append({
                        "query_id": row["query_id"],
                        "annotation_document_id": row["annotation_document_id"],
                        "grade": item["grade"],
                        "judge_model": MODEL,
                        "judge_backend": "ollama",
                        "label_source": "LLM-assisted preliminary relevance annotation",
                        "prompt_version": PROMPT_VERSION,
                        "latency_sec": round(latency / len(batch), 6),
                        "success": True,
                        "justification": item["justification"],
                        "timestamp": timestamp,
                    })
                with output.open("a", encoding="utf-8") as handle:
                    for record in records[-len(batch):]:
                        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                batch_ok = True
                break
            except Exception as error:
                last_error = f"{type(error).__name__}: {error}"
                time.sleep(1 + attempt)
        if not batch_ok:
            for row in batch:
                failures.append({
                    "query_id": row["query_id"],
                    "annotation_document_id": row["annotation_document_id"],
                    "grade": None,
                    "judge_model": MODEL,
                    "judge_backend": "ollama",
                    "prompt_version": PROMPT_VERSION,
                    "latency_sec": None,
                    "success": False,
                    "error": last_error,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
        if (start // BATCH_SIZE + 1) % 10 == 0:
            print(f"processed={min(start + BATCH_SIZE, len(pool))}/{len(pool)} successes={len(records)} failures={len(failures)}", flush=True)
    if failures:
        with output.open("a", encoding="utf-8") as handle:
            for record in failures:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    labeled = [record for record in records if record["success"]]
    sample = random.Random(60).sample(labeled, min(50, len(labeled)))
    audit_path = ROOT / "rag2_llm_annotation_audit.tsv"
    with audit_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("query_id", "annotation_document_id", "llm_grade", "human_audit_grade", "agreement", "audit_note"),
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        for record in sample:
            writer.writerow({
                "query_id": record["query_id"],
                "annotation_document_id": record["annotation_document_id"],
                "llm_grade": record["grade"],
                "human_audit_grade": "",
                "agreement": "",
                "audit_note": "",
            })
    distribution = Counter(record["grade"] for record in labeled)
    latencies = [record["latency_sec"] for record in labeled]
    print(json.dumps({
        "total_pairs": len(pool),
        "successfully_labeled": len(labeled),
        "failed_pairs": len(failures),
        "grade_distribution": dict(sorted(distribution.items())),
        "mean_latency_sec": sum(latencies) / len(latencies) if latencies else None,
        "median_latency_sec": sorted(latencies)[len(latencies) // 2] if latencies else None,
        "audit_sample_size": len(sample),
    }, indent=2))


if __name__ == "__main__":
    main()
