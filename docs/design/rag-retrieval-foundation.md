# RAG-1: Retrieval foundation

This milestone adds an isolated operational-context provider. It does not
change deterministic RCA, the agent A/C experiment, ML experiments,
remediation, or evaluation baselines.

## Flow

```text
observed incident context
  -> allowlisted lexical query
  -> local JSON corpus
  -> deterministic TF-IDF/cosine retriever
  -> ranked documents with metadata and bounded snippets
  -> bounded formatted context
```

`digital_detective.rag.DeterministicRetriever.retrieve(query, k)` is the
future integration boundary. No agent or orchestrator currently calls it.

The example corpus under `data/rag/` is explicitly synthetic/local test
knowledge, not a record of historical incidents. Query construction accepts
only candidate service, observed signals, and observed modalities. It does not
read ground truth, injection timing, fault labels, root-cause labels, or future
telemetry.

Run the focused tests with `pytest tests/test_rag.py` and the standalone smoke
experiment with `python scripts/run_rag_smoke.py`.
