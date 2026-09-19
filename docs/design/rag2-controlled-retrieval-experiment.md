# RAG-2 controlled retrieval experiment

RAG-2 keeps the frozen RAG-1 corpus, document boundaries, metadata, and query
construction unchanged. It compares five retrieval conditions:

| Condition | Retrieval |
|---|---|
| C0 | Frozen RAG-1 TF-IDF |
| C1 | Deterministic BM25 |
| C2 | `BAAI/bge-small-en-v1.5` dense retrieval |
| C3 | BM25 + BGE-small + unweighted RRF (`k=60`) |
| C4 | C3 followed by `cross-encoder/ettin-reranker-17m-v1` |

Sparse and dense arms use top-20, the hybrid candidate pool is top-20, and
the returned context is top-5. No weighted fusion, query expansion,
metadata filtering, chunking changes, agent integration, or fallback model is
allowed.

The active project environment currently has NumPy/SciPy but does not have
`sentence-transformers`, and the model weights are therefore not loaded.
C2-C4 fail explicitly with a model-availability blocker; the smoke harness
records that status and never substitutes another model.

`scripts/run_rag2_smoke.py` writes
`eval/results/rag2_smoke_v1.json`. Its labels are synthetic local qrels
separate from retrieval inputs and must not be treated as a scientific
benchmark. Confidence intervals and GO/NO-GO comparisons are exposed as a
future evaluation concern and are not inferred from this tiny smoke set.
