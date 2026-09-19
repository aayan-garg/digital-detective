# DD-IR-60 corpus reproducibility

Run `PYTHONPATH=src python scripts/build_rag2_corpus.py` from the repository root. The script downloads only the pinned official source paths declared in its `SOURCES` tuple, preserves Markdown heading sections, deduplicates normalized chunks by SHA-256, and writes the corpus plus provenance and exclusion logs. The retrieval corpus is separate from `data/rag/operational_knowledge.json`.

The source tags and raw URLs are recorded per unit. Publication dates are blank for repository material; the pinned tag is the temporal admission basis. No final queries are read by the builder, and no qrels or retrieval evaluation are generated.
