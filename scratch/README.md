# Scratch Directory: Research Provenance & Exploratory Material

This directory contains temporary exploratory scripts, ablation drivers, and raw benchmark result dumps created during earlier research milestones (Milestones 10.8 through 10.16) and the RCD baseline comparison study.

These files are retained solely for **research provenance and reproducibility verification**. They are not part of the production library (`src/digital_detective`), test suite (`tests/`), or active evaluation infrastructure (`eval/`).

## Contents Overview

1. **RCD Baseline Reproduction**:
   - `run_rcd_benchmark.py`: Benchmark driver for evaluating the official RCD algorithm against the 30-case RE2-OB benchmark.
   - `standalone_rcd.py`, `official_*.py`: Standalone execution harness and helper modules derived from the official RCD repository.
   - `official_causallearn/`, `official_rcd_runtime/`: Reference runtime dependencies used during RCD evaluation.
   - `rcd_benchmark_results.json`: Raw results from the RCD reproduction run (summarized in `docs/research/established_rca_methods.md`).
   - `sock-shop-data/`: Minimal sample CSV telemetry fixture for local RCD execution testing.

2. **Milestone 10.8 - 10.16 Ablation & Diagnostic Scripts**:
   - `run_strength_ablation_milestone_10_8.py`: Strength ablation experiments.
   - `run_directional_sink_ablation_milestone_10_10.py`: Directional sink ablation.
   - `run_soft_downstream_attenuation_milestone_10_12.py`: Soft downstream attenuation ablation.
   - `run_trace_attribution_ablation.py`: Trace latency attribution ablation.
   - `run_trace_latency_diagnostics_all_30.py`: Trace duration diagnostics across 30 cases.
   - Associated `*.json` files: Raw diagnostic logs and quantile outputs.

## Reproducibility Note

All stable conclusions, algorithmic definitions, and quantitative comparisons resulting from these exploratory scripts have been incorporated into:
- The frozen baseline implementations in `src/digital_detective/rca.py` ($S_{\text{comb}}$) and `src/digital_detective/trace_attribution.py` ($E_{\text{elev}}$)
- The authoritative research note: `docs/research/established_rca_methods.md`
- The system architecture: `docs/design/digital-detective-core.md`
