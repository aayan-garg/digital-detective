# Isolated RCAEval baseline reproduction

This directory contains experiment-only orchestration. The pinned upstream
checkout is ignored by Git at `RCAEval/`; it is never imported by
`src/digital_detective/`.

## Pin and environments

- RCAEval: `bb48c5aa9a24f1d5fcc716bdd479ea2d63145c90` (`1.7.0`).
- Default environment: Python 3.12, `RCAEval[default]` from upstream
  `requirements.txt`.
- RCD environment: Python 3.8, `RCAEval[rcd]` from upstream
  `requirements_rcd.lock`.
- The default environment is required for BARO, CausalRCA, and MicroRank.
  RCD and Multi-source RCD use the separate RCD environment.
- No dependency is installed into the Digital Detective environment by this
  experiment setup.

## Current smoke status

`run_smoke.py` is deliberately bounded to three RE2-OB repetition-2 cases:

1. `re2ob_checkoutservice_cpu_2`
2. `re2ob_currencyservice_cpu_2`
3. `re2ob_emailservice_cpu_2`

It invokes only BARO, RCD, CausalRCA, MicroRank, and Multi-source RCD. It uses
the case-provided `inject_time` and RCAEval's documented default window
(600 seconds before and after injection). It does not use `tau_confirm` or
Stage 9 truncation.

On the inspected machine the smoke test is expected to report blockers rather
than fabricate rankings: Python 3.8 is not installed, the default scientific
dependencies are not installed, and the local Parquet cases contain raw
`logs.parquet`/`traces.parquet` rather than RCAEval's required pre-aggregated
`logts.csv`, `tracets_err.csv`, and `tracets_lat.csv` inputs for Multi-source
RCD. These are reproducibility blockers, not algorithm substitutions.

No local compatibility patch was applied. The upstream algorithms and source
checkout are unchanged. The smoke report is an integration artifact only; it
is not a research result and must not be treated as the 60-case experiment.
