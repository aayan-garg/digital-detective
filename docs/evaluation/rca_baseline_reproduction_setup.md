# RCAEval baseline reproduction setup

This stage establishes isolated invocation infrastructure only. It does not
change Digital Detective RCA, `S_comb`, `E_elev`, Stage 9, or any production
evaluation code.

## Verified upstream pin

The public `phamquiluan/RCAEval` checkout is pinned to
`bb48c5aa9a24f1d5fcc716bdd479ea2d63145c90` (`1.7.0`). The checkout is kept
under the ignored experiment directory
[`experiments/external_rca/`](../../experiments/external_rca/) and is not part
of the production package.

## Methods and environments

The primary suite is exactly BARO, RCD, CausalRCA, MicroRank, and Multi-source
RCD (`mmrcd` in RCAEval). BARO, CausalRCA, and MicroRank use RCAEval's Python
3.12 default requirements. RCD and Multi-source RCD use the upstream Python
3.8 RCD lockfile. TORAI, EventADL, CIRCA, TraceRCA, RUN, and Multi-source BARO
are excluded.

## Condition and cases

The smoke runner uses the same local RCAEval RE2-OB Parquet cache used by
Digital Detective and preserves exact case IDs. It uses the benchmark
`inject_time` from both `cases.parquet` and `inject_time.txt`, with RCAEval's
default 600-second normal and 600-second incident slices. It does not use
`tau_confirm`, causal-prefix truncation, Stage 9, tuning, or candidate-universe
changes.

The bounded cases are:

- `re2ob_checkoutservice_cpu_2`
- `re2ob_currencyservice_cpu_2`
- `re2ob_emailservice_cpu_2`

## Compatibility and status

No upstream source or algorithm was modified. No local compatibility patch was
required. The historical artifact
`experiments/external_rca/results/smoke_external_rca.json` was generated before
the isolated RCAEval environment existed and is not a current method result.
The current three-method smoke uses
`eval/results/rcaeval_smoke_current_v1.json` and must be treated separately.
The current machine lacks Python 3.8 for RCD. Multi-source RCD's published
interface also requires pre-aggregated `logts.csv`, `tracets_err.csv`, and
`tracets_lat.csv`; the local cache contains raw logs/traces instead. The smoke
runner records these conditions as explicit failures and does not derive an
unverified replacement representation.

Run the bounded smoke test with:

```powershell
.\.venv_rcaeval_default\Scripts\python.exe experiments/external_rca/run_current_smoke.py
```

The current output is `eval/results/rcaeval_smoke_current_v1.json`; the old
historical artifact remains at
`experiments/external_rca/results/smoke_external_rca.json`.
This stage must stop after the smoke test; it does not run the 60 executions or
produce final baseline conclusions.
