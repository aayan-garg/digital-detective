# Temporal DL complement experiment

This single-run experiment tests whether causal-prefix metric sequences add
candidate-service ranking signal beyond accepted Model B static features. It
uses the existing RE2-SS/RE2-TT development population and locked RE2-OB
repetition-2/3 population, with family separation unchanged.

The raw `metrics.parquet` and verified `inject_time.txt` artifacts already
contain 721 timestamped causal-prefix rows per case. No telemetry extraction or
sequence reconstruction from aggregate features is performed. The runner is
[`run_temporal_dl_experiment.py`](../../experiments/run_temporal_dl_experiment.py);
the result is [`temporal_dl_experiment_v1.json`](../../eval/results/temporal_dl_experiment_v1.json).

## Decision

**REJECT** — the temporal DL complement did not improve the locked-test
ranking metrics. **Model B is retained.**
