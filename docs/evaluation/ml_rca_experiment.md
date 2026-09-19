# ML RCA experiment

## Scope and design

This experiment implements the approved compact supervised ranking design:

- pairwise candidate-service differences `x_root - x_negative`;
- binary `sklearn.LogisticRegression`;
- deterministic candidate ranking from pairwise win scores;
- Model A: existing `S_comb`, `E_elev`, and existing RCA evidence;
- Model B: Model A plus compact anomaly/episode/trace evidence;
- grouped three-fold validation by scenario family.

The implementation is in
[`experiments/ml_rca_ranker.py`](../../experiments/ml_rca_ranker.py) and
[`experiments/run_ml_rca_experiment.py`](../../experiments/run_ml_rca_experiment.py).
It does not modify deterministic RCA, the detector, Stage 9, `S_comb`, or
`E_elev`. Feature-name validation rejects service/fault/family/repetition/
injection/root-label identifiers.

## Data and leakage gate

The official RCAEval Parquet data were obtained from the pinned RCAEval source
and staged locally without using RE2-OB for training or model selection.
Development contains 180 cases and 60 scenario families:

- RE2-SS: 90 cases;
- RE2-TT: 90 cases.

The locked test contains exactly 60 RE2-OB repetition-2/3 cases. Family
repetitions remained grouped, and no RE2-OB case entered development.

The runner now projects trace input to the columns required by the existing
trace evidence extractor, omits unused logs, and persists case-level features
under the local cache so interrupted extraction can resume. These are
execution/caching changes only; feature definitions and model specification
were not changed.

## Measured result

The experiment completed successfully. Model B was selected using grouped
development validation by MRR:

| Population | Model | Top@1 | Top@3 | Top@5 | MRR |
|---|---|---:|---:|---:|---:|
| Development, grouped validation | Model A | 0.439 | 0.761 | 0.850 | 0.615 |
| Development, grouped validation | Model B | 0.450 | 0.789 | 0.828 | 0.623 |
| Locked RE2-OB reps 2–3 | Digital Detective frozen RCA | 0.617 | 0.817 | 0.867 | 0.738 |
| Locked RE2-OB reps 2–3 | Model A | 0.800 | 0.950 | 1.000 | 0.878 |
| Locked RE2-OB reps 2–3 | Model B (selected) | 0.817 | 0.967 | 1.000 | 0.892 |

Relative to the frozen RCA on the locked test, selected Model B improved
Top@1 by 0.200, Top@3 by 0.150, Top@5 by 0.133, and MRR by 0.153. This is
an observed result on the locked population, not a claim of statistical
significance.

Feature extraction and validation took approximately 2,019 seconds
(33.6 minutes) in the configured local environment. The complete per-case
rankings and validation records are stored in the two JSON artifacts below.

## Artifacts and validation

- [`ml_rca_development_v1.json`](../../eval/results/ml_rca_development_v1.json)
- [`ml_rca_locked_test_v1.json`](../../eval/results/ml_rca_locked_test_v1.json)
- Focused ranker, universe, and RCAEval adapter tests: 17 passed.
- Deterministic RCA, Stage 9, detector, `S_comb`, and `E_elev` were not modified.
