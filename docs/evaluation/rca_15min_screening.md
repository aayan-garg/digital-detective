# RCA 15-minute screening

This is a time-bounded screening experiment, not the final 60-case RCAEval
literature reproduction. It uses the standard RCAEval/oracle condition:
benchmark-provided injection time and 600-second normal/incident windows.
Stage 9, `S_comb`, `E_elev`, and all RCA algorithm implementations are
unchanged.

## Scope

Cases:

- `re2ob_checkoutservice_cpu_2`
- `re2ob_currencyservice_cpu_2`
- `re2ob_emailservice_cpu_2`
- `re2ob_checkoutservice_delay_2`
- `re2ob_currencyservice_delay_2`

Methods:

- BARO
- Multi-source RCD
- Existing frozen Digital Detective RCA reference

BARO was bounded to 30 seconds per case and Multi-source RCD to 45 seconds per
case. Missing data/layout and timeout outcomes are recorded explicitly in
[`rca_15min_screening_v1.json`](../../eval/results/rca_15min_screening_v1.json).

## Reproducibility notes

RCAEval is pinned to version 1.7.0 at commit
`bb48c5aa9a24f1d5fcc716bdd479ea2d63145c90`. Multi-source RCD requires the
published `logts.csv`, `tracets_err.csv`, and `tracets_lat.csv` inputs; no
ad-hoc conversion from raw Parquet logs/traces was created. CausalRCA,
MicroRank, and standalone RCD were intentionally deferred from this screening
to preserve the deadline.

The Digital Detective values are reused from existing frozen evaluation
artifacts rather than rerunning expensive pipeline components. This screening
does not establish statistical significance and must not be called the final
benchmark.

## Observed results

| Method | Cases completed | Top@1 | Top@3 | Top@5 | MRR |
|---|---:|---:|---:|---:|---:|
| BARO | 5/5 | 0.00 | 0.80 | 1.00 | 0.40 |
| Multi-source RCD | 0/5 | — | — | — | — |
| Digital Detective frozen RCA | 5/5 | 0.40 | 0.80 | 0.80 | 0.595 |

BARO completed all five cases within the 30-second per-case bound. Multi-source
RCD was explicitly blocked because the local cache lacks the official
pre-aggregated multimodal CSV layout; no substitute aggregation was used.
The two additional cases were selected directly from the existing cached
RE2-OB case directories and manifest.

## Screening decision

**RCA SCREEN: PASS — provisional; move to ML stage.**

On the completed cases Digital Detective is not clearly or consistently worse
than BARO: Top@3 is equal, Top@1 is higher, and Top@5 is lower. The
Multi-source RCD comparison remains unresolved because its official inputs were
unavailable. This is a screening decision only, with no claim of statistical
significance and no claim that the final 60-case reproduction is complete.
