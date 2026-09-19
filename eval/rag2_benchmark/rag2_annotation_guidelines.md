# DD-IR-60 annotation guidelines

Annotators see only the query and blinded document presentation. They must not see condition, rank, root cause, inject time, final RCA, or remediation outcome.

0 = Not operationally relevant: no useful information for investigating, understanding, verifying, or safely responding.

1 = Contextually relevant: useful background, component behavior, dependency information, diagnostic context, or a direction, but not directly actionable.

2 = Directly operationally relevant: directly useful diagnostic, verification, troubleshooting, or safe mitigation information for the observable condition.

Positive example: a runbook explaining how to inspect observed timeout counters and dependency latency. Negative example: an unrelated database tuning guide.

Do not infer relevance from a hidden root cause or from the retrieval method.
