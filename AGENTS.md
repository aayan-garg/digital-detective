# Digital Detective agent guide

## Scope and workflow

- Inspect relevant code, tests, configuration, documentation, and Git state before changing anything. Do not change files merely because they look incomplete.
- Reuse suitable existing abstractions. Before adding a component, confirm one does not already solve the problem; keep a single source of truth for shared concepts.
- Implement only the current requested milestone. Do not silently continue into later roadmap phases or build speculative infrastructure.
- For non-trivial work, identify the problem, relevant components, smallest proposed change, and validation strategy before coding. Avoid process overhead for very small changes.
- Make the smallest reasonable change. Prefer clear, correct functions and modules over unnecessary classes, layers, factories, managers, registries, wrappers, or fashionable design patterns.
- Do not create dead code: unused modules, functions, interfaces, configuration, or dependencies. Remove obsolete code only when the requested change makes it unnecessary. Avoid duplicate implementations unless an experiment explicitly compares them.

## Dependencies, environment, and configuration

- Add dependencies only for a concrete current use. Prefer established libraries for mature problems, but verify unfamiliar external APIs against current authoritative documentation before using them.
- Keep runtime dependencies separate from development or test dependencies when they are introduced. Avoid dependency sprawl.
- Use the Python version and environment declared by the project. Do not install packages globally or create arbitrary virtual environments outside the project.
- Do not commit virtual environments, caches, downloaded datasets, experiment outputs, model checkpoints, secrets, or machine-specific files. Keep environment-specific settings out of source code.
- Keep defaults explicit. Do not scatter magic constants, and do not introduce a configuration system before the complexity warrants one.

## Research, data, and architecture

- Never fabricate benchmark numbers, dataset properties, package APIs, papers, implementation details, or experimental results. Do not claim a run succeeded unless it was actually run.
- Clearly distinguish verified facts, project assumptions, hypotheses, and experimental results. Use authoritative sources or the actual repository/documentation when external research is needed.
- Inspect a dataset before writing dataset-specific code; never assume its labels, timestamps, file layout, structure, or semantics. Keep preprocessing reproducible and datasets out of normal source control unless explicitly required.
- Keep experimental implementations isolated from stable/core utilities so approaches are easy to compare and remove. Do not let one experiment's assumptions become core architecture.
- Document important architectural decisions, non-obvious algorithms, dataset assumptions, experiment methodology, and reproducibility instructions—without producing boilerplate documentation.

### Intended architecture (planned, not implemented)

The eventual flow is: telemetry → canonical representation → anomaly detection → dependency graph → causal RCA → historical/operational knowledge retrieval → evidence fusion → investigation agent → safety/guardrails → remediation → recovery verification → evaluation.

The LLM/agent is an investigator and orchestrator, not the source of truth. Deterministic, statistical, and causal tools must provide the evidence. RAG supplies historical and operational context; it does not replace causal analysis. Deep learning is experimental and must earn inclusion through evaluation. Do not hard-code this future architecture into the current source tree.

## Quality and validation

- Add focused, meaningful tests for each behavioral change. Prefer deterministic unit tests; add integration tests where component interactions matter. Do not write tests that exercise code without assertions of useful behavior.
- Run the smallest relevant test set first, then broader tests when appropriate. After implementation, inspect the diff, run tests, check configured formatting/linting, verify imports, check for generated artifacts, and review Git status.
- Do not hide errors to make tests pass or silently fall back to incorrect behavior. Handle expected failures intentionally and make unexpected failures explicit and useful.
- Add logging only when it materially helps operation or debugging; do not make it noisy or use it in place of proper structure and error handling.
- Optimize only measured bottlenecks. Do not add complex concurrency, caching, distributed processing, or vector databases without demonstrated need.

## Git and stop conditions

- Do not commit or push unless explicitly instructed. When asked to commit, keep commits logically scoped.
- Never rewrite, delete, or overwrite unrelated user changes. Inspect the working tree before destructive operations.
- Stop and ask for clarification when a genuinely ambiguous choice would materially affect the architecture. Do not invent an unverified external API, and report unexplained validation failures rather than claiming success.
- Treat the requested milestone as the scope boundary; do not proceed into unrelated work merely because it is technically possible.
