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

## External Research and API Verification

Digital Detective is research-grade: correctness of claims and reproducibility take priority over speed.

- **Datasets:** Before writing dataset-specific code, inspect the actual dataset or repository. Verify the current structure, labels and ground truth, timestamps and telemetry formats, and how cases/incidents are identified. Do not infer these from papers, blog posts, or memory when the data is available.
- **Papers:** Prefer the original paper. Separate what it reports from our interpretation; never invent numbers, datasets, baselines, or conclusions. Record experimental assumptions that affect reproducibility.
- **Repositories and packages:** Inspect an external repository's relevant files, interfaces, installation instructions, and—when reproducibility matters—its exact revision/version. For unfamiliar Python packages, verify current documentation or source, installed version, actual import path, and callable interface. Never invent method names, parameters, or return values; adapt old examples to the current API.
- **Versions and sources:** Record versions used by important experiments; do not casually upgrade or downgrade dependencies, and explain meaningful compatibility decisions. Prefer, in order: official project/repository documentation, the original paper, official package documentation, official source code, and other project-maintained authoritative documentation. Use secondary sources only when primary sources are unavailable; tutorials, generated articles, and search snippets are not authoritative evidence.
- **Uncertainty and conflicts:** Label information as **Verified**, **Assumption**, **Hypothesis**, or **Experimental result**. Never present an inference as directly verified. When sources conflict, identify the conflict, check version/date/context, use the most appropriate current primary source, and document decisions that affect implementation.
- **Results and reproducibility:** Never report an unmeasured result. Distinguish expected behavior, observed behavior, benchmark results, and informal smoke tests. For important experiments, record dataset/version, code revision, relevant dependency versions, configuration, evaluation procedure, and random seed when applicable.
- **Stop condition:** If an API, dataset structure, or research claim cannot be reliably verified, stop before implementing assumptions that could materially affect the system. State what is unknown and ask for clarification when necessary.

## Implementation Protocol

Prefer a smaller correct system over a larger speculative system.

1. **Understand:** Read the task; inspect the relevant files and architecture; identify scope and constraints; determine whether external research, APIs, datasets, or package behavior matter.
2. **Plan:** For non-trivial work, state the minimal changes, reused components, necessary new components, and validation strategy before editing. Do not create a large design document for a small change.
3. **Verify assumptions:** Follow **External Research and API Verification**. Do not proceed on a materially important unverified API, dataset, research claim, or version.
4. **Implement minimally:** Make the smallest coherent change, reuse existing code, prefer straightforward implementations, and avoid unrelated cleanup or speculative infrastructure.
5. **Test and validate:** Add or update meaningful tests; start focused and expand for integration or regression risk. Then run relevant tests, inspect imports and configured formatting/linting, review the final diff, check for accidental artifacts, and check Git status. Fix failures rather than hiding them; never weaken tests merely to pass.
6. **Review and report:** Before completion, confirm the scope is limited, complexity and duplication are justified, assumptions are documented, tests are meaningful, and results are reproducible where appropriate. For substantive changes, report what and why changed, affected files, dependency changes, validation and observed results, unresolved limitations, and deliberate omissions.
7. **Stop:** Complete and validate the requested task, then stop. One milestone does not authorize later phases: anomaly detection does not authorize RAG, RAG does not authorize an agent, and remediation does not authorize evaluation-methodology changes.

- State why each new dependency is needed before adding it, and verify compatibility with the project's Python version and existing dependency set.
- A failed test is not success; an unverified API is not a valid implementation basis; an unrun experiment is not a result; and unresolved architectural ambiguity must not be silently guessed. If the task cannot be completed safely within scope, report the blocker rather than expanding scope.
- Design experimental comparisons so baselines and evaluation procedures remain reproducible and comparable, important configuration is recorded, experimental code does not silently alter the stable/core pipeline, and positive and negative results are preserved honestly.

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
