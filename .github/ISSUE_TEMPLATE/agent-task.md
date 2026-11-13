---
name: Digital Detective agent task
about: Queue a scoped implementation task for Antigravity, with ChatGPT as architect and reviewer.
title: "[DD-AGENT:READY] "
---

## Goal
What outcome should this task deliver?

## Context and rationale
Why is this needed? Include links to relevant research or project files where useful.

## Scope
### In scope
- 

### Out of scope
- 

## Invariants and constraints
- Read and follow `AGENTS.md`.
- Preserve the frozen detector, TCEC / Stage 9, `S_comb`, and established benchmark artifacts unless this task explicitly changes those constraints.
- Keep deterministic RCA authoritative; the LLM is an investigator/orchestrator, not the source of causal truth.
- Do not use ground truth, future telemetry, injection timestamps, or hidden labels in inference or retrieval.
- Do not fabricate metrics or report unrun tests as successful.

## Acceptance criteria
- [ ] 
- [ ] 
- [ ] 

## Validation
Required test commands, datasets, metrics, seeds, and comparison baselines:

- 

## Deliverables
Expected code, tests, documentation, benchmark artifacts, and PR report:

- 

## Notes for Antigravity
Process this issue using the `process-digital-detective-queue` workspace skill. The `[DD-AGENT:READY]` title prefix authorizes commits and pushes only to a dedicated feature branch and creation of a PR. Never push to or merge `main`.
