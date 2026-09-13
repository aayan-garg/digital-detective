# Digital Detective

Digital Detective is a planned autonomous AIOps incident-investigation system for transient cascading failures in microservice environments.

The repository is currently in the engineering and setup phase. No incident-analysis or remediation functionality has been implemented yet.

## Planned research areas

- Metrics, logs, and traces
- Anomaly detection and service dependency graphs
- Causal root-cause analysis
- Historical incident knowledge with RAG and GraphRAG
- Investigation agents, safety guardrails, and query budgets
- Remediation, recovery verification, and benchmark evaluation

These areas are planned research directions, not current capabilities.

## Development

Use Python 3.11 or 3.12. Create a virtual environment, install the package in editable mode, and run the smoke test:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python -m pip install --editable .
.\.venv\Scripts\python -m unittest discover -s tests
```
