"""FastAPI application for Digital Detective Microservice RCA and Investigation System.

Provides REST endpoints for:
- Health and model readiness checks
- Incident catalogue and telemetry exploration
- End-to-end incident investigation with RCA, RAG, and Safety gating
- Interactive human-in-the-loop remediation approval/rejection
- Direct hybrid runbook search
- Web replay dashboard static assets
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
import time
from typing import Any, List, Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Ensure project paths
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "experiments") not in sys.path:
    sys.path.insert(0, str(ROOT / "experiments"))

from digital_detective.pipeline import InvestigationPipeline
from digital_detective.schema import (
    CanonicalIncident,
    PostInterventionCausalVerdict,
    RecoveryStatus,
    RemediationActionType,
    RemediationStatus,
)

app = FastAPI(
    title="Digital Detective RCA API",
    description="Microservice Root Cause Analysis, Hybrid RAG Runbook Retrieval, and Safe Remediation Engine",
    version="1.0.0",
)

REQUIRED_ALLOWED_ORIGINS = [
    "https://digital-detective-web-nine.vercel.app",
    "https://digital-detective-web-i7jn.vercel.app",
    "http://localhost:3000",
    "http://localhost:5173",
    "http://localhost:8000",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:8000",
]


def resolve_allowed_origins(env_origins: str | None = None) -> list[str]:
    """Merge configured CORS origins with required production and local dev origins.

    Guarantees production Vercel frontend and local development origins are always
    allowed even when an existing CORS_ORIGINS environment variable contains an
    outdated or partial domain list. Disallows wildcard '*'.
    """
    origins: list[str] = []
    seen: set[str] = set()

    def add_origin(raw: str) -> None:
        clean = raw.strip().strip("'\"").rstrip("/")
        if clean and clean not in seen and clean != "*":
            seen.add(clean)
            origins.append(clean)

    if env_origins:
        for item in env_origins.split(","):
            add_origin(item)

    for item in REQUIRED_ALLOWED_ORIGINS:
        add_origin(item)

    return origins


# Configure CORS strictly for production and local development origins
cors_origins_env = os.getenv("CORS_ORIGINS", "")
allowed_origins = resolve_allowed_origins(cors_origins_env)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


def _resolve_dataset_root() -> Path:
    env_root = os.getenv("RCAEVAL_DATASET_ROOT")
    if env_root and Path(env_root).is_dir():
        return Path(env_root)
    demo_root = ROOT / "data" / "demo_cases"
    if demo_root.is_dir():
        return demo_root
    return Path.home() / ".cache" / "rcaeval_validation"


DATASET_ROOT = _resolve_dataset_root()

# Global lazy-initialized pipeline singleton
_PIPELINE: Optional[InvestigationPipeline] = None


def get_pipeline() -> InvestigationPipeline:
    global _PIPELINE
    if _PIPELINE is None:
        enable_dense = os.getenv("ENABLE_DENSE_RAG", "false").lower() in ("true", "1", "yes")
        _PIPELINE = InvestigationPipeline(dataset_root=DATASET_ROOT, enable_dense_rag=enable_dense)
    return _PIPELINE


# ---------------------------------------------------------------------------
# Rate Limiting & Abuse Prevention
# ---------------------------------------------------------------------------

class InMemoryRateLimiter:
    """Sliding-window rate limiter per client IP."""

    def __init__(self, requests_per_minute: int = 30) -> None:
        self.rpm = requests_per_minute
        self._history: dict[str, list[float]] = {}

    def check(self, client_ip: str) -> bool:
        now = time.time()
        window_start = now - 60.0
        timestamps = self._history.get(client_ip, [])
        recent = [ts for ts in timestamps if ts > window_start]
        if len(recent) >= self.rpm:
            self._history[client_ip] = recent
            return False
        recent.append(now)
        self._history[client_ip] = recent
        return True


_investigate_limiter = InMemoryRateLimiter(int(os.getenv("RATE_LIMIT_INVESTIGATE", "30")))
_remediation_limiter = InMemoryRateLimiter(int(os.getenv("RATE_LIMIT_REMEDIATION", "15")))



# ---------------------------------------------------------------------------
# Request and Response Models
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "1.0.0"
    model_name: str = "Model B (Pairwise Supervised Ranker)"
    dataset_root: str
    cached_cases_count: int
    rag_retriever_ready: bool
    timestamp: float = Field(default_factory=time.time)


class IncidentSummary(BaseModel):
    case_id: str
    dataset: str
    system: str
    fault: str
    root_cause_service: str
    duration_minutes: float = 0.0
    has_metrics: bool = True
    has_traces: bool = True
    has_logs: bool = False


class InvestigationRequest(BaseModel):
    case_id: str = Field(..., description="ID of the incident case (e.g. 're2ob_checkoutservice_cpu_1')")
    model_name: str = Field(default="mock", description="Investigator model ('mock', 'qwen3-8b', 'ollama')")
    use_bocpd: bool = Field(default=False, description="Whether to run full multivariate Bayesian changepoint detection")
    budget: int = Field(default=20, description="Investigation query budget")
    execute_remediation: bool = Field(default=False, description="Simulate remediation verification automatically")


class RemediationActionRequest(BaseModel):
    case_id: str
    action: str
    target_service: str
    user_decision: str = Field(..., description="'approve', 'reject', or 'dry_run'")
    notes: Optional[str] = None


class RemediationActionResponse(BaseModel):
    case_id: str
    action: str
    target_service: str
    user_decision: str
    status: RemediationStatus
    execution_success: bool
    is_simulation: bool = True
    audit_log: List[str]
    message: str


class RAGQueryRequest(BaseModel):
    query: str
    k: int = Field(default=5, ge=1, le=20)
    category: Optional[str] = None
    target_service: Optional[str] = None


class RAGQueryResult(BaseModel):
    citation_id: str
    document_id: str
    title: str
    source: str
    section: str
    score: float
    snippet: str
    category: str


# ---------------------------------------------------------------------------
# API Routes
# ---------------------------------------------------------------------------

@app.get("/healthz")
def deployment_health_check() -> dict[str, Any]:
    """Lightweight deployment health probe for hosting platform checks."""
    return {
        "status": "ok",
        "service": "digital-detective-api",
        "dataset_available": DATASET_ROOT.is_dir(),
        "timestamp": time.time(),
    }


@app.get("/api/health", response_model=HealthResponse)
def health_check() -> HealthResponse:

    """Return system readiness, dataset availability, and index status."""
    cases_count = 0
    if DATASET_ROOT.is_dir():
        cases_count = len([d for d in DATASET_ROOT.iterdir() if d.is_dir() and not d.name.startswith(".")])

    pipe = get_pipeline()
    return HealthResponse(
        status="ok",
        version="1.0.0",
        model_name="Model B (Pairwise Supervised Ranker)",
        dataset_root=str(DATASET_ROOT),
        cached_cases_count=cases_count,
        rag_retriever_ready=pipe._retriever is not None,
    )


@app.get("/api/incidents", response_model=List[IncidentSummary])
def list_incidents(
    system: Optional[str] = Query(None, description="Filter by system ('ob', 'ss', 'tt')"),
    limit: int = Query(60, ge=1, le=300),
) -> List[IncidentSummary]:
    """List available benchmark incidents available in the local environment."""
    cases_file = DATASET_ROOT / "cases.parquet"
    if not cases_file.is_file():
        raise HTTPException(status_code=404, detail="cases.parquet not found in dataset cache.")

    import pandas as pd
    df = pd.read_parquet(cases_file)
    if system:
        df = df[df["system"].str.lower() == system.lower()]

    # Filter to cases actually existing in directory
    results: List[IncidentSummary] = []
    for row in df.itertuples():
        c_id = str(row.case)
        if (DATASET_ROOT / c_id).is_dir():
            results.append(
                IncidentSummary(
                    case_id=c_id,
                    dataset=str(row.dataset),
                    system=str(row.system),
                    fault=str(row.fault),
                    root_cause_service=str(row.root_cause_service),
                    duration_minutes=float(getattr(row, "duration_minutes", 10.0)),
                    has_metrics=(DATASET_ROOT / c_id / "metrics.parquet").is_file(),
                    has_traces=(DATASET_ROOT / c_id / "traces.parquet").is_file(),
                    has_logs=(DATASET_ROOT / c_id / "logs.parquet").is_file(),
                )
            )
            if len(results) >= limit:
                break
    return results


@app.get("/api/incidents/{case_id}", response_model=IncidentSummary)
def get_incident_summary(case_id: str) -> IncidentSummary:
    """Get metadata for a single incident case."""
    case_dir = DATASET_ROOT / case_id
    if not case_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"Case directory {case_id} not found.")

    cases_file = DATASET_ROOT / "cases.parquet"
    if cases_file.is_file():
        import pandas as pd
        df = pd.read_parquet(cases_file)
        matching = df[df["case"] == case_id]
        if not matching.empty:
            row = matching.iloc[0]
            return IncidentSummary(
                case_id=case_id,
                dataset=str(row["dataset"]),
                system=str(row["system"]),
                fault=str(row["fault"]),
                root_cause_service=str(row["root_cause_service"]),
                duration_minutes=float(row.get("duration_minutes", 10.0)),
                has_metrics=(case_dir / "metrics.parquet").is_file(),
                has_traces=(case_dir / "traces.parquet").is_file(),
                has_logs=(case_dir / "logs.parquet").is_file(),
            )

    return IncidentSummary(
        case_id=case_id,
        dataset="unknown",
        system="unknown",
        fault="unknown",
        root_cause_service="unknown",
    )


@app.post("/api/investigate", response_model=CanonicalIncident)
def run_investigation(req: InvestigationRequest, request: Request) -> CanonicalIncident:
    """Execute the complete end-to-end RCA, RAG, and Safety pipeline on an incident."""
    client_ip = request.client.host if request.client else "unknown"
    if not _investigate_limiter.check(client_ip):
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded (max {_investigate_limiter.rpm} requests/min). Please slow down.",
        )

    case_dir = DATASET_ROOT / req.case_id
    if not case_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"Case {req.case_id} not found in dataset cache.")

    pipe = get_pipeline()
    try:
        incident = pipe.run(
            case_id=req.case_id,
            model_name=req.model_name,
            budget=req.budget,
            use_bocpd=req.use_bocpd,
            execute_remediation=req.execute_remediation,
        )
        return incident
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Pipeline execution failed: {str(exc)}")


# Server-side registry of executed actions to prevent duplicate execution vulnerabilities
_executed_remediations: set[tuple[str, str, str]] = set()


@app.post("/api/remediation/action", response_model=RemediationActionResponse)
def execute_remediation_action(req: RemediationActionRequest, request: Request) -> RemediationActionResponse:
    """Human-in-the-loop endpoint to explicitly approve, reject, or test remediation actions.

    Enforces server-side safety rules:
    - Checks per-IP rate limiting (HTTP 429).
    - Checks optional REMEDIATION_SECRET authorization token if configured (HTTP 401).
    - Rejects empty, whitespace, or malformed parameters (HTTP 400).
    - Validates action against supported RemediationActionType enum (HTTP 400).
    - Validates target service belongs to the incident's candidate universe (HTTP 400).
    - Rejects duplicate repeated executions of the same action (HTTP 409).
    - Guarantees execution is simulated and never executes live destructive commands.
    """
    client_ip = request.client.host if request.client else "unknown"
    if not _remediation_limiter.check(client_ip):
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded (max {_remediation_limiter.rpm} requests/min). Please wait.",
        )

    # Optional server-side secret protection for remediation endpoint
    rem_secret = os.getenv("REMEDIATION_SECRET")
    if rem_secret:
        auth_hdr = request.headers.get("x-remediation-secret") or request.headers.get("authorization", "")
        if auth_hdr != rem_secret and auth_hdr != f"Bearer {rem_secret}":
            raise HTTPException(
                status_code=401,
                detail="Unauthorized: REMEDIATION_SECRET configured on server. Provide valid X-Remediation-Secret header.",
            )

    from eval.universe import resolve_candidate_universe

    # 1. Parameter presence validation
    case_id = (req.case_id or "").strip()
    action = (req.action or "").strip()
    target_svc = (req.target_service or "").strip()
    user_choice = (req.user_decision or "").strip().lower()

    if not case_id:
        raise HTTPException(status_code=400, detail="Missing or empty required parameter: case_id.")
    if not action:
        raise HTTPException(status_code=400, detail="Missing or empty required parameter: action.")
    if not target_svc:
        raise HTTPException(status_code=400, detail="Missing or empty required parameter: target_service.")

    # 2. Decision validation
    if user_choice not in ("approve", "reject", "dry_run"):
        raise HTTPException(status_code=400, detail="Decision must be 'approve', 'reject', or 'dry_run'.")


    # 3. Action type validation
    supported_actions = {a.value for a in RemediationActionType}
    if action not in supported_actions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported remediation action '{action}'. Supported actions: {sorted(supported_actions)}.",
        )

    # 4. Target service candidate universe validation
    system = "ob"
    if "ss" in case_id.lower():
        system = "ss"
    elif "tt" in case_id.lower():
        system = "tt"
    elif "ob" in case_id.lower():
        system = "ob"

    valid_universe = set(resolve_candidate_universe(system))
    if system == "ob":
        valid_universe.add("frontendservice")  # Alias for frontend

    if target_svc not in valid_universe:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid target service '{target_svc}' for incident system '{system}'. "
                f"Target must belong to candidate universe: {sorted(valid_universe)}."
            ),
        )

    # 5. Handle operator rejection
    if user_choice == "reject":
        return RemediationActionResponse(
            case_id=case_id,
            action=action,
            target_service=target_svc,
            user_decision="reject",
            status=RemediationStatus.REJECTED,
            execution_success=False,
            is_simulation=True,
            audit_log=[
                f"Operator rejected remediation action '{action}' on '{target_svc}'.",
                f"Incident case: {case_id}",
                f"Operator notes: {req.notes or 'No notes provided.'}",
            ],
            message=f"Remediation action '{action}' on {target_svc} was rejected by human operator.",
        )

    # 6. Duplicate execution prevention
    action_key = (case_id, action, target_svc)
    if user_choice == "approve":
        if action_key in _executed_remediations:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"Duplicate remediation rejected: action '{action}' on '{target_svc}' "
                    f"has already been executed for incident '{case_id}'."
                ),
            )
        _executed_remediations.add(action_key)

    # 7. Authorized simulated execution
    status = RemediationStatus.EXECUTED if user_choice == "approve" else RemediationStatus.APPROVED
    rollback_hook = f"rollback_{action}({target_svc})"
    audit = [
        f"Operator decision: {user_choice.upper()} for action '{action}' targeting '{target_svc}'.",
        f"Server-side candidate universe check: verified '{target_svc}' is member of {system} topology.",
        f"Rollback mechanism registered: {rollback_hook}.",
        "Simulation sandboxing: execution is strictly in-memory simulation; live Kubernetes/cloud infrastructure is not modified.",
        f"Simulated execution {'succeeded' if user_choice == 'approve' else 'verified dry-run'}. Grace period: 30s.",
    ]
    return RemediationActionResponse(
        case_id=case_id,
        action=action,
        target_service=target_svc,
        user_decision=user_choice,
        status=status,
        execution_success=True,
        is_simulation=True,
        audit_log=audit,
        message=(
            f"Remediation action '{action}' on {target_svc} simulated and marked as {status.value}. "
            "Real infrastructure operations were NOT executed."
        ),
    )


@app.post("/api/rag/query", response_model=List[RAGQueryResult])
def query_rag(req: RAGQueryRequest) -> List[RAGQueryResult]:
    """Direct search endpoint over the operational runbook corpus."""
    pipe = get_pipeline()
    if pipe._retriever is None:
        raise HTTPException(status_code=503, detail="RAG retriever is not initialized.")

    results = pipe._retriever.retrieve(
        query=req.query,
        k=req.k,
        target_category=req.category,
        target_service=req.target_service,
    )

    output = []
    for rank_idx, res in enumerate(results, 1):
        doc = pipe._retriever.doc_map.get(res.document_id)
        cat = doc.metadata.get("knowledge_category", "general") if doc else "general"
        output.append(
            RAGQueryResult(
                citation_id=f"cit-{rank_idx}",
                document_id=res.document_id,
                title=res.title,
                source=res.metadata.get("source", "knowledge_base"),
                section=res.metadata.get("section", "General"),
                score=round(res.score, 4),
                snippet=res.snippet,
                category=cat,
            )
        )
    return output


# ---------------------------------------------------------------------------
# Static Web UI Hosting
# ---------------------------------------------------------------------------
WEB_DIR = Path(__file__).resolve().parents[1] / "web"
if WEB_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")


def main() -> None:
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    print(f"Starting Digital Detective API on http://0.0.0.0:{port}")
    uvicorn.run("digital_detective.api.main:app", host="0.0.0.0", port=port, reload=False)



if __name__ == "__main__":
    main()
