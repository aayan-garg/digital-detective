/**
 * Digital Detective — Web Replay Dashboard Application
 * Connects to FastAPI backend (/api/...) for live telemetry, RCA, RAG, and Safe Remediation.
 */

// Dynamic hierarchical API Base resolution:
// 1. URL query (?api=https://...)
// 2. localStorage ("DIGITAL_DETECTIVE_API_BASE")
// 3. window.__DIGITAL_DETECTIVE_API_BASE__ (config.js)
// 4. Fallback to window.location.origin
function resolveApiBase() {
  const urlParams = new URLSearchParams(window.location.search);
  const paramApi = urlParams.get("api") || urlParams.get("api_base");
  if (paramApi) {
    const clean = paramApi.trim().replace(/\/+$/, "");
    localStorage.setItem("DIGITAL_DETECTIVE_API_BASE", clean);
    return clean;
  }

  const saved = localStorage.getItem("DIGITAL_DETECTIVE_API_BASE");
  if (saved && saved.trim()) {
    return saved.trim().replace(/\/+$/, "");
  }

  if (window.__DIGITAL_DETECTIVE_API_BASE__ && typeof window.__DIGITAL_DETECTIVE_API_BASE__ === "string") {
    const configured = window.__DIGITAL_DETECTIVE_API_BASE__.trim().replace(/\/+$/, "");
    if (configured) return configured;
  }

  return window.location.origin;
}

let API_BASE = resolveApiBase();

// State container
let currentIncident = null;
let allIncidents = [];

// DOM Elements
const systemFilterEl = document.getElementById("system-filter");
const incidentSelectEl = document.getElementById("incident-select");
const modelSelectEl = document.getElementById("model-select");
const btnInvestigate = document.getElementById("btn-investigate");
const statusBanner = document.getElementById("status-banner");
const statusText = document.getElementById("status-text");
const execTimeBadge = document.getElementById("execution-time-badge");

// API Connection Elements
const apiConnDot = document.getElementById("api-conn-dot");
const apiConnLabel = document.getElementById("api-conn-label");
const btnConfigureApi = document.getElementById("btn-configure-api");

// KPI Elements
const kpiOnsetTime = document.getElementById("kpi-onset-time");
const kpiOnsetSub = document.getElementById("kpi-onset-sub");
const onsetMethodBadge = document.getElementById("onset-method");
const kpiRootCause = document.getElementById("kpi-root-cause");
const diagnosisConfidence = document.getElementById("diagnosis-confidence");
const inferenceProvenanceBadge = document.getElementById("inference-provenance-badge");
const kpiGroundTruthVal = document.getElementById("kpi-ground-truth-val");
const kpiMarginVal = document.getElementById("kpi-margin-value");
const marginTriggerBadge = document.getElementById("margin-trigger-badge");
const kpiRemAction = document.getElementById("kpi-remediation-action");
const safetyStatusBadge = document.getElementById("safety-status-badge");

// Content Containers
const topologyView = document.getElementById("topology-view");
const candidatesContainer = document.getElementById("candidates-container");
const ragContainer = document.getElementById("rag-container");
const citationsCountBadge = document.getElementById("citations-count");
const actionLogBox = document.getElementById("action-log-box");

// Remediation proposal & buttons
const proposalAction = document.getElementById("proposal-action");
const proposalTarget = document.getElementById("proposal-target");
const proposalRollback = document.getElementById("proposal-rollback");
const proposalVerdict = document.getElementById("proposal-verdict");
const btnApproveRem = document.getElementById("btn-approve-rem");
const btnDryRunRem = document.getElementById("btn-dry-run-rem");
const btnRejectRem = document.getElementById("btn-reject-rem");

// ==========================================================================
// Initialization & Incident Catalogue
// ==========================================================================

async function init() {
  updateApiStatusUI();
  await checkBackendConnection();
  await loadIncidents();
  bindEvents();
}

function updateApiStatusUI() {
  if (!apiConnLabel) return;
  try {
    const url = new URL(API_BASE);
    apiConnLabel.textContent = url.hostname;
    apiConnLabel.title = `Connected to: ${API_BASE}`;
  } catch {
    apiConnLabel.textContent = "Local";
  }
}

async function checkBackendConnection() {
  if (!apiConnDot) return;
  try {
    const res = await fetch(`${API_BASE}/healthz`, { method: "GET" });
    if (res.ok) {
      apiConnDot.className = "status-dot api-dot connected";
    } else {
      apiConnDot.className = "status-dot api-dot error";
    }
  } catch {
    // Try fallback /api/health
    try {
      const res2 = await fetch(`${API_BASE}/api/health`);
      apiConnDot.className = res2.ok ? "status-dot api-dot connected" : "status-dot api-dot error";
    } catch {
      apiConnDot.className = "status-dot api-dot error";
    }
  }
}

async function loadIncidents() {
  try {
    statusText.textContent = "Loading incident catalogue from backend...";
    const res = await fetch(`${API_BASE}/api/incidents?limit=90`);
    if (!res.ok) throw new Error(`HTTP ${res.status} from ${API_BASE}`);
    allIncidents = await res.json();
    populateIncidentDropdown();
    statusText.textContent = `Ready. Loaded ${allIncidents.length} benchmark incidents from backend.`;
    if (apiConnDot) apiConnDot.className = "status-dot api-dot connected";
  } catch (err) {
    statusText.textContent = `Error connecting to backend (${API_BASE}): ${err.message}`;
    if (apiConnDot) apiConnDot.className = "status-dot api-dot error";
    logAction(`[Connection Error] Failed to reach ${API_BASE}/api/incidents: ${err.message}`);
  }
}

function populateIncidentDropdown() {
  const selectedSystem = systemFilterEl.value.toLowerCase();
  const filtered = selectedSystem
    ? allIncidents.filter(inc => inc.system.toLowerCase() === selectedSystem)
    : allIncidents;

  incidentSelectEl.innerHTML = "";
  if (filtered.length === 0) {
    incidentSelectEl.innerHTML = `<option value="">No matching incidents found</option>`;
    btnInvestigate.disabled = true;
    return;
  }

  filtered.forEach(inc => {
    const opt = document.createElement("option");
    opt.value = inc.case_id;
    opt.textContent = `${inc.case_id} [${inc.system.toUpperCase()} - ${inc.fault}]`;
    incidentSelectEl.appendChild(opt);
  });
  btnInvestigate.disabled = false;
}

function bindEvents() {
  systemFilterEl.addEventListener("change", populateIncidentDropdown);
  btnInvestigate.addEventListener("click", runInvestigation);

  btnApproveRem.addEventListener("click", () => handleRemediationDecision("approve"));
  btnDryRunRem.addEventListener("click", () => handleRemediationDecision("dry_run"));
  btnRejectRem.addEventListener("click", () => handleRemediationDecision("reject"));

  if (btnConfigureApi) {
    btnConfigureApi.addEventListener("click", () => {
      const current = localStorage.getItem("DIGITAL_DETECTIVE_API_BASE") || API_BASE;
      const next = prompt(
        "Enter Digital Detective FastAPI Backend URL (e.g. https://digital-detective-api.onrender.com or leave blank to reset):",
        current
      );
      if (next !== null) {
        if (next.trim() === "") {
          localStorage.removeItem("DIGITAL_DETECTIVE_API_BASE");
        } else {
          localStorage.setItem("DIGITAL_DETECTIVE_API_BASE", next.trim().replace(/\/+$/, ""));
        }
        API_BASE = resolveApiBase();
        updateApiStatusUI();
        checkBackendConnection();
        loadIncidents();
      }
    });
  }

  const btnCloseInspector = document.getElementById("btn-close-inspector");
  if (btnCloseInspector) {
    btnCloseInspector.addEventListener("click", () => {
      document.getElementById("node-inspector").classList.add("hidden");
    });
  }
}


// ==========================================================================
// Investigation Execution
// ==========================================================================

async function runInvestigation() {
  const caseId = incidentSelectEl.value;
  if (!caseId) return;

  const modelChoice = modelSelectEl.value;

  // Set loading state
  btnInvestigate.disabled = true;
  statusBanner.classList.add("running");
  statusText.textContent = `Investigating ${caseId} with multi-modal causal inference...`;
  execTimeBadge.classList.add("hidden");

  try {
    const t0 = performance.now();
    const res = await fetch(`${API_BASE}/api/investigate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        case_id: caseId,
        model_name: modelChoice,
        use_bocpd: false,
        execute_remediation: false,
      }),
    });

    if (!res.ok) {
      let detail = `HTTP ${res.status}`;
      try {
        const err = await res.json();
        detail = err.detail || detail;
      } catch {
        const text = await res.text();
        if (text) detail = `${detail}: ${text.slice(0, 120)}`;
      }
      throw new Error(detail);
    }

    currentIncident = await res.json();
    const elapsedSec = ((performance.now() - t0) / 1000).toFixed(2);

    renderInvestigation(currentIncident);

    statusBanner.classList.remove("running");
    statusText.textContent = `Investigation complete for ${caseId}. Root Cause: ${currentIncident.ranking?.selected_root_cause}`;
    execTimeBadge.textContent = `${elapsedSec}s`;
    execTimeBadge.classList.remove("hidden");
  } catch (err) {
    statusBanner.classList.remove("running");
    statusText.textContent = `Investigation error: ${err.message}`;
    logAction(`[Error] ${err.message}`);
  } finally {
    btnInvestigate.disabled = false;
  }
}

// ==========================================================================
// View Rendering
// ==========================================================================

function renderInvestigation(data) {
  // 1. KPI Cards
  const cp = data.changepoint;
  if (cp) {
    kpiOnsetTime.textContent = cp.onset_timestamp ? new Date(cp.onset_timestamp * 1000).toLocaleTimeString() : "Estimated";
    kpiOnsetSub.textContent = `Boundary: ${cp.onset_timestamp || "--"}`;
    onsetMethodBadge.textContent = (cp.detection_method || "STREAMING").toUpperCase();
  }

  const rk = data.ranking;
  if (rk) {
    kpiRootCause.textContent = rk.selected_root_cause;
    const inv = data.investigation || data.investigation_findings;
    const confVal = inv?.confidence !== undefined ? inv.confidence : 0.85;
    diagnosisConfidence.textContent = `${Math.round(confVal * 100)}% CONFIDENCE`;
    kpiMarginVal.textContent = rk.score_margin !== undefined ? rk.score_margin.toFixed(4) : "--";
    marginTriggerBadge.textContent = rk.uncertainty_detected ? "MARGIN TRIGGERED" : "DETERMINISTIC";
    marginTriggerBadge.className = rk.uncertainty_detected ? "kpi-badge badge-warning" : "kpi-badge badge-success";

    if (inferenceProvenanceBadge) {
      if (inv?.fallback_active) {
        inferenceProvenanceBadge.textContent = "FALLBACK MOCK";
        inferenceProvenanceBadge.className = "kpi-badge badge-provenance fallback";
        inferenceProvenanceBadge.title = `Fallback active: ${inv.fallback_reason || "Live inference unreachable"}`;
      } else if (inv?.is_mock) {
        inferenceProvenanceBadge.textContent = "MOCK / MODEL B";
        inferenceProvenanceBadge.className = "kpi-badge badge-provenance";
        inferenceProvenanceBadge.title = "Deterministic engine + Supervised Model B pairwise ranker";
      } else {
        const mName = (inv?.model_name || "AGENT").toUpperCase();
        inferenceProvenanceBadge.textContent = `LIVE: ${mName}`;
        inferenceProvenanceBadge.className = "kpi-badge badge-provenance live";
        inferenceProvenanceBadge.title = `Live agent inference: ${inv?.model_name}`;
      }
    }
  }


  kpiGroundTruthVal.textContent = data.ground_truth_root_cause
    ? `${data.ground_truth_root_cause} (${data.ground_truth_fault_type || "fault"})`
    : "Unlabeled benchmark case";

  // Check Ground Truth match
  if (data.ground_truth_root_cause && rk) {
    const isMatch = rk.selected_root_cause === data.ground_truth_root_cause;
    kpiGroundTruthVal.style.color = isMatch ? "#10B981" : "#F59E0B";
  }

  const rem = data.remediation_plan;
  if (rem) {
    kpiRemAction.textContent = rem.action;
    safetyStatusBadge.textContent = rem.safety_check.safety_authorized ? "SAFETY VERIFIED" : "APPROVAL REQUIRED";
    safetyStatusBadge.className = rem.safety_check.safety_authorized ? "kpi-badge badge-success" : "kpi-badge badge-warning";

    proposalAction.textContent = rem.action;
    proposalTarget.textContent = rem.target_service;
    proposalRollback.textContent = rem.rollback_action;
    proposalVerdict.textContent = data.recovery_verification?.causal_verdict || "INTERVENTION_INCONCLUSIVE";
    proposalVerdict.className = `badge ${data.recovery_verification?.recovery_status === "resolved" ? "badge-success" : "badge-warning"}`;

    // Enable buttons
    btnApproveRem.disabled = false;
    btnDryRunRem.disabled = false;
    btnRejectRem.disabled = false;

    // Tick safety checks
    const sc = rem.safety_check;
    document.getElementById("check-supported").className = `check-item ${sc.supported_action ? "passed" : ""}`;
    document.getElementById("check-rollback").className = `check-item ${sc.has_rollback ? "passed" : ""}`;
    document.getElementById("check-universe").className = `check-item ${sc.target_in_universe ? "passed" : ""}`;
    document.getElementById("check-rootmatch").className = `check-item ${sc.target_matches_root_cause ? "passed" : ""}`;
    document.getElementById("check-degraded").className = `check-item ${sc.target_shows_degradation ? "passed" : ""}`;
    document.getElementById("check-confidence").className = `check-item ${sc.confidence_above_threshold ? "passed" : ""}`;
  }

  // 2. Render Topology SVG
  renderTopologySVG(data.topology, rk?.selected_root_cause);

  // 3. Render Candidate Ranking List
  renderCandidatesList(rk?.candidates || []);

  // 4. Render RAG Runbooks
  renderRAGRunbooks(data.rag_context);

  logAction(`[Pipeline] Investigation loaded for ${data.incident_id}. Selected candidate: ${rk?.selected_root_cause}.`);
}

function renderTopologySVG(topology, selectedRootCause) {
  if (!topology || !topology.nodes || topology.nodes.length === 0) {
    topologyView.innerHTML = `<div class="empty-state"><p>No topology data available</p></div>`;
    return;
  }

  const nodes = topology.nodes;
  const edges = topology.edges || [];
  const width = topologyView.clientWidth || 700;
  const height = 340;
  const centerX = width / 2;
  const centerY = height / 2;
  const radius = Math.min(width, height) * 0.38;

  // Calculate circular layout positions
  const nodeCoords = {};
  nodes.forEach((node, idx) => {
    const angle = (idx / nodes.length) * 2 * Math.PI - Math.PI / 2;
    nodeCoords[node.service_id] = {
      x: centerX + radius * Math.cos(angle),
      y: centerY + radius * Math.sin(angle),
      node: node,
    };
  });

  // Build SVG XML
  let svgHtml = `<svg width="100%" height="${height}" viewBox="0 0 ${width} ${height}" xmlns="http://www.w3.org/2000/svg">`;
  svgHtml += `<defs>
    <marker id="arrow" viewBox="0 0 10 10" refX="22" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1 L 10 5 L 0 9 z" fill="rgba(148, 163, 184, 0.4)"/>
    </marker>
    <filter id="glow-rose">
      <feGaussianBlur stdDeviation="4" result="coloredBlur"/>
      <feMerge><feMergeNode in="coloredBlur"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
  </defs>`;

  // Draw Edges
  edges.forEach(edge => {
    const src = nodeCoords[edge.caller];
    const dst = nodeCoords[edge.callee];
    if (src && dst) {
      svgHtml += `<line x1="${src.x}" y1="${src.y}" x2="${dst.x}" y2="${dst.y}" stroke="rgba(148, 163, 184, 0.25)" stroke-width="1.5" marker-end="url(#arrow)" />`;
    }
  });

  // Draw Nodes
  nodes.forEach(node => {
    const pos = nodeCoords[node.service_id];
    if (!pos) return;

    const isRoot = node.service_id === selectedRootCause;
    const isAnom = node.is_anomaly;

    let fill = "#1E293B";
    let stroke = "#64748B";
    let filter = "";

    if (isRoot) {
      fill = "#F43F5E";
      stroke = "#FDA4AF";
      filter = 'filter="url(#glow-rose)"';
    } else if (isAnom) {
      fill = "#D97706";
      stroke = "#FCD34D";
    }

    svgHtml += `
      <g class="topo-node" style="cursor: pointer;" onclick="window.inspectNode('${node.service_id}')">
        <circle cx="${pos.x}" cy="${pos.y}" r="${isRoot ? 14 : 10}" fill="${fill}" stroke="${stroke}" stroke-width="2" ${filter}/>
        ${isRoot ? `<circle cx="${pos.x}" cy="${pos.y}" r="20" fill="none" stroke="#F43F5E" stroke-width="1.5" opacity="0.6">
          <animate attributeName="r" values="14;24;14" dur="2s" repeatCount="indefinite"/>
          <animate attributeName="opacity" values="0.8;0;0.8" dur="2s" repeatCount="indefinite"/>
        </circle>` : ''}
        <text x="${pos.x}" y="${pos.y + (isRoot ? 24 : 20)}" fill="#F8FAFC" font-size="10" font-family="'JetBrains Mono', monospace" text-anchor="middle">
          ${node.service_id}
        </text>
      </g>
    `;
  });

  svgHtml += `</svg>`;
  topologyView.innerHTML = svgHtml;
}

function renderCandidatesList(candidates) {
  if (!candidates || candidates.length === 0) {
    candidatesContainer.innerHTML = `<div class="empty-state"><p>No candidates available</p></div>`;
    return;
  }

  const top3 = candidates.slice(0, 3);
  let html = "";

  top3.forEach(c => {
    const isRank1 = c.rank === 1;
    const widthPct = Math.min(100, Math.max(10, c.score * 100));

    html += `
      <div class="candidate-card ${isRank1 ? 'rank-1' : ''}">
        <div class="cand-top-row">
          <div class="cand-name-wrap">
            <span class="cand-rank">${c.rank}</span>
            <span class="cand-name">${c.service}</span>
          </div>
          <span class="cand-score">${c.score.toFixed(4)}</span>
        </div>
        <div class="cand-progress-bar">
          <div class="cand-progress-fill" style="width: ${widthPct}%;"></div>
        </div>
        <div class="cand-evidence-tags">
          <span class="ev-tag">s_comb: ${c.s_comb_score !== null ? c.s_comb_score.toFixed(2) : '--'}</span>
          <span class="ev-tag ${c.trace_elev_score > 0.5 ? 'elevated' : ''}">trace_elev: ${c.trace_elev_score !== null ? c.trace_elev_score.toFixed(2) : '--'}</span>
          <span class="ev-tag">causal_z: ${c.causal_residual_z !== null ? c.causal_residual_z.toFixed(2) : '--'}</span>
          <span class="ev-tag">log_anom: ${c.log_anomaly_score !== null ? c.log_anomaly_score.toFixed(2) : '--'}</span>
        </div>
      </div>
    `;
  });

  candidatesContainer.innerHTML = html;
}

function renderRAGRunbooks(ragContext) {
  if (!ragContext || !ragContext.citations || ragContext.citations.length === 0) {
    ragContainer.innerHTML = `<div class="empty-state"><p>No runbook evidence retrieved</p></div>`;
    citationsCountBadge.textContent = "0 Citations";
    return;
  }

  const citations = ragContext.citations;
  citationsCountBadge.textContent = `${citations.length} Grounded Citations`;
  let html = "";

  citations.forEach(cit => {
    html += `
      <div class="citation-card">
        <div class="cit-header">
          <span class="cit-title">${cit.title}</span>
          <span class="cit-score-badge">score: ${cit.score.toFixed(4)}</span>
        </div>
        <div class="cit-meta">
          <span>Source: ${cit.source}</span> &bull; <span>Section: ${cit.section}</span> &bull; <span>Category: ${cit.category}</span>
        </div>
        <div class="cit-snippet">${cit.snippet}</div>
      </div>
    `;
  });

  ragContainer.innerHTML = html;
}

// ==========================================================================
// Interactive Actions & Node Inspection
// ==========================================================================

window.inspectNode = function(serviceName) {
  if (!currentIncident) return;
  const drawer = document.getElementById("node-inspector");
  const title = document.getElementById("inspector-service-name");
  const content = document.getElementById("inspector-content");

  title.textContent = serviceName;
  const ep = currentIncident.topology?.nodes?.find(n => n.service_id === serviceName);
  const cand = currentIncident.ranking?.candidates?.find(c => c.service === serviceName);

  content.innerHTML = `
    <div style="font-size: 12px; display: flex; flex-direction: column; gap: 6px; padding: 10px 0;">
      <div><strong>Anomaly Status:</strong> ${ep?.is_anomaly ? '<span style="color:#F59E0B">DEGRADED</span>' : '<span style="color:#10B981">HEALTHY</span>'}</div>
      <div><strong>Active Anomaly Score:</strong> ${ep?.anomaly_score?.toFixed(2) || '0.00'}</div>
      <div><strong>Active Metrics:</strong> ${ep?.active_metrics_count || 0}</div>
      <div><strong>Rank Position:</strong> ${cand ? `#${cand.rank} (Score: ${cand.score.toFixed(4)})` : 'Not in top ranked'}</div>
      <div><strong>Causal Residual Z:</strong> ${cand?.causal_residual_z?.toFixed(2) || '0.00'}</div>
    </div>
  `;
  drawer.classList.remove("hidden");
};

async function handleRemediationDecision(decision) {
  if (!currentIncident) return;

  const rem = currentIncident.remediation_plan;
  if (!rem) return;

  try {
    const res = await fetch(`${API_BASE}/api/remediation/action`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        case_id: currentIncident.incident_id,
        action: rem.action,
        target_service: rem.target_service,
        user_decision: decision,
      }),
    });

    const data = await res.json();
    logAction(`[Remediation ${decision.toUpperCase()}] ${data.message}`);
    if (data.audit_log) {
      data.audit_log.forEach(entry => logAction(`  -> ${entry}`));
    }

    if (decision === "approve") {
      safetyStatusBadge.textContent = "SIMULATION EXECUTED";
      safetyStatusBadge.className = "kpi-badge badge-success";
    } else if (decision === "reject") {
      safetyStatusBadge.textContent = "REJECTED BY OPERATOR";
      safetyStatusBadge.className = "kpi-badge badge-danger";
    }
  } catch (err) {
    logAction(`[Remediation Error] ${err.message}`);
  }
}

function logAction(msg) {
  const entry = document.createElement("div");
  entry.className = "log-entry";
  entry.textContent = `[${new Date().toLocaleTimeString()}] ${msg}`;
  actionLogBox.appendChild(entry);
  actionLogBox.scrollTop = actionLogBox.scrollHeight;
}

// Start application
document.addEventListener("DOMContentLoaded", init);
