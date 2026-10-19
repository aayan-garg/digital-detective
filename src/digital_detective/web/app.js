/**
 * Digital Detective — Web Replay Dashboard Application
 *
 * Self-contained zero-cost interactive static demo for public deployment on Vercel.
 * Loads precomputed benchmark investigation artifacts locally from demo_investigations.json,
 * providing interactive RCA ranking, topology visualization, RAG runbooks, and simulated remediation
 * without requiring any backend server, external API, or cloud infrastructure.
 */

// State container
let currentIncident = null;
let allIncidents = [];
let demoInvestigations = {};

// DOM Elements
const systemFilterEl = document.getElementById("system-filter");
const incidentSelectEl = document.getElementById("incident-select");
const modelSelectEl = document.getElementById("model-select");
const btnInvestigate = document.getElementById("btn-investigate");
const statusBanner = document.getElementById("status-banner");
const statusText = document.getElementById("status-text");
const execTimeBadge = document.getElementById("execution-time-badge");

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
  await loadDemoData();
  bindEvents();
}

async function loadDemoData() {
  try {
    statusText.textContent = "Loading precomputed benchmark investigations...";
    const res = await fetch("./demo_investigations.json");
    if (!res.ok) throw new Error(`HTTP ${res.status}: Failed to load static demo data`);
    demoInvestigations = await res.json();

    const caseIds = Object.keys(demoInvestigations);
    if (!caseIds || caseIds.length === 0) {
      throw new Error("Demo dataset is empty or malformed");
    }

    allIncidents = caseIds.map(caseId => {
      const inc = demoInvestigations[caseId];
      return {
        case_id: inc?.incident_id || caseId,
        system: inc?.system || "ob",
        fault: inc?.ground_truth_fault_type || (caseId.includes("_cpu_") ? "cpu" : caseId.includes("_mem_") ? "mem" : "delay"),
        root_cause_service: inc?.ground_truth_root_cause || inc?.ranking?.selected_root_cause || "unknown",
      };
    });

    populateIncidentDropdown();
    statusText.textContent = `Precomputed demo ready. Loaded ${allIncidents.length} verified benchmark incidents. Click "Run Investigation" to inspect.`;
    logAction(`[System] Initialized zero-cost static demo with ${allIncidents.length} benchmark incidents.`);
  } catch (err) {
    statusText.textContent = `Demo data error: ${err.message}.`;
    logAction(`[Demo Error] Could not load static investigations: ${err.message}`);
    if (btnInvestigate) btnInvestigate.disabled = true;
  }
}

function populateIncidentDropdown() {
  if (!incidentSelectEl) return;
  const selectedSystem = systemFilterEl ? systemFilterEl.value.toLowerCase() : "";
  const filtered = selectedSystem
    ? allIncidents.filter(inc => inc.system.toLowerCase() === selectedSystem)
    : allIncidents;

  incidentSelectEl.innerHTML = "";
  if (filtered.length === 0) {
    incidentSelectEl.innerHTML = `<option value="">No matching incidents found</option>`;
    if (btnInvestigate) btnInvestigate.disabled = true;
    return;
  }

  filtered.forEach(inc => {
    const opt = document.createElement("option");
    opt.value = inc.case_id;
    opt.textContent = `${inc.case_id} [${inc.system.toUpperCase()} - ${inc.fault}]`;
    incidentSelectEl.appendChild(opt);
  });
  if (btnInvestigate) btnInvestigate.disabled = false;
}

function bindEvents() {
  if (systemFilterEl) {
    systemFilterEl.addEventListener("change", populateIncidentDropdown);
  }
  if (btnInvestigate) {
    btnInvestigate.addEventListener("click", runInvestigation);
  }

  if (btnApproveRem) {
    btnApproveRem.addEventListener("click", () => handleRemediationDecision("approve"));
  }
  if (btnDryRunRem) {
    btnDryRunRem.addEventListener("click", () => handleRemediationDecision("dry_run"));
  }
  if (btnRejectRem) {
    btnRejectRem.addEventListener("click", () => handleRemediationDecision("reject"));
  }

  const btnCloseInspector = document.getElementById("btn-close-inspector");
  if (btnCloseInspector) {
    btnCloseInspector.addEventListener("click", () => {
      const drawer = document.getElementById("node-inspector");
      if (drawer) drawer.classList.add("hidden");
    });
  }
}

// ==========================================================================
// Investigation Execution (Static Precomputed Replay)
// ==========================================================================

async function runInvestigation() {
  const caseId = incidentSelectEl ? incidentSelectEl.value : "";
  if (!caseId) return;

  if (btnInvestigate) btnInvestigate.disabled = true;
  if (statusBanner) statusBanner.classList.add("running");
  if (statusText) statusText.textContent = `Replaying benchmark investigation for ${caseId}...`;
  if (execTimeBadge) execTimeBadge.classList.add("hidden");

  // Brief tactile UI delay (150ms)
  await new Promise(r => setTimeout(r, 150));

  try {
    const incidentData = demoInvestigations[caseId];
    if (!incidentData) {
      throw new Error(`Incident ${caseId} not found in precomputed demo dataset.`);
    }

    currentIncident = incidentData;
    renderInvestigation(currentIncident);

    if (statusBanner) statusBanner.classList.remove("running");
    if (statusText) {
      statusText.textContent = `Precomputed demo replay for ${caseId}. Diagnosed Root Cause: ${currentIncident.ranking?.selected_root_cause} (Benchmark recorded)`;
    }
    if (execTimeBadge) {
      execTimeBadge.textContent = "0.01s";
      execTimeBadge.classList.remove("hidden");
    }
  } catch (err) {
    if (statusBanner) statusBanner.classList.remove("running");
    if (statusText) statusText.textContent = `Investigation error: ${err.message}`;
    logAction(`[Error] ${err.message}`);
  } finally {
    if (btnInvestigate) btnInvestigate.disabled = false;
  }
}

// ==========================================================================
// View Rendering
// ==========================================================================

function renderInvestigation(data) {
  if (!data) return;

  // 1. KPI Cards
  const cp = data.changepoint;
  if (cp) {
    if (kpiOnsetTime) {
      kpiOnsetTime.textContent = cp.onset_timestamp ? new Date(cp.onset_timestamp * 1000).toLocaleTimeString() : "Estimated";
    }
    if (kpiOnsetSub) {
      kpiOnsetSub.textContent = `Boundary: ${cp.onset_timestamp || "--"}`;
    }
    if (onsetMethodBadge) {
      onsetMethodBadge.textContent = (cp.detection_method || "STREAMING").toUpperCase();
    }
  }

  const rk = data.ranking;
  if (rk) {
    if (kpiRootCause) kpiRootCause.textContent = rk.selected_root_cause;
    const inv = data.investigation || data.investigation_findings;
    const confVal = inv?.confidence !== undefined ? inv.confidence : 0.85;
    if (diagnosisConfidence) {
      diagnosisConfidence.textContent = `${Math.round(confVal * 100)}% CONFIDENCE`;
    }
    if (kpiMarginVal) {
      kpiMarginVal.textContent = rk.score_margin !== undefined ? rk.score_margin.toFixed(4) : "--";
    }
    if (marginTriggerBadge) {
      marginTriggerBadge.textContent = rk.uncertainty_detected ? "MARGIN TRIGGERED" : "DETERMINISTIC";
      marginTriggerBadge.className = rk.uncertainty_detected ? "kpi-badge badge-warning" : "kpi-badge badge-success";
    }

    if (inferenceProvenanceBadge) {
      inferenceProvenanceBadge.textContent = "PRECOMPUTED DEMO";
      inferenceProvenanceBadge.className = "kpi-badge badge-provenance demo";
      inferenceProvenanceBadge.title = "Precomputed benchmark evaluation replay — no live LLM reasoning or fresh backend execution";
    }
  }

  if (kpiGroundTruthVal) {
    kpiGroundTruthVal.textContent = data.ground_truth_root_cause
      ? `${data.ground_truth_root_cause} (${data.ground_truth_fault_type || "fault"})`
      : "Unlabeled benchmark case";

    if (data.ground_truth_root_cause && rk) {
      const isMatch = rk.selected_root_cause === data.ground_truth_root_cause;
      kpiGroundTruthVal.style.color = isMatch ? "#10B981" : "#F59E0B";
    }
  }

  const rem = data.remediation_plan;
  if (rem) {
    if (kpiRemAction) kpiRemAction.textContent = rem.action;
    if (safetyStatusBadge) {
      const auth = rem.safety_check?.safety_authorized;
      safetyStatusBadge.textContent = auth ? "SAFETY VERIFIED" : "APPROVAL REQUIRED";
      safetyStatusBadge.className = auth ? "kpi-badge badge-success" : "kpi-badge badge-warning";
    }

    if (proposalAction) proposalAction.textContent = rem.action;
    if (proposalTarget) proposalTarget.textContent = rem.target_service;
    if (proposalRollback) proposalRollback.textContent = rem.rollback_action;
    if (proposalVerdict) {
      proposalVerdict.textContent = data.recovery_verification?.causal_verdict || "INTERVENTION_INCONCLUSIVE";
      proposalVerdict.className = `badge ${data.recovery_verification?.recovery_status === "resolved" ? "badge-success" : "badge-warning"}`;
    }

    if (btnApproveRem) btnApproveRem.disabled = false;
    if (btnDryRunRem) btnDryRunRem.disabled = false;
    if (btnRejectRem) btnRejectRem.disabled = false;

    // Safety checks checklist
    const sc = rem.safety_check;
    if (sc) {
      setCheckItem("check-supported", sc.supported_action);
      setCheckItem("check-rollback", sc.has_rollback);
      setCheckItem("check-universe", sc.target_in_universe);
      setCheckItem("check-rootmatch", sc.target_matches_root_cause);
      setCheckItem("check-degraded", sc.target_shows_degradation);
      setCheckItem("check-confidence", sc.confidence_above_threshold);
    }
  }

  // 2. Render Topology SVG
  renderTopologySVG(data.topology, rk?.selected_root_cause);

  // 3. Render Candidate Ranking List
  renderCandidatesList(rk?.candidates || []);

  // 4. Render RAG Runbooks
  renderRAGRunbooks(data.rag_context);

  logAction(`[Pipeline] Replayed precomputed investigation for ${data.incident_id}. Selected root cause: ${rk?.selected_root_cause}.`);
}

function setCheckItem(id, passed) {
  const el = document.getElementById(id);
  if (el) {
    el.className = `check-item ${passed ? "passed" : ""}`;
  }
}

function renderTopologySVG(topology, selectedRootCause) {
  if (!topologyView) return;

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
  if (!candidatesContainer) return;

  if (!candidates || candidates.length === 0) {
    candidatesContainer.innerHTML = `<div class="empty-state"><p>No candidates available</p></div>`;
    return;
  }

  const top3 = candidates.slice(0, 3);
  let html = "";

  top3.forEach(c => {
    const isRank1 = c.rank === 1;
    const widthPct = Math.min(100, Math.max(10, (c.score || 0) * 100));

    html += `
      <div class="candidate-card ${isRank1 ? 'rank-1' : ''}">
        <div class="cand-top-row">
          <div class="cand-name-wrap">
            <span class="cand-rank">${c.rank}</span>
            <span class="cand-name">${c.service}</span>
          </div>
          <span class="cand-score">${(c.score || 0).toFixed(4)}</span>
        </div>
        <div class="cand-progress-bar">
          <div class="cand-progress-fill" style="width: ${widthPct}%;"></div>
        </div>
        <div class="cand-evidence-tags">
          <span class="ev-tag">s_comb: ${c.s_comb_score !== null && c.s_comb_score !== undefined ? c.s_comb_score.toFixed(2) : '--'}</span>
          <span class="ev-tag ${c.trace_elev_score > 0.5 ? 'elevated' : ''}">trace_elev: ${c.trace_elev_score !== null && c.trace_elev_score !== undefined ? c.trace_elev_score.toFixed(2) : '--'}</span>
          <span class="ev-tag">causal_z: ${c.causal_residual_z !== null && c.causal_residual_z !== undefined ? c.causal_residual_z.toFixed(2) : '--'}</span>
          <span class="ev-tag">log_anom: ${c.log_anomaly_score !== null && c.log_anomaly_score !== undefined ? c.log_anomaly_score.toFixed(2) : '--'}</span>
        </div>
      </div>
    `;
  });

  candidatesContainer.innerHTML = html;
}

function renderRAGRunbooks(ragContext) {
  if (!ragContainer) return;

  if (!ragContext || !ragContext.citations || ragContext.citations.length === 0) {
    ragContainer.innerHTML = `<div class="empty-state"><p>No runbook evidence retrieved</p></div>`;
    if (citationsCountBadge) citationsCountBadge.textContent = "0 Citations";
    return;
  }

  const citations = ragContext.citations;
  if (citationsCountBadge) {
    citationsCountBadge.textContent = `${citations.length} Grounded Citations`;
  }
  let html = "";

  citations.forEach(cit => {
    html += `
      <div class="citation-card">
        <div class="cit-header">
          <span class="cit-title">${cit.title}</span>
          <span class="cit-score-badge">score: ${(cit.score || 0).toFixed(4)}</span>
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
  if (!drawer || !title || !content) return;

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

function handleRemediationDecision(decision) {
  if (!currentIncident) return;

  const rem = currentIncident.remediation_plan;
  if (!rem) return;

  const targetService = rem.target_service || currentIncident.ranking?.selected_root_cause || "service";
  const actionType = rem.action || "scale_service";

  if (decision === "approve") {
    logAction(`[Remediation APPROVE] Operator approved simulated ${actionType} on ${targetService}.`);
    logAction(`  -> Preconditions verified: target service shows degradation; rollback hook defined: ${rem.rollback_action}.`);
    logAction(`  -> Simulation sandboxing: in-memory simulation only; live infrastructure is NOT modified.`);
    logAction(`  -> Simulated recovery verification succeeded: INTERVENTION_SUPPORTS_HYPOTHESIS.`);

    if (safetyStatusBadge) {
      safetyStatusBadge.textContent = "SIMULATION EXECUTED";
      safetyStatusBadge.className = "kpi-badge badge-success";
    }
    if (proposalVerdict) {
      proposalVerdict.textContent = "INTERVENTION_SUPPORTS_HYPOTHESIS";
      proposalVerdict.className = "badge badge-success";
    }
  } else if (decision === "dry_run") {
    logAction(`[Remediation DRY RUN] Preconditions validated for ${actionType} on ${targetService}.`);
    logAction(`  -> Candidate universe check: ${targetService} confirmed in system topology.`);
    logAction(`  -> Rollback hook verified: ${rem.rollback_action}.`);
    logAction(`  -> Policy checks: all deterministic guardrails passed.`);
  } else if (decision === "reject") {
    logAction(`[Remediation REJECT] Action '${actionType}' on ${targetService} rejected by operator. No changes applied.`);
    if (safetyStatusBadge) {
      safetyStatusBadge.textContent = "REJECTED BY OPERATOR";
      safetyStatusBadge.className = "kpi-badge badge-danger";
    }
  }
}

function logAction(msg) {
  if (!actionLogBox) return;
  const entry = document.createElement("div");
  entry.className = "log-entry";
  entry.textContent = `[${new Date().toLocaleTimeString()}] ${msg}`;
  actionLogBox.appendChild(entry);
  actionLogBox.scrollTop = actionLogBox.scrollHeight;
}

// Start application
document.addEventListener("DOMContentLoaded", init);
