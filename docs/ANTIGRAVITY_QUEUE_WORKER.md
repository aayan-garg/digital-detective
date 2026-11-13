# Windows Unattended Antigravity Queue Runner

## 1. Overview & Architecture

The **Digital Detective Queue Worker** is an opt-in, user-scoped Windows background runner. It automatically checks the GitHub task queue in `aayan-garg/digital-detective` and launches the Antigravity CLI in headless mode (`agy -p`) only when eligible work exists.

This automation replaces the manual workflow of opening the Antigravity IDE and typing `/process-digital-detective-queue` for every task, while strictly failing closed and preserving user control of the workstation.

### Architecture Flow

```
[Windows Task Scheduler / Logon Trigger]
                   │
                   ▼
┌────────────────────────────────────────────────────────┐
│ scripts/agent_queue/Invoke-DigitalDetectiveQueueWorker │
└──────────────────────────┬─────────────────────────────┘
                           │
    1. Resolve executables (git, gh, agy)
    2. Check gh auth status & agy --version
    3. Verify process lock (%LOCALAPPDATA%\DigitalDetective\agent_queue\worker.lock)
    4. Validate working tree (clean except 5 approved untracked aliases)
    5. Query GitHub Issues via authenticated gh CLI
                           │
       ┌───────────────────┴───────────────────┐
       ▼                                       ▼
 [No Eligible Work /                     [Eligible Task]
  Review Pending /                        - Prioritize [DD-AGENT:REVISION]
  Queue Empty]                            - Else oldest [DD-AGENT:READY] (on main)
       │                                       │
       ▼                                       ▼
  Exit cleanly (0 tokens)                 Launch Antigravity CLI in headless mode:
                                          agy -p "/process-digital-detective-queue"
                                              --output-format json
                                              --print-timeout 1800s
                                               │
                                               ▼
                                          - Capture stdout/stderr to local log
                                          - Redact secrets / tokens (Sanitize-LogContent)
                                          - Detect soft-denials / approval gates
                                          - Verify issue post-state
                                          - Record structured summary in history.jsonl
                                          - Release lock & rotate logs
```

---

## 2. One-Time Prerequisites & Setup

### 2.1 Software Prerequisites

| Tool | Required Version | Verification Command | Installation / Setup |
| :--- | :--- | :--- | :--- |
| **Windows OS** | Windows 10/11 | `[System.Environment]::OSVersion` | Windows 11 Build 26100+ tested |
| **PowerShell** | 5.1+ or 7+ | `$PSVersionTable.PSVersion` | Built into Windows (Desktop Edition 5.1+) |
| **Git for Windows** | 2.40+ | `git --version` | Standard install (`C:\Program Files\Git\cmd\git.exe`) |
| **GitHub CLI (`gh`)**| 2.30+ | `gh --version` | `winget install --id GitHub.cli` or [cli.github.com](https://cli.github.com) |
| **Antigravity CLI** | 1.5.0+ | `agy --version` | Auto-installed at `%USERPROFILE%\.gemini\bin\agy.exe` |

### 2.2 GitHub CLI Authentication

The worker uses the authenticated `gh` CLI for lightweight queue polling. Run this once interactively:

```powershell
gh auth login
```

Verify authentication without printing credentials:

```powershell
gh auth status
```

### 2.3 Antigravity CLI Authentication & MCP Configuration

1. **Antigravity CLI:** Authenticate interactively once:
   ```powershell
   & "$env:USERPROFILE\.gemini\bin\agy.exe" models
   ```
   If cached credentials are active, available models are listed immediately.

2. **Global MCP Configuration:**
   Verify `%USERPROFILE%\.gemini\config\mcp_config.json` contains the `github` MCP server configuration with read/write access to `aayan-garg/digital-detective`. This configuration is shared between Antigravity CLI and IDE.

---

## 3. Operations & Commands

All utility scripts reside under `scripts/agent_queue/`.

### 3.1 Status & Diagnostics

Inspect Task Scheduler registration, executable paths, authentication health, repository cleanliness, and last run outcome:

```powershell
# Human-readable table
powershell -ExecutionPolicy Bypass -File scripts/agent_queue/Get-DigitalDetectiveQueueWorkerStatus.ps1

# Structured JSON (for automated monitoring or scripts)
powershell -ExecutionPolicy Bypass -File scripts/agent_queue/Get-DigitalDetectiveQueueWorkerStatus.ps1 -AsJson
```

### 3.2 Dry-Run Mode

Evaluate what the worker would do right now without starting Antigravity or acquiring a permanent lock:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/agent_queue/Invoke-DigitalDetectiveQueueWorker.ps1 -DryRun
```

### 3.3 Manual One-Shot Execution

Run a single pass of the queue worker directly in the foreground:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/agent_queue/Invoke-DigitalDetectiveQueueWorker.ps1
```

### 3.4 Scheduled Task Installation

Install the user-scoped Windows Task Scheduler task:

```powershell
# Dry-run inspect planned task configuration
powershell -ExecutionPolicy Bypass -File scripts/agent_queue/Install-DigitalDetectiveQueueWorker.ps1 -DryRun

# Install with default 30-minute interval
powershell -ExecutionPolicy Bypass -File scripts/agent_queue/Install-DigitalDetectiveQueueWorker.ps1

# Install with custom polling interval (e.g., every 15 minutes)
powershell -ExecutionPolicy Bypass -File scripts/agent_queue/Install-DigitalDetectiveQueueWorker.ps1 -IntervalMinutes 15 -Force
```

### 3.5 Scheduled Task Uninstallation

Remove the scheduled task and optionally clean local state:

```powershell
# Dry-run inspect uninstallation
powershell -ExecutionPolicy Bypass -File scripts/agent_queue/Uninstall-DigitalDetectiveQueueWorker.ps1 -DryRun

# Unregister scheduled task (preserves logs & history)
powershell -ExecutionPolicy Bypass -File scripts/agent_queue/Uninstall-DigitalDetectiveQueueWorker.ps1

# Unregister and purge local worker logs/state
powershell -ExecutionPolicy Bypass -File scripts/agent_queue/Uninstall-DigitalDetectiveQueueWorker.ps1 -RemoveState
```

---

## 4. Workstation & Scheduling Boundaries

### 4.1 On vs Off Behavior

* **Workstation Awake & User Logged In:**
  The task runs in user-interactive scope (`LogonType Interactive`), requiring no administrator privileges and storing no user passwords. Scheduled runs execute invisibly in the background.
* **Workstation Powered Down / Suspended:**
  Windows Task Scheduler cannot execute when the machine is shut down or in deep sleep.
* **Wake & Logon Catch-Up:**
  The task is configured with `-StartWhenAvailable`. If a scheduled poll was missed while the computer was suspended or offline, Task Scheduler starts the worker as soon as the computer wakes and the user is logged in.

---

## 5. Security Guardrails & Invariants

1. **Strict Permission Discipline:**
   * Never uses `--dangerously-skip-permissions`.
   * Never configures wildcards `command(*)`, `mcp(*)`, `read_file(*)`, or `write_file(*)`.
   * If a tool operation requires unavailable approval in headless mode, Antigravity soft-denies it. The worker recognizes this (`SOFT_DENIED`) and does not report false completion.
2. **Fixed Command Invocation:**
   * Prompt argument is strictly static: `/process-digital-detective-queue`.
   * Untrusted issue titles or bodies are never interpolated into shell command lines, preventing command injection.
3. **Working-Tree Safety:**
   * Exactly five pre-existing untracked compatibility alias files are recognized as an approved baseline:
     * `src/agent/__init__.py`
     * `src/agent/disambiguator.py`
     * `src/features/__init__.py`
     * `src/features/causal_residuals.py`
     * `src/features/log_scorer.py`
   * Any other tracked modification, staged change, or unexpected untracked file causes the worker to immediately fail closed (`exit 4`).
4. **Git Branch & Concurrency Invariants:**
   * Starting a new `[DD-AGENT:READY]` task requires the repository to be on `main`.
   * Revisions resume their dedicated PR branch.
   * Concurrency file lock (`worker.lock`) prevents multiple scheduled tasks or background processes from running simultaneously.
5. **Credential Protection & Log Sanitization:**
   * Authentication checks never log tokens or authorization headers.
   * Local logs scrub GitHub PATs (`ghp_...`, `github_pat_...`), Bearer tokens, and secrets.
   * Local logs reside in `%LOCALAPPDATA%\DigitalDetective\agent_queue\logs`, keeping the git working tree clean.

---

## 6. Exit Codes & Troubleshooting

| Exit Code | Condition | Remediation |
| :---: | :--- | :--- |
| **0** | Clean exit (success, empty queue, review pending, or dry-run) | Normal operation. |
| **1** | Fatal script initialization error (missing core module or repository root) | Ensure script is run inside `digital-detective`. |
| **2** | Missing required executable (`git`, `gh`, or `agy`) | Install missing executable (`winget install --id GitHub.cli` or verify `~/.gemini/bin/agy.exe`). |
| **3** | Authentication failure (`gh auth status` or `agy --version`) | Run `gh auth login` or verify interactive Antigravity login. |
| **4** | Working-tree dirty outside the 5 approved alias files | Inspect `git status`. Commit, stash, or clean unapproved modifications. |
| **5** | Blocked state (repository not on `main` for new ready task, or branch mismatch) | Checkout `main` or the correct PR branch. |
| **6** | GitHub CLI query or JSON parsing error | Check network connectivity to `api.github.com`. |
| **124** | Antigravity execution timed out (exceeded `-TimeoutSeconds`) | Inspect log in `%LOCALAPPDATA%\DigitalDetective\agent_queue\logs`. |

---

## 7. Automated Test Suite

A dedicated Pester test suite validates all decision logic without network calls or model invocations:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/agent_queue/Invoke-AgentQueueTests.ps1
```

**Test Coverage (24/24 passing):**
* Approved untracked alias whitelist vs unauthorized modifications and dirty working trees.
* Empty queue, review pending blocks, and multiple in-progress blocks.
* Revision priority over ready tasks and oldest-first selection.
* Process file lock acquisition, contention detection, and release.
* Shell command injection resistance on untrusted issue titles.
* Soft-denial pattern detection and credential sanitization in logs.
* Scheduled task installer dry-run, idempotency, status inspector JSON output, and uninstaller.
