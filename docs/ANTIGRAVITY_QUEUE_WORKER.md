# Windows Unattended Antigravity Queue Runner

## 1. Overview & Architecture

The **Digital Detective Queue Worker** is an opt-in, user-scoped Windows background runner. It automatically checks the GitHub task queue in `aayan-garg/digital-detective` and launches the Antigravity CLI in headless mode (`agy -p`) only when eligible work exists.

This automation eliminates the manual workflow of opening the Antigravity IDE and typing `/process-digital-detective-queue` for every task, while strictly failing closed and preserving user control of the workstation.

### Architecture Flow

```
[Windows Task Scheduler / Logon Trigger]
                   │
                   ▼
┌────────────────────────────────────────────────────────┐
│ scripts/agent_queue/Invoke-DigitalDetectiveQueueWorker │
└──────────────────────────┬─────────────────────────────┘
                           │
    1. Resolve executables (git, gh, agy) with fallback paths
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
       │                                  - Exact branch pattern: (^|[/-])issue-$num($|-)
       ▼                                       │
  Exit cleanly (0 tokens)                      ▼
                                          Launch Antigravity CLI in headless mode:
                                          agy -p "/process-digital-detective-queue"
                                              --output-format json
                                              --print-timeout 1800s
                                          (routed via cmd.exe wrapper if .cmd/.bat)
                                               │
                                               ▼
                                          - Capture stdout/stderr to local log
                                          - Redact secrets / tokens (Sanitize-LogContent)
                                          - Detect soft-denials / approval gates
                                          - Verify issue post-state (transitions to REVIEW)
                                          - Record structured summary in history.jsonl
                                          - Release lock, rotate logs & bound history.jsonl
```

---

## 2. Tested Host Environment

The worker, test suite, and Task Scheduler integration were validated on:

| Component | Verified Version | Path / Source |
| :--- | :--- | :--- |
| **Operating System** | Windows 11 Pro 64-bit (Build 10.0.26100.9818) | Microsoft Windows NT 10.0.26100.0 |
| **PowerShell** | Desktop Edition 5.1.26100.9818 | `C:\WINDOWS\System32\WindowsPowerShell\v1.0\powershell.exe` |
| **Git for Windows** | 2.55.0.windows.3 | `C:\Program Files\Git\cmd\git.exe` |
| **Antigravity CLI** | 1.5.8 | `C:\Users\Aayan\.gemini\bin\agy.exe` |
| **GitHub CLI** | 2.30+ required (fails closed when missing) | `winget install --id GitHub.cli` |
| **Pester** | 3.4.0 (Windows built-in) | `C:\Program Files\WindowsPowerShell\Modules\Pester\3.4.0` |
| **Audited Base SHA** | `cbfc2c0794bcaa5951097da32cd43d9a7d4f934f` | `origin/main` |

---

## 3. One-Time Prerequisites & Setup

### 3.1 Software Prerequisites

1. **GitHub CLI (`gh`):**
   Install if not present:
   ```powershell
   winget install --id GitHub.cli
   ```
   Authenticate interactively once:
   ```powershell
   gh auth login
   ```
   Verify authentication without printing credentials:
   ```powershell
   gh auth status
   ```

2. **Antigravity CLI (`agy`):**
   Ensure `%USERPROFILE%\.gemini\bin\agy.exe` is present and authenticate interactively once:
   ```powershell
   & "$env:USERPROFILE\.gemini\bin\agy.exe" models
   ```

3. **Global MCP Configuration:**
   Verify `%USERPROFILE%\.gemini\config\mcp_config.json` contains the `github` MCP server configuration with read/write access to `aayan-garg/digital-detective`.

---

## 4. Operations & Commands

All utility scripts reside under `scripts/agent_queue/`.

### 4.1 Status & Diagnostics

Inspect Task Scheduler registration, executable paths, authentication health, repository cleanliness, and last run outcome:

```powershell
# Human-readable table
powershell -ExecutionPolicy Bypass -File scripts/agent_queue/Get-DigitalDetectiveQueueWorkerStatus.ps1

# Structured JSON (for automated monitoring or scripts)
powershell -ExecutionPolicy Bypass -File scripts/agent_queue/Get-DigitalDetectiveQueueWorkerStatus.ps1 -AsJson
```

### 4.2 Dry-Run Mode

Evaluate what the worker would do right now without starting Antigravity or acquiring a permanent lock:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/agent_queue/Invoke-DigitalDetectiveQueueWorker.ps1 -DryRun
```

### 4.3 Manual One-Shot Execution

Run a single pass of the queue worker directly in the foreground:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/agent_queue/Invoke-DigitalDetectiveQueueWorker.ps1
```

### 4.4 Scheduled Task Installation

Install the user-scoped Windows Task Scheduler task:

```powershell
# Dry-run inspect planned task configuration
powershell -ExecutionPolicy Bypass -File scripts/agent_queue/Install-DigitalDetectiveQueueWorker.ps1 -DryRun

# Install with default 30-minute interval
powershell -ExecutionPolicy Bypass -File scripts/agent_queue/Install-DigitalDetectiveQueueWorker.ps1

# Install with custom polling interval (e.g., every 15 minutes)
powershell -ExecutionPolicy Bypass -File scripts/agent_queue/Install-DigitalDetectiveQueueWorker.ps1 -IntervalMinutes 15 -Force
```

### 4.5 Scheduled Task Uninstallation

Remove the scheduled task and optionally clean local state:

```powershell
# Dry-run inspect uninstallation
powershell -ExecutionPolicy Bypass -File scripts/agent_queue/Uninstall-DigitalDetectiveQueueWorker.ps1 -DryRun

# Unregister scheduled task (preserves logs & history)
powershell -ExecutionPolicy Bypass -File scripts/agent_queue/Uninstall-DigitalDetectiveQueueWorker.ps1

# Unregister and purge local worker logs/state (strictly constrained to %LOCALAPPDATA%\DigitalDetective\agent_queue)
powershell -ExecutionPolicy Bypass -File scripts/agent_queue/Uninstall-DigitalDetectiveQueueWorker.ps1 -RemoveState
```

---

## 5. Workstation & Scheduling Boundaries

* **Workstation Awake & User Logged In:**
  The task runs in user-interactive scope (`LogonType Interactive`), requiring no administrator privileges and storing no user passwords. Scheduled runs execute invisibly in the background.
* **Workstation Powered Down / Suspended:**
  Windows Task Scheduler cannot execute when the machine is shut down or in deep sleep.
* **Wake & Logon Catch-Up:**
  The task is configured with `-StartWhenAvailable`. If a scheduled poll was missed while the computer was suspended or offline, Task Scheduler starts the worker as soon as the computer wakes and the user is logged in.

---

## 6. Security Guardrails & Invariants

1. **Strict Permission Discipline:**
   * Never uses `--dangerously-skip-permissions`.
   * Never configures wildcards `command(*)`, `mcp(*)`, `read_file(*)`, or `write_file(*)`.
   * If a tool operation requires unavailable approval in headless mode, Antigravity soft-denies it. The worker detects this (`SOFT_DENIED`) and does not report false completion.
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
   * In-progress branch matching uses exact segment matching `(^|[/-])issue-$number($|-)` to prevent issue #4 from matching issue #40.
   * Concurrency file lock (`worker.lock`) prevents multiple scheduled tasks or background processes from running simultaneously.
5. **Credential Protection & Log Sanitization:**
   * Authentication checks never log tokens or authorization headers.
   * Local logs scrub GitHub PATs (`ghp_...`, `github_pat_...`), Bearer tokens, and secrets.
   * Local logs reside in `%LOCALAPPDATA%\DigitalDetective\agent_queue\logs`, keeping the git working tree clean.
   * Log files are bounded (keeps latest 50 files) and `history.jsonl` is bounded to the latest 500 entries.
6. **Destructive Cleanup Boundaries:**
   * `Uninstall-DigitalDetectiveQueueWorker.ps1 -RemoveState` strictly constrains deletions to `%LOCALAPPDATA%\DigitalDetective\agent_queue`.
   * External directories or paths containing reparse points (symlinks / directory junctions) are rejected with a safety error.
7. **Post-Run State Verification:**
   * The worker verifies whether the issue actually transitioned to `[DD-AGENT:REVIEW]` or `[DD-AGENT:DONE]`.
   * If the CLI process exits with code 0 but the issue remains untransitioned, history records `CLI_SUCCEEDED_UNTRANSITIONED` rather than false task success.

---

## 7. Exit Codes & Troubleshooting

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

## 8. Automated Test Suite

A dedicated Pester test suite validates all decision logic without live network calls or model invocations:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/agent_queue/Invoke-AgentQueueTests.ps1
```

**Test Coverage (32/32 passing in ~35s):**
* Approved untracked alias whitelist vs unauthorized modifications, staged files, and deletions.
* Empty queue, review pending blocks, and multiple in-progress blocks.
* Exact branch segment matching preventing issue #4 from matching issue #40.
* Revision priority over ready tasks and oldest-first selection.
* Process file lock acquisition, contention detection, and release.
* Post-run state verification distinguishing CLI process success from issue transition.
* Representative captured Antigravity CLI stderr fixtures for soft-denials.
* Credential sanitization in logs.
* Safe `.cmd` batch execution via `cmd.exe /d /c` wrapper.
* Scheduled task installer dry-run, idempotency, status inspector JSON output, and uninstaller.
* Rejection of destructive state cleanup on external directories (Safety Gate Test).
* Live Task Scheduler registration, idempotency, and clean unregistration.
