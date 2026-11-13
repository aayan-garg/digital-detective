<#
.SYNOPSIS
    Unattended background worker for Digital Detective GitHub task queue.

.DESCRIPTION
    Polls the Digital Detective GitHub task queue using the authenticated GitHub CLI,
    verifies working-tree safety against the 5 approved compatibility aliases,
    enforces single-task concurrency locking, and invokes the Antigravity CLI in
    headless mode only when eligible tasks exist.

.PARAMETER RepoPath
    Path to the digital-detective repository root. Defaults to two levels above this script.

.PARAMETER StateDir
    Path to store local logs and process lock. Defaults to %LOCALAPPDATA%\DigitalDetective\agent_queue.

.PARAMETER GhPath
    Explicit path to gh.exe (auto-discovered if omitted).

.PARAMETER AgyPath
    Explicit path to agy.exe (auto-discovered if omitted).

.PARAMETER GitPath
    Explicit path to git.exe (auto-discovered if omitted).

.PARAMETER RepoSlug
    GitHub repository in 'owner/repo' format. Defaults to 'aayan-garg/digital-detective'.

.PARAMETER TimeoutSeconds
    Maximum timeout for headless Antigravity turn. Defaults to 1800 seconds (30m).

.PARAMETER DryRun
    Simulates queue check and displays actions without launching Antigravity.
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory = $false)]
    [string]$RepoPath,

    [Parameter(Mandatory = $false)]
    [string]$StateDir,

    [Parameter(Mandatory = $false)]
    [string]$GhPath,

    [Parameter(Mandatory = $false)]
    [string]$AgyPath,

    [Parameter(Mandatory = $false)]
    [string]$GitPath,

    [Parameter(Mandatory = $false)]
    [string]$RepoSlug = "aayan-garg/digital-detective",

    [Parameter(Mandatory = $false)]
    [int]$TimeoutSeconds = 1800,

    [Parameter(Mandatory = $false)]
    [switch]$DryRun
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = "Stop"

# Import companion module
$coreModulePath = Join-Path -Path $PSScriptRoot -ChildPath "QueueWorkerCore.psm1"
if (-not (Test-Path -Path $coreModulePath)) {
    Write-Error "Required module not found: $coreModulePath"
    exit 1
}
Import-Module -Name $coreModulePath -Force -DisableNameChecking

# 1. Resolve repository root
if (-not $RepoPath) {
    $RepoPath = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
}
else {
    if (-not (Test-Path -Path $RepoPath -PathType Container)) {
        Write-Error "Specified RepoPath does not exist: $RepoPath"
        exit 1
    }
    $RepoPath = (Resolve-Path -Path $RepoPath).Path
}

$agentsMd = Join-Path $RepoPath "AGENTS.md"
$gitDir = Join-Path $RepoPath ".git"
if (-not (Test-Path -Path $gitDir) -or -not (Test-Path -Path $agentsMd)) {
    Write-Error "Path '$RepoPath' is not a valid digital-detective repository (missing .git or AGENTS.md)."
    exit 1
}

# 2. Resolve State Directory
if (-not $StateDir) {
    $StateDir = Join-Path $env:LOCALAPPDATA "DigitalDetective\agent_queue"
}
$logsDir = Join-Path $StateDir "logs"
if (-not (Test-Path -Path $logsDir)) {
    New-Item -Path $logsDir -ItemType Directory -Force | Out-Null
}

# 3. Locate required executables with fallbacks
$resolvedGit = Resolve-WorkerExecutable -Name "git" -ExplicitPath $GitPath -FallbackPaths @(
    "C:\Program Files\Git\cmd\git.exe",
    "C:\Program Files (x86)\Git\cmd\git.exe"
)

$resolvedGh = Resolve-WorkerExecutable -Name "gh" -ExplicitPath $GhPath -FallbackPaths @(
    "$env:LOCALAPPDATA\Programs\GitHub CLI\gh.exe",
    "C:\Program Files\GitHub CLI\gh.exe",
    "$env:ProgramFiles\GitHub CLI\gh.exe",
    "$env:ProgramFiles(x86)\GitHub CLI\gh.exe",
    "$env:USERPROFILE\scoop\shims\gh.exe",
    "$env:ProgramData\chocolatey\bin\gh.exe"
)

$resolvedAgy = Resolve-WorkerExecutable -Name "agy" -ExplicitPath $AgyPath -FallbackPaths @(
    "$env:USERPROFILE\.gemini\bin\agy.exe",
    "$env:LOCALAPPDATA\Programs\Antigravity\agy.exe",
    "$env:LOCALAPPDATA\antigravity\bin\agy.cmd"
)

$missing = @()
if (-not $resolvedGit) { $missing += "git" }
if (-not $resolvedGh) { $missing += "gh (GitHub CLI - install via 'winget install --id GitHub.cli')" }
if (-not $resolvedAgy) { $missing += "agy (Antigravity CLI - ensure ~/.gemini/bin/agy.exe exists)" }

if ($missing.Count -gt 0) {
    $msg = "Missing required executable(s): $($missing -join ', ')"
    Write-Warning $msg
    exit 2
}

# 4. Verify Authentication without exposing tokens
Write-Verbose "Checking GitHub CLI authentication..."
$ghAuthOutput = & $resolvedGh auth status 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Warning "GitHub CLI is not authenticated. Run 'gh auth login' before starting worker."
    exit 3
}

Write-Verbose "Checking Antigravity CLI version..."
$agyVersion = & $resolvedAgy --version 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Warning "Antigravity CLI failed check: $agyVersion"
    exit 3
}

# 5. Concurrency lock
$lockFile = Join-Path $StateDir "worker.lock"
$lockResult = $null

if (-not $DryRun) {
    $lockResult = Acquire-WorkerLock -LockFilePath $lockFile
    if (-not $lockResult.Success) {
        Write-Output "Queue worker already active or lock held: $($lockResult.ErrorMessage). Exiting."
        exit 0
    }
}

try {
    # 6. Working-tree safety check (using resolved git executable)
    $treeCleanliness = Test-WorkingTreeCleanliness -RepoPath $RepoPath -GitExe $resolvedGit
    if (-not $treeCleanliness.IsClean) {
        Write-Warning "Working tree safety check FAILED: $($treeCleanliness.Reason)"
        foreach ($v in $treeCleanliness.Violations) {
            Write-Warning "  - $v"
        }
        exit 4
    }

    # 7. Check current branch using resolved git executable
    $currentBranch = (& $resolvedGit -C $RepoPath rev-parse --abbrev-ref HEAD 2>&1).Trim()
    Write-Verbose "Current branch: $currentBranch"

    # 8. Query GitHub issue queue
    Write-Verbose "Querying open issues for $RepoSlug..."
    $rawIssuesJson = & $resolvedGh issue list --repo $RepoSlug --state open --json number,title,createdAt,updatedAt,comments --limit 50 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to query GitHub issues: $rawIssuesJson"
        exit 6
    }

    $issues = @()
    if ($rawIssuesJson -and $rawIssuesJson.Trim().Length -gt 0) {
        try {
            $issues = ConvertFrom-Json -InputObject $rawIssuesJson
        }
        catch {
            Write-Error "Failed to parse GitHub issues JSON: $($_.Exception.Message)"
            exit 6
        }
    }

    # 9. Evaluate queue decision logic
    $decision = Select-QueueTask -Issues $issues -CurrentBranch $currentBranch
    Write-Output "Queue decision: $($decision.Action) - $($decision.Reason)"

    if ($decision.Action -eq "EmptyQueue" -or $decision.Action -eq "WaitReview") {
        Write-Output "No action required. Exiting cleanly without invoking Antigravity."
        exit 0
    }

    if ($decision.Action -eq "Blocked" -or $decision.Action -eq "BlockedNotOnMain") {
        Write-Warning "Task blocked: $($decision.Reason)"
        exit 5
    }

    $selectedIssue = $decision.SelectedIssue

    # 10. Handle DryRun
    if ($DryRun) {
        Write-Output "-------------------- [DRY-RUN SUMMARY] --------------------"
        Write-Output "Repository Root : $RepoPath"
        Write-Output "Git Executable  : $resolvedGit"
        Write-Output "Current Branch  : $currentBranch"
        Write-Output "Selected Issue  : #$($selectedIssue.number) - $($selectedIssue.title)"
        Write-Output "Task Action     : $($decision.Action) ($($decision.TaskType))"
        Write-Output "Command to Run  : `"$resolvedAgy`" -p `"/process-digital-detective-queue`" --output-format json --print-timeout `"$($TimeoutSeconds)s`""
        Write-Output "State Directory : $StateDir"
        Write-Output "-----------------------------------------------------------"
        exit 0
    }

    # 11. Launch Antigravity in headless mode safely via Invoke-AntigravityCli
    $timestamp = (Get-Date).ToUniversalTime().ToString("yyyyMMdd-HHmmss")
    $logFile = Join-Path $logsDir "run-$timestamp-issue-$($selectedIssue.number).log"
    $historyFile = Join-Path $logsDir "history.jsonl"

    Write-Output "Launching Antigravity headless worker for Issue #$($selectedIssue.number)..."
    Write-Output "Log destination: $logFile"

    $startTime = Get-Date
    $prompt = "/process-digital-detective-queue"

    # Execute agy safely from repository root without interpolating arbitrary issue text
    $cliResult = Invoke-AntigravityCli `
        -AgyPath $resolvedAgy `
        -Prompt $prompt `
        -WorkingDirectory $RepoPath `
        -TimeoutSeconds $TimeoutSeconds

    $duration = ((Get-Date) - $startTime).TotalSeconds
    $combinedOutput = "=== STDOUT ===`n$($cliResult.StandardOutput)`n=== STDERR ===`n$($cliResult.StandardError)"
    $sanitizedOutput = Sanitize-LogContent -Text $combinedOutput

    # Write log file
    Set-Content -Path $logFile -Value $sanitizedOutput -Encoding UTF8

    # Check for soft-denials
    $isSoftDenied = Test-SoftDenial -OutputText $sanitizedOutput -ExitCode $cliResult.ExitCode

    # Query issue post-state from GitHub
    $postIssueJson = & $resolvedGh issue view $selectedIssue.number --repo $RepoSlug --json state,title 2>&1
    $postIssueObj = $null
    if ($LASTEXITCODE -eq 0 -and $postIssueJson) {
        try {
            $postIssueObj = ConvertFrom-Json -InputObject $postIssueJson
        } catch {}
    }

    # Evaluate post-run state distinguishing CLI process exit from actual task completion
    $evalState = Evaluate-PostRunState `
        -InitialIssue $selectedIssue `
        -PostIssue $postIssueObj `
        -ExitCode $cliResult.ExitCode `
        -IsSoftDenied $isSoftDenied `
        -OutputText $sanitizedOutput

    # Record history entry with full evaluation
    $historyEntry = [PSCustomObject]@{
        Timestamp       = $timestamp
        IssueNumber     = $selectedIssue.number
        InitialTitle    = $selectedIssue.title
        PostTitle       = if ($postIssueObj) { $postIssueObj.title } else { "Unknown" }
        ExitCode        = $cliResult.ExitCode
        DurationSeconds = [math]::Round($duration, 2)
        Outcome         = $evalState.Outcome
        TaskCompleted   = $evalState.TaskCompleted
        EvaluationReason= $evalState.Reason
        IsSoftDenied    = $isSoftDenied
        LaunchMode      = $cliResult.LaunchMode
        LogFile         = $logFile
    }

    $jsonLine = ($historyEntry | ConvertTo-Json -Compress)
    Add-Content -Path $historyFile -Value $jsonLine -Encoding UTF8

    # Rotate old logs and bound history.jsonl
    Rotate-WorkerLogs -LogsDir $logsDir -MaxLogsToKeep 50 -MaxHistoryLines 500

    Write-Output "Execution finished: $($evalState.Outcome) (Process exit: $($cliResult.ExitCode), Task completed: $($evalState.TaskCompleted), Duration: $([math]::Round($duration, 1))s)"
    Write-Output "Reason: $($evalState.Reason)"

    if ($isSoftDenied) {
        Write-Warning "Antigravity reported an approval soft-denial. User action or interactive approval may be required."
    }

    exit $cliResult.ExitCode
}
finally {
    if ($lockResult) {
        Release-WorkerLock -LockResult $lockResult
    }
}
