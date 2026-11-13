<#
.SYNOPSIS
    Reports status and diagnostics for Digital Detective unattended queue worker.

.DESCRIPTION
    Inspects Windows Task Scheduler registration, executable availability,
    authentication status (without credential leakage), working-tree cleanliness,
    process lock state, and recent execution history.

.PARAMETER RepoPath
    Path to repository root. Defaults to two levels above this script.

.PARAMETER StateDir
    Path to worker state directory. Defaults to %LOCALAPPDATA%\DigitalDetective\agent_queue.

.PARAMETER TaskName
    Name of scheduled task. Defaults to 'DigitalDetectiveQueueWorker'.

.PARAMETER AsJson
    Outputs status as structured JSON instead of human-readable console tables.
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory = $false)]
    [string]$RepoPath,

    [Parameter(Mandatory = $false)]
    [string]$StateDir,

    [Parameter(Mandatory = $false)]
    [string]$TaskName = "DigitalDetectiveQueueWorker",

    [Parameter(Mandatory = $false)]
    [string]$GhPath,

    [Parameter(Mandatory = $false)]
    [string]$AgyPath,

    [Parameter(Mandatory = $false)]
    [string]$GitPath,

    [Parameter(Mandatory = $false)]
    [switch]$AsJson
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = "Stop"

$coreModulePath = Join-Path -Path $PSScriptRoot -ChildPath "QueueWorkerCore.psm1"
if (-not (Test-Path -Path $coreModulePath)) {
    Write-Error "Required module not found: $coreModulePath"
    exit 1
}
Import-Module -Name $coreModulePath -Force -DisableNameChecking

# 1. Resolve paths
if (-not $RepoPath) {
    $RepoPath = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
}
else {
    $RepoPath = (Resolve-Path -Path $RepoPath).Path
}

if (-not $StateDir) {
    $StateDir = Join-Path $env:LOCALAPPDATA "DigitalDetective\agent_queue"
}

# 2. Inspect Task Scheduler
$task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
$taskInfo = if ($task) { Get-ScheduledTaskInfo -TaskName $TaskName -ErrorAction SilentlyContinue } else { $null }

$schedulerStatus = [PSCustomObject]@{
    Registered     = [bool]$task
    TaskName       = $TaskName
    State          = if ($task) { $task.State.ToString() } else { "NotInstalled" }
    LastRunTime    = if ($taskInfo) { $taskInfo.LastRunTime } else { $null }
    LastTaskResult = if ($taskInfo) { $taskInfo.LastTaskResult } else { $null }
    NextRunTime    = if ($taskInfo) { $taskInfo.NextRunTime } else { $null }
}

# 3. Inspect Executables & Auth (with fallbacks)
$gitExe = Resolve-WorkerExecutable -Name "git" -ExplicitPath $GitPath -FallbackPaths @(
    "C:\Program Files\Git\cmd\git.exe",
    "C:\Program Files (x86)\Git\cmd\git.exe"
)
$gitVer = if ($gitExe) { (& $gitExe --version 2>&1).Trim() } else { "Not Found" }

$ghExe = Resolve-WorkerExecutable -Name "gh" -ExplicitPath $GhPath -FallbackPaths @(
    "$env:LOCALAPPDATA\Programs\GitHub CLI\gh.exe",
    "C:\Program Files\GitHub CLI\gh.exe",
    "$env:ProgramFiles\GitHub CLI\gh.exe",
    "$env:ProgramFiles(x86)\GitHub CLI\gh.exe",
    "$env:USERPROFILE\scoop\shims\gh.exe",
    "$env:ProgramData\chocolatey\bin\gh.exe"
)
$ghVer = if ($ghExe) { (& $ghExe --version 2>&1 | Select-Object -First 1).Trim() } else { "Not Found" }
$ghAuth = if ($ghExe) {
    $authOut = & $ghExe auth status 2>&1
    if ($LASTEXITCODE -eq 0) { "Authenticated" } else { "Not Authenticated (run 'gh auth login')" }
} else { "N/A" }

$agyExe = Resolve-WorkerExecutable -Name "agy" -ExplicitPath $AgyPath -FallbackPaths @(
    "$env:USERPROFILE\.gemini\bin\agy.exe",
    "$env:LOCALAPPDATA\Programs\Antigravity\agy.exe",
    "$env:LOCALAPPDATA\antigravity\bin\agy.cmd"
)
$agyVer = if ($agyExe) { (& $agyExe --version 2>&1).Trim() } else { "Not Found" }

$executablesStatus = [PSCustomObject]@{
    GitPath     = if ($gitExe) { $gitExe } else { "Not Found" }
    GitVersion  = $gitVer
    GhPath      = if ($ghExe) { $ghExe } else { "Not Found" }
    GhVersion   = $ghVer
    GhAuth      = $ghAuth
    AgyPath     = if ($agyExe) { $agyExe } else { "Not Found" }
    AgyVersion  = $agyVer
}

# 4. Inspect Working Tree (using resolved git executable)
$currentBranch = if ($gitExe) {
    try { (& $gitExe -C $RepoPath rev-parse --abbrev-ref HEAD 2>&1).Trim() } catch { "Unknown" }
} else { "Git Not Found" }

$treeCheck = Test-WorkingTreeCleanliness -RepoPath $RepoPath -GitExe $gitExe

$workspaceStatus = [PSCustomObject]@{
    RepositoryRoot = $RepoPath
    CurrentBranch  = $currentBranch
    IsClean        = $treeCheck.IsClean
    CleanReason    = $treeCheck.Reason
    Violations     = $treeCheck.Violations
}

# 5. Inspect Concurrency Lock
$lockFile = Join-Path $StateDir "worker.lock"
$isLockActive = $false
$lockDetails = ""

if (Test-Path -Path $lockFile) {
    try {
        $testStream = [System.IO.File]::Open($lockFile, [System.IO.FileMode]::Open, [System.IO.FileAccess]::ReadWrite, [System.IO.FileShare]::None)
        $testStream.Dispose()
        $isLockActive = $false
        $lockDetails = "Stale (lock file present, but not actively held)"
    }
    catch {
        $isLockActive = $true
        $lockDetails = "Active (held by another process)"
    }
}
else {
    $lockDetails = "None (no lock file)"
}

# 6. Inspect Execution History
$historyFile = Join-Path $StateDir "logs\history.jsonl"
$lastRun = $null
$totalRuns = 0

if (Test-Path -Path $historyFile) {
    $lines = @(Get-Content -Path $historyFile | Where-Object { $_ -match '\S' })
    $totalRuns = $lines.Count
    if ($totalRuns -gt 0) {
        try {
            $lastRun = ConvertFrom-Json -InputObject ($lines[-1])
        } catch {}
    }
}

$historyStatus = [PSCustomObject]@{
    StateDir   = $StateDir
    TotalRuns  = $totalRuns
    LockActive = $isLockActive
    LockStatus = $lockDetails
    LatestRun  = $lastRun
}

$fullStatus = [PSCustomObject]@{
    Timestamp      = [datetime]::UtcNow.ToString("o")
    TaskScheduler  = $schedulerStatus
    Executables    = $executablesStatus
    Workspace      = $workspaceStatus
    WorkerState    = $historyStatus
}

if ($AsJson) {
    $fullStatus | ConvertTo-Json -Depth 5
    return
}

Write-Output "================ Digital Detective Queue Worker Diagnostics ================"
Write-Output "Timestamp (UTC) : $($fullStatus.Timestamp)"
Write-Output ""
Write-Output "--- Task Scheduler ---"
Write-Output "  Task Name     : $($schedulerStatus.TaskName)"
Write-Output "  Registered    : $($schedulerStatus.Registered)"
Write-Output "  State         : $($schedulerStatus.State)"
Write-Output "  Last Run Time : $($schedulerStatus.LastRunTime)"
Write-Output "  Last Result   : $($schedulerStatus.LastTaskResult)"
Write-Output "  Next Run Time : $($schedulerStatus.NextRunTime)"
Write-Output ""
Write-Output "--- Executables & Authentication ---"
Write-Output "  Git           : $($executablesStatus.GitVersion) ($($executablesStatus.GitPath))"
Write-Output "  GitHub CLI    : $($executablesStatus.GhVersion) ($($executablesStatus.GhPath))"
Write-Output "  GH Auth Status: $($executablesStatus.GhAuth)"
Write-Output "  Antigravity   : $($executablesStatus.AgyVersion) ($($executablesStatus.AgyPath))"
Write-Output ""
Write-Output "--- Repository & Working Tree ---"
Write-Output "  Repo Root     : $($workspaceStatus.RepositoryRoot)"
Write-Output "  Current Branch: $($workspaceStatus.CurrentBranch)"
Write-Output "  Tree Clean    : $($workspaceStatus.IsClean) - $($workspaceStatus.CleanReason)"
if ($workspaceStatus.Violations.Count -gt 0) {
    Write-Output "  Violations    :"
    foreach ($v in $workspaceStatus.Violations) { Write-Output "    * $v" }
}
Write-Output ""
Write-Output "--- Worker State & Recent History ---"
Write-Output "  State Dir     : $($historyStatus.StateDir)"
Write-Output "  Process Lock  : $($historyStatus.LockStatus)"
Write-Output "  Total Runs    : $($historyStatus.TotalRuns)"
if ($lastRun) {
    Write-Output "  Latest Run    : $($lastRun.Timestamp) - Issue #$($lastRun.IssueNumber) - Outcome: $($lastRun.Outcome) (Completed: $($lastRun.TaskCompleted), Exit: $($lastRun.ExitCode))"
} else {
    Write-Output "  Latest Run    : None recorded"
}
Write-Output "============================================================================"
