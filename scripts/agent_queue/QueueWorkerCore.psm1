# QueueWorkerCore.psm1
# Core helper module for Digital Detective Windows Unattended Antigravity Queue Worker.
# Enforces queue prioritization, working-tree safety, concurrency locking, and log sanitization.

Set-StrictMode -Version 2.0

$script:ApprovedAliasFiles = @(
    "src/agent/__init__.py",
    "src/agent/disambiguator.py",
    "src/features/__init__.py",
    "src/features/causal_residuals.py",
    "src/features/log_scorer.py"
)

function Get-ApprovedAliasFiles {
    [CmdletBinding()]
    param()
    return $script:ApprovedAliasFiles
}

function Resolve-WorkerExecutable {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]
        [string]$Name,

        [Parameter(Mandatory = $false)]
        [string]$ExplicitPath,

        [Parameter(Mandatory = $false)]
        [string[]]$FallbackPaths = @()
    )

    if ($ExplicitPath -and (Test-Path -Path $ExplicitPath -PathType Leaf)) {
        return (Resolve-Path -Path $ExplicitPath).Path
    }

    $cmd = Get-Command $Name -ErrorAction SilentlyContinue
    if ($cmd -and $cmd.Source -and (Test-Path -Path $cmd.Source -PathType Leaf)) {
        return (Resolve-Path -Path $cmd.Source).Path
    }

    foreach ($candidate in $FallbackPaths) {
        if ($candidate -and (Test-Path -Path $candidate -PathType Leaf)) {
            return (Resolve-Path -Path $candidate).Path
        }
    }

    return $null
}

function Test-WorkingTreeCleanliness {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $false)]
        [string]$RepoPath,

        [Parameter(Mandatory = $false)]
        [string[]]$PorcelainLines = @()
    )

    if (-not $PorcelainLines -and $RepoPath) {
        $gitCmd = Resolve-WorkerExecutable -Name "git"
        if (-not $gitCmd) {
            return [PSCustomObject]@{
                IsClean = $false
                Reason  = "Git executable not found to check working tree cleanliness."
                Violations = @("git-not-found")
            }
        }
        $prevPwd = Get-Location
        try {
            Set-Location $RepoPath
            $rawStatus = & $gitCmd status --porcelain=v1 -uall 2>&1
            $PorcelainLines = @($rawStatus | Where-Object { $_ -match '\S' })
        }
        finally {
            Set-Location $prevPwd
        }
    }

    $violations = [System.Collections.Generic.List[string]]::new()
    $approvedNormalized = @($script:ApprovedAliasFiles | ForEach-Object { $_.Replace('\', '/').Trim().ToLowerInvariant() })

    foreach ($line in $PorcelainLines) {
        if ([string]::IsNullOrWhiteSpace($line)) { continue }
        $trimmed = $line.Trim()
        if ($trimmed.Length -lt 3) { continue }

        $statusCode = $trimmed.Substring(0, 2)
        $filePath = $trimmed.Substring(2).Trim()
        # Handle rename syntax 'old -> new'
        if ($filePath -match '->\s*(.+)$') {
            $filePath = $matches[1].Trim()
        }
        $normalizedPath = $filePath.Replace('\', '/').Trim().ToLowerInvariant()

        if ($statusCode -eq '??') {
            # Untracked file: only the 5 approved alias paths are permitted
            if ($approvedNormalized -notcontains $normalizedPath) {
                $violations.Add("Untracked: $filePath")
            }
        }
        else {
            # Any tracked modification, staging, deletion, or conflict is a violation
            $violations.Add("Status[$statusCode]: $filePath")
        }
    }

    if ($violations.Count -gt 0) {
        return [PSCustomObject]@{
            IsClean     = $false
            Reason      = "Working tree has $($violations.Count) unauthorized modification(s) outside approved aliases."
            Violations  = @($violations)
        }
    }

    return [PSCustomObject]@{
        IsClean     = $true
        Reason      = "Working tree is clean (only approved compatibility aliases present)."
        Violations  = @()
    }
}

function Select-QueueTask {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]
        [AllowEmptyCollection()]
        $Issues,

        [Parameter(Mandatory = $false)]
        [string]$CurrentBranch = "main"
    )

    if (-not $Issues -or $Issues.Count -eq 0) {
        return [PSCustomObject]@{
            Action        = "EmptyQueue"
            SelectedIssue = $null
            TaskType      = "None"
            Reason        = "Queue is empty; no open issues found."
        }
    }

    # 1. Check for any issue in [DD-AGENT:REVIEW]
    $reviewIssues = @($Issues | Where-Object { $_.title -match '^\[DD-AGENT:REVIEW\]' })
    if ($reviewIssues.Count -gt 0) {
        $firstReview = $reviewIssues[0]
        return [PSCustomObject]@{
            Action        = "WaitReview"
            SelectedIssue = $firstReview
            TaskType      = "ReviewPending"
            Reason        = "Issue #$($firstReview.number) is awaiting architect review on PR. No new tasks may start."
        }
    }

    # 2. Check for [DD-AGENT:IN-PROGRESS]
    $inProgressIssues = @($Issues | Where-Object { $_.title -match '^\[DD-AGENT:IN-PROGRESS\]' })
    if ($inProgressIssues.Count -gt 1) {
        return [PSCustomObject]@{
            Action        = "Blocked"
            SelectedIssue = $null
            TaskType      = "MultipleInProgress"
            Reason        = "Multiple issues are currently marked [DD-AGENT:IN-PROGRESS]. Manual intervention required."
        }
    }
    elseif ($inProgressIssues.Count -eq 1) {
        $inProg = $inProgressIssues[0]
        # Check if current branch matches the issue
        $expectedBranchPattern = "issue-$($inProg.number)"
        if ($CurrentBranch -and $CurrentBranch -match $expectedBranchPattern) {
            return [PSCustomObject]@{
                Action        = "Resume"
                SelectedIssue = $inProg
                TaskType      = "InProgress"
                Reason        = "Issue #$($inProg.number) is already IN-PROGRESS and matches checked-out branch '$CurrentBranch'. Resuming task."
            }
        }
        else {
            return [PSCustomObject]@{
                Action        = "Blocked"
                SelectedIssue = $inProg
                TaskType      = "BranchMismatch"
                Reason        = "Issue #$($inProg.number) is IN-PROGRESS, but current branch '$CurrentBranch' does not match. Stopping to avoid cross-branch corruption."
            }
        }
    }

    # 3. Prioritize [DD-AGENT:REVISION]
    $revisionIssues = @($Issues | Where-Object { $_.title -match '^\[DD-AGENT:REVISION\]' } | Sort-Object {
        if ($_.createdAt) { [datetime]$_.createdAt } else { [datetime]::MaxValue }
    })
    if ($revisionIssues.Count -gt 0) {
        $selected = $revisionIssues[0]
        return [PSCustomObject]@{
            Action        = "Execute"
            SelectedIssue = $selected
            TaskType      = "Revision"
            Reason        = "Selected revision task #$($selected.number): '$($selected.title)'."
        }
    }

    # 4. Select oldest [DD-AGENT:READY]
    $readyIssues = @($Issues | Where-Object { $_.title -match '^\[DD-AGENT:READY\]' } | Sort-Object {
        if ($_.createdAt) { [datetime]$_.createdAt } else { [datetime]::MaxValue }
    })
    if ($readyIssues.Count -gt 0) {
        if ($CurrentBranch -and $CurrentBranch -ne "main") {
            return [PSCustomObject]@{
                Action        = "BlockedNotOnMain"
                SelectedIssue = $readyIssues[0]
                TaskType      = "Ready"
                Reason        = "Repository is on branch '$CurrentBranch', but must be on 'main' to start new task #$($readyIssues[0].number)."
            }
        }

        $selected = $readyIssues[0]
        return [PSCustomObject]@{
            Action        = "Execute"
            SelectedIssue = $selected
            TaskType      = "Ready"
            Reason        = "Selected oldest ready task #$($selected.number): '$($selected.title)'."
        }
    }

    return [PSCustomObject]@{
        Action        = "EmptyQueue"
        SelectedIssue = $null
        TaskType      = "None"
        Reason        = "No eligible [DD-AGENT:REVISION] or [DD-AGENT:READY] tasks found."
    }
}

function Test-SoftDenial {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $false)]
        [string]$OutputText,

        [Parameter(Mandatory = $false)]
        [int]$ExitCode = 0
    )

    if ([string]::IsNullOrWhiteSpace($OutputText)) {
        return ($ExitCode -ne 0)
    }

    $softDenialPatterns = @(
        'tool approval required',
        'requires approval',
        'permission denied',
        'soft-denied',
        'approval-gated',
        'was denied by user or policy',
        'not authorized to execute',
        'operation requires an unavailable approval'
    )

    foreach ($pat in $softDenialPatterns) {
        if ($OutputText -like "*$pat*") {
            return $true
        }
    }

    return $false
}

function Sanitize-LogContent {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $false)]
        [string]$Text
    )

    if ([string]::IsNullOrWhiteSpace($Text)) {
        return ""
    }

    $sanitized = $Text
    # GitHub PAT tokens
    $sanitized = [regex]::Replace($sanitized, 'ghp_[A-Za-z0-9_]{20,}', '[REDACTED_GH_PAT]')
    $sanitized = [regex]::Replace($sanitized, 'github_pat_[A-Za-z0-9_]{20,}', '[REDACTED_GH_FINE_GRAINED_PAT]')
    $sanitized = [regex]::Replace($sanitized, 'Bearer\s+[A-Za-z0-9\-_\.=]{20,}', 'Bearer [REDACTED_TOKEN]')
    $sanitized = [regex]::Replace($sanitized, '(?i)(password|secret|token|api_key)\s*[:=]\s*(?!\[REDACTED)[^\s,;]+', '$1: [REDACTED]')

    return $sanitized
}

function Acquire-WorkerLock {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]
        [string]$LockFilePath
    )

    $lockDir = Split-Path -Path $LockFilePath -Parent
    if (-not (Test-Path -Path $lockDir)) {
        New-Item -Path $lockDir -ItemType Directory -Force | Out-Null
    }

    try {
        $fileStream = [System.IO.File]::Open(
            $LockFilePath,
            [System.IO.FileMode]::OpenOrCreate,
            [System.IO.FileAccess]::ReadWrite,
            [System.IO.FileShare]::None
        )

        $info = "PID=$PID`nHost=$env:COMPUTERNAME`nUser=$env:USERNAME`nStartTime=$([datetime]::UtcNow.ToString('o'))`n"
        $bytes = [System.Text.Encoding]::UTF8.GetBytes($info)
        $fileStream.SetLength(0)
        $fileStream.Write($bytes, 0, $bytes.Length)
        $fileStream.Flush()

        return [PSCustomObject]@{
            Success    = $true
            Stream     = $fileStream
            LockPath   = $LockFilePath
            ErrorMessage = $null
        }
    }
    catch [System.IO.IOException] {
        return [PSCustomObject]@{
            Success    = $false
            Stream     = $null
            LockPath   = $LockFilePath
            ErrorMessage = "Concurrent lock already held by another process: $($_.Exception.Message)"
        }
    }
    catch {
        return [PSCustomObject]@{
            Success    = $false
            Stream     = $null
            LockPath   = $LockFilePath
            ErrorMessage = "Failed to acquire lock: $($_.Exception.Message)"
        }
    }
}

function Release-WorkerLock {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $false)]
        $LockResult
    )

    if ($LockResult -and $LockResult.Stream) {
        try {
            $LockResult.Stream.Dispose()
        }
        catch {}
    }
}

function Rotate-WorkerLogs {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]
        [string]$LogsDir,

        [Parameter(Mandatory = $false)]
        [int]$MaxLogsToKeep = 50
    )

    if (-not (Test-Path -Path $LogsDir)) { return }

    $logFiles = Get-ChildItem -Path $LogsDir -Filter "*.log" -File | Sort-Object LastWriteTime -Descending
    if ($logFiles.Count -gt $MaxLogsToKeep) {
        $toRemove = $logFiles | Select-Object -Skip $MaxLogsToKeep
        foreach ($f in $toRemove) {
            try { Remove-Item -Path $f.FullName -Force -ErrorAction SilentlyContinue } catch {}
        }
    }
}

Export-ModuleMember -Function `
    Get-ApprovedAliasFiles, `
    Resolve-WorkerExecutable, `
    Test-WorkingTreeCleanliness, `
    Select-QueueTask, `
    Test-SoftDenial, `
    Sanitize-LogContent, `
    Acquire-WorkerLock, `
    Release-WorkerLock, `
    Rotate-WorkerLogs
