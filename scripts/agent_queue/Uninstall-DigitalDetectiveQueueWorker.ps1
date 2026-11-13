<#
.SYNOPSIS
    Uninstalls the Digital Detective Windows Scheduled Task and cleans local worker state.

.DESCRIPTION
    Safely unregisters the scheduled task created by Install-DigitalDetectiveQueueWorker.ps1.
    Optionally removes local worker state (%LOCALAPPDATA%\DigitalDetective\agent_queue).
    Strictly prevents deletion of external directories and guards against symlinks/reparse points.
    Supports dry-run mode and is idempotent.

.PARAMETER TaskName
    Name of the scheduled task. Defaults to 'DigitalDetectiveQueueWorker'.

.PARAMETER StateDir
    Path to worker state directory. Must reside under %LOCALAPPDATA%\DigitalDetective\agent_queue.
    Defaults to %LOCALAPPDATA%\DigitalDetective\agent_queue.

.PARAMETER RemoveState
    If specified, deletes local logs, history, and lock files within the approved worker directory.

.PARAMETER DryRun
    Simulates uninstallation without deleting anything.
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory = $false)]
    [string]$TaskName = "DigitalDetectiveQueueWorker",

    [Parameter(Mandatory = $false)]
    [string]$StateDir,

    [Parameter(Mandatory = $false)]
    [switch]$RemoveState,

    [Parameter(Mandatory = $false)]
    [switch]$DryRun
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = "Stop"

# Canonical worker root under LOCALAPPDATA
$defaultCanonicalRoot = [System.IO.Path]::GetFullPath((Join-Path $env:LOCALAPPDATA "DigitalDetective\agent_queue"))

if (-not $StateDir) {
    $StateDir = $defaultCanonicalRoot
}

$canonicalStateDir = [System.IO.Path]::GetFullPath($StateDir)

$task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue

if ($DryRun) {
    Write-Output "-------------------- [UNINSTALL DRY-RUN] --------------------"
    Write-Output "Task Name       : $TaskName"
    Write-Output "Task Exists     : $(if ($task) { 'Yes' } else { 'No' })"
    Write-Output "State Directory : $canonicalStateDir (Remove: $RemoveState)"
    Write-Output "-------------------------------------------------------------"
    exit 0
}

# 1. Unregister Scheduled Task
if ($task) {
    try {
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
        Write-Output "Successfully unregistered scheduled task '$TaskName'."
    }
    catch {
        Write-Error "Failed to unregister scheduled task '$TaskName': $($_.Exception.Message)"
        exit 1
    }
}
else {
    Write-Output "Scheduled task '$TaskName' is not currently registered (no action needed)."
}

# 2. Clean stale lock file if present
$lockFile = Join-Path $canonicalStateDir "worker.lock"
if (Test-Path -Path $lockFile) {
    try {
        Remove-Item -Path $lockFile -Force -ErrorAction SilentlyContinue
        Write-Verbose "Removed worker lock file: $lockFile"
    } catch {}
}

# 3. Constrained destructive state cleanup
if ($RemoveState) {
    # Strictly enforce that StateDir is within %LOCALAPPDATA%\DigitalDetective\agent_queue
    $isUnderCanonicalRoot = $canonicalStateDir.Equals($defaultCanonicalRoot, [System.StringComparison]::OrdinalIgnoreCase) -or `
        $canonicalStateDir.StartsWith($defaultCanonicalRoot + "\", [System.StringComparison]::OrdinalIgnoreCase)

    if (-not $isUnderCanonicalRoot) {
        Write-Error "Safety restriction: -RemoveState is restricted to the worker-owned directory '$defaultCanonicalRoot'. Path '$canonicalStateDir' was rejected to prevent accidental deletion."
        exit 1
    }

    if (Test-Path -Path $canonicalStateDir) {
        # Guard against reparse points (symlinks / directory junctions)
        $dirItem = Get-Item -Path $canonicalStateDir -Force
        if (($dirItem.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -eq [System.IO.FileAttributes]::ReparsePoint) {
            Write-Error "Safety restriction: Target directory '$canonicalStateDir' is a reparse point (symlink/junction). Deletion aborted to prevent directory traversal."
            exit 1
        }

        try {
            Remove-Item -Path $canonicalStateDir -Recurse -Force
            Write-Output "Removed worker state directory: $canonicalStateDir"
        }
        catch {
            Write-Warning "Could not completely remove state directory: $($_.Exception.Message)"
        }
    }
}
else {
    Write-Output "Preserved worker logs and history at: $canonicalStateDir (pass -RemoveState to delete)."
}
