<#
.SYNOPSIS
    Uninstalls the Digital Detective Windows Scheduled Task and cleans local worker state.

.DESCRIPTION
    Safely unregisters the scheduled task created by Install-DigitalDetectiveQueueWorker.ps1.
    Optionally removes local worker state (%LOCALAPPDATA%\DigitalDetective\agent_queue).
    Strictly locked to the single canonical worker-owned directory to prevent arbitrary deletion.
    Guards against symlinks/reparse points across all directory components.
    Supports dry-run mode and is idempotent.

.PARAMETER TaskName
    Name of the scheduled task. Defaults to 'DigitalDetectiveQueueWorker'.

.PARAMETER StateDir
    Optional validation path. If specified, must strictly match the canonical path
    %LOCALAPPDATA%\DigitalDetective\agent_queue. Any other path is rejected.

.PARAMETER RemoveState
    If specified, deletes local logs, history, and lock files within the canonical worker directory.

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

# Single canonical worker root under LOCALAPPDATA
$canonicalRoot = [System.IO.Path]::GetFullPath((Join-Path $env:LOCALAPPDATA "DigitalDetective\agent_queue"))

# Strictly enforce that StateDir (if provided) matches the canonical worker root
if ($StateDir) {
    $resolvedGiven = [System.IO.Path]::GetFullPath($StateDir)
    if (-not $resolvedGiven.Equals($canonicalRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
        Write-Error "Safety restriction: StateDir cannot be redirected to external path '$resolvedGiven'. Only the canonical worker directory '$canonicalRoot' is permitted."
        exit 1
    }
}

$targetStateDir = $canonicalRoot

# Reparse point check: ensure no path component from LOCALAPPDATA to worker directory is a junction or symlink
function Test-PathHasReparsePoint([string]$Path) {
    $curr = [System.IO.Path]::GetFullPath($Path)
    while ($curr -and (Test-Path -Path $curr)) {
        $item = Get-Item -Path $curr -Force
        if (($item.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -eq [System.IO.FileAttributes]::ReparsePoint) {
            return $true
        }
        $p = Split-Path -Path $curr -Parent
        if (-not $p -or $p -eq $curr) { break }
        $curr = $p
    }
    return $false
}

if (Test-PathHasReparsePoint $targetStateDir) {
    Write-Error "Safety restriction: Target directory '$targetStateDir' or one of its parents is a reparse point (symlink/junction). Cleanup aborted."
    exit 1
}

$task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue

if ($DryRun) {
    Write-Output "-------------------- [UNINSTALL DRY-RUN] --------------------"
    Write-Output "Task Name       : $TaskName"
    Write-Output "Task Exists     : $(if ($task) { 'Yes' } else { 'No' })"
    Write-Output "State Directory : $targetStateDir (Remove: $RemoveState)"
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

# 2. Clean lock file strictly within validated canonicalStateDir
$lockFile = Join-Path $targetStateDir "worker.lock"
if (Test-Path -Path $lockFile) {
    try {
        Remove-Item -Path $lockFile -Force -ErrorAction SilentlyContinue
        Write-Verbose "Removed worker lock file: $lockFile"
    } catch {}
}

# 3. Constrained destructive state cleanup
if ($RemoveState) {
    if (Test-Path -Path $targetStateDir) {
        try {
            Remove-Item -Path $targetStateDir -Recurse -Force
            Write-Output "Removed worker state directory: $targetStateDir"
        }
        catch {
            Write-Warning "Could not completely remove state directory: $($_.Exception.Message)"
        }
    }
}
else {
    Write-Output "Preserved worker logs and history at: $targetStateDir (pass -RemoveState to delete)."
}
