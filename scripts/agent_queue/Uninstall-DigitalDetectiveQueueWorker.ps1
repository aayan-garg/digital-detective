<#
.SYNOPSIS
    Uninstalls the Digital Detective Windows Scheduled Task and cleans local worker state.

.DESCRIPTION
    Safely unregisters the scheduled task created by Install-DigitalDetectiveQueueWorker.ps1.
    Optionally removes local worker state (%LOCALAPPDATA%\DigitalDetective\agent_queue).
    Supports dry-run mode and is idempotent.

.PARAMETER TaskName
    Name of the scheduled task. Defaults to 'DigitalDetectiveQueueWorker'.

.PARAMETER StateDir
    Path to worker state directory. Defaults to %LOCALAPPDATA%\DigitalDetective\agent_queue.

.PARAMETER RemoveState
    If specified, also deletes local logs, history, and lock files.

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

if (-not $StateDir) {
    $StateDir = Join-Path $env:LOCALAPPDATA "DigitalDetective\agent_queue"
}

$task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue

if ($DryRun) {
    Write-Output "-------------------- [UNINSTALL DRY-RUN] --------------------"
    Write-Output "Task Name       : $TaskName"
    Write-Output "Task Exists     : $(if ($task) { 'Yes' } else { 'No' })"
    Write-Output "State Directory : $StateDir (Remove: $RemoveState)"
    Write-Output "-------------------------------------------------------------"
    exit 0
}

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

# Clean stale lock file if present
$lockFile = Join-Path $StateDir "worker.lock"
if (Test-Path -Path $lockFile) {
    try {
        Remove-Item -Path $lockFile -Force -ErrorAction SilentlyContinue
        Write-Verbose "Removed worker lock file: $lockFile"
    } catch {}
}

if ($RemoveState) {
    if (Test-Path -Path $StateDir) {
        try {
            Remove-Item -Path $StateDir -Recurse -Force
            Write-Output "Removed worker state directory: $StateDir"
        }
        catch {
            Write-Warning "Could not completely remove state directory: $($_.Exception.Message)"
        }
    }
}
else {
    Write-Output "Preserved worker logs and history at: $StateDir (pass -RemoveState to delete)."
}
