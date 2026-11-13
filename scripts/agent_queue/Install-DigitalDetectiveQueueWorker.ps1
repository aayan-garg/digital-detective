<#
.SYNOPSIS
    Installs a user-scoped Windows Scheduled Task for the Digital Detective queue worker.

.DESCRIPTION
    Creates an opt-in scheduled task that starts on user logon and polls at a
    configurable interval (default: 30 minutes).
    Runs without administrator privileges, stores no passwords, and uses
    -StartWhenAvailable so missed tasks can run when the workstation wakes or logs on.

.PARAMETER RepoPath
    Root path of the digital-detective repository. Defaults to two levels above this script.

.PARAMETER TaskName
    Name of the scheduled task in Windows Task Scheduler. Defaults to 'DigitalDetectiveQueueWorker'.

.PARAMETER IntervalMinutes
    Repetition interval in minutes. Defaults to 30.

.PARAMETER Force
    Re-registers the scheduled task if it already exists.

.PARAMETER DryRun
    Displays the planned scheduled task configuration without creating or modifying anything.
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory = $false)]
    [string]$RepoPath,

    [Parameter(Mandatory = $false)]
    [string]$TaskName = "DigitalDetectiveQueueWorker",

    [Parameter(Mandatory = $false)]
    [ValidateRange(5, 1440)]
    [int]$IntervalMinutes = 30,

    [Parameter(Mandatory = $false)]
    [switch]$Force,

    [Parameter(Mandatory = $false)]
    [switch]$DryRun
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = "Stop"

# 1. Resolve repository root and worker script
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

$workerScript = Join-Path $RepoPath "scripts\agent_queue\Invoke-DigitalDetectiveQueueWorker.ps1"
if (-not (Test-Path -Path $workerScript -PathType Leaf)) {
    Write-Error "Worker script not found at expected location: $workerScript"
    exit 1
}

# 2. Check existing task for idempotency
$existingTask = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue

if ($existingTask -and -not $Force -and -not $DryRun) {
    Write-Output "Scheduled task '$TaskName' is already installed (State: $($existingTask.State)). Use -Force to overwrite."
    exit 0
}

# 3. Build Action
$psExe = (Get-Command powershell.exe -ErrorAction Stop).Source
$taskArgs = "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$workerScript`" -RepoPath `"$RepoPath`""
$action = New-ScheduledTaskAction -Execute $psExe -Argument $taskArgs -WorkingDirectory $RepoPath

# 4. Build Trigger (Logon + Interval Repetition)
$trigger = New-ScheduledTaskTrigger -AtLogOn -User "$env:USERDOMAIN\$env:USERNAME"
$repetitionTemplate = New-ScheduledTaskTrigger -Once -At (Get-Date) -RepetitionInterval (New-TimeSpan -Minutes $IntervalMinutes)
$trigger.Repetition = $repetitionTemplate.Repetition

# 5. Build Principal (Interactive User - No Admin, No Password Stored)
$principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType Interactive

# 6. Build Settings
$settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit (New-TimeSpan -Hours 2)

# 7. DryRun
if ($DryRun) {
    Write-Output "-------------------- [SCHEDULED TASK INSTALL DRY-RUN] --------------------"
    Write-Output "Task Name       : $TaskName"
    Write-Output "Action Exec     : $psExe"
    Write-Output "Action Arguments: $taskArgs"
    Write-Output "Working Dir     : $RepoPath"
    Write-Output "Trigger         : AtLogOn for user '$env:USERDOMAIN\$env:USERNAME' with repeat every $IntervalMinutes minutes"
    Write-Output "Principal       : Interactive ($env:USERDOMAIN\$env:USERNAME)"
    Write-Output "Settings        : StartWhenAvailable=True, BatteryExecution=True, ExecutionLimit=2h"
    Write-Output "Current Status  : $(if ($existingTask) { 'Already installed' } else { 'Not installed' })"
    Write-Output "--------------------------------------------------------------------------"
    exit 0
}

# 8. Unregister existing if -Force
if ($existingTask -and $Force) {
    Write-Verbose "Removing existing task '$TaskName' before re-registering..."
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
}

# 9. Register Task
try {
    $registered = Register-ScheduledTask `
        -TaskName $TaskName `
        -Action $action `
        -Trigger $trigger `
        -Principal $principal `
        -Settings $settings `
        -Description "Unattended background worker polling Digital Detective GitHub tasks and running Antigravity CLI."

    Write-Output "Successfully installed Scheduled Task '$TaskName' (State: $($registered.State))."
    Write-Output "Polls every $IntervalMinutes minutes while logged on. Missed runs will start on wake/logon."
}
catch {
    Write-Error "Failed to register Scheduled Task '$TaskName': $($_.Exception.Message)"
    exit 1
}
