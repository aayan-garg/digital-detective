<#
.SYNOPSIS
    Test runner for Digital Detective unattended queue worker test suite.

.DESCRIPTION
    Executes tests/agent_queue/AgentQueueWorker.Tests.ps1 using Pester.
    Exits with code 0 on all tests passed, non-zero on failure.
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory = $false)]
    [string]$TestFile
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = "Stop"

if (-not $TestFile) {
    $TestFile = Join-Path $PSScriptRoot "..\..\tests\agent_queue\AgentQueueWorker.Tests.ps1"
}

if (-not (Test-Path -Path $TestFile)) {
    Write-Error "Test file not found: $TestFile"
    exit 1
}

Write-Output "Running queue worker tests: $TestFile"

$pesterModule = Get-Module -ListAvailable Pester | Select-Object -First 1
if (-not $pesterModule) {
    Write-Error "Pester module is not available in the current PowerShell environment."
    exit 1
}

# Run Pester
$result = Invoke-Pester -Script $TestFile -PassThru

Write-Output ""
Write-Output "================ Test Execution Summary ================"
Write-Output "Passed : $($result.PassedCount)"
Write-Output "Failed : $($result.FailedCount)"
Write-Output "Skipped: $($result.SkippedCount)"
Write-Output "Total  : $($result.TotalCount)"
Write-Output "Time   : $($result.Time)"
Write-Output "========================================================"

if ($result.FailedCount -gt 0) {
    exit 1
}

exit 0
