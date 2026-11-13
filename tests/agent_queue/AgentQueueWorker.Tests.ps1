# AgentQueueWorker.Tests.ps1
# Pester unit and contract tests for Digital Detective unattended queue worker.
# Strictly deterministic: no network calls, no model invocations.

$coreModule = Join-Path $PSScriptRoot "..\..\scripts\agent_queue\QueueWorkerCore.psm1"
Import-Module (Resolve-Path $coreModule).Path -Force -DisableNameChecking

Describe "QueueWorkerCore - Working Tree Cleanliness" {
    It "Approves an empty porcelain status" {
        $result = Test-WorkingTreeCleanliness -PorcelainLines @()
        $result.IsClean | Should Be $true
        $result.Violations.Count | Should Be 0
    }

    It "Approves exactly the five pre-existing compatibility aliases when untracked" {
        $porcelain = @(
            "?? src/agent/__init__.py",
            "?? src/agent/disambiguator.py",
            "?? src/features/__init__.py",
            "?? src/features/causal_residuals.py",
            "?? src/features/log_scorer.py"
        )
        $result = Test-WorkingTreeCleanliness -PorcelainLines $porcelain
        $result.IsClean | Should Be $true
        $result.Violations.Count | Should Be 0
    }

    It "Accepts backslash paths for the five approved aliases on Windows" {
        $porcelain = @(
            "?? src\agent\__init__.py",
            "?? src\agent\disambiguator.py",
            "?? src\features\__init__.py",
            "?? src\features\causal_residuals.py",
            "?? src\features\log_scorer.py"
        )
        $result = Test-WorkingTreeCleanliness -PorcelainLines $porcelain
        $result.IsClean | Should Be $true
    }

    It "Rejects an unauthorized untracked file" {
        $porcelain = @(
            "?? src/agent/__init__.py",
            "?? src/agent/unauthorized_new_file.py"
        )
        $result = Test-WorkingTreeCleanliness -PorcelainLines $porcelain
        $result.IsClean | Should Be $false
        $result.Violations.Count | Should Be 1
        $result.Violations[0] | Should Match "unauthorized_new_file"
    }

    It "Rejects tracked modified files (unstaged)" {
        $porcelain = @(
            " M README.md"
        )
        $result = Test-WorkingTreeCleanliness -PorcelainLines $porcelain
        $result.IsClean | Should Be $false
        $result.Violations.Count | Should Be 1
    }

    It "Rejects staged additions or modifications" {
        $porcelain = @(
            "M  src/digital_detective/pipeline.py",
            "A  new_staged_script.ps1"
        )
        $result = Test-WorkingTreeCleanliness -PorcelainLines $porcelain
        $result.IsClean | Should Be $false
        $result.Violations.Count | Should Be 2
    }

    It "Rejects deletions and unmerged conflicts" {
        $porcelain = @(
            " D deleted_file.txt",
            "UU conflict.py"
        )
        $result = Test-WorkingTreeCleanliness -PorcelainLines $porcelain
        $result.IsClean | Should Be $false
        $result.Violations.Count | Should Be 2
    }
}

Describe "QueueWorkerCore - Queue Task Selection & Priority" {
    It "Reports EmptyQueue when no issues exist" {
        $decision = Select-QueueTask -Issues @() -CurrentBranch "main"
        $decision.Action | Should Be "EmptyQueue"
        $decision.SelectedIssue | Should Be $null
    }

    It "Blocks when an issue is in [DD-AGENT:REVIEW]" {
        $issues = @(
            [PSCustomObject]@{ number = 3; title = "[DD-AGENT:REVIEW] Audit architecture"; createdAt = "2026-11-13T04:00:00Z" },
            [PSCustomObject]@{ number = 4; title = "[DD-AGENT:READY] Build queue runner"; createdAt = "2026-11-13T05:00:00Z" }
        )
        $decision = Select-QueueTask -Issues $issues -CurrentBranch "main"
        $decision.Action | Should Be "WaitReview"
        $decision.SelectedIssue.number | Should Be 3
        $decision.Reason | Should Match "awaiting architect review"
    }

    It "Blocks if multiple issues are [DD-AGENT:IN-PROGRESS]" {
        $issues = @(
            [PSCustomObject]@{ number = 3; title = "[DD-AGENT:IN-PROGRESS] Task A"; createdAt = "2026-11-13T04:00:00Z" },
            [PSCustomObject]@{ number = 4; title = "[DD-AGENT:IN-PROGRESS] Task B"; createdAt = "2026-11-13T05:00:00Z" }
        )
        $decision = Select-QueueTask -Issues $issues -CurrentBranch "agent/issue-3-a"
        $decision.Action | Should Be "Blocked"
        $decision.Reason | Should Match "Multiple issues"
    }

    It "Resumes an IN-PROGRESS issue when checked out branch matches" {
        $issues = @(
            [PSCustomObject]@{ number = 4; title = "[DD-AGENT:IN-PROGRESS] Build queue runner"; createdAt = "2026-11-13T05:00:00Z" }
        )
        $decision = Select-QueueTask -Issues $issues -CurrentBranch "agent/issue-4-windows-queue-runner"
        $decision.Action | Should Be "Resume"
        $decision.SelectedIssue.number | Should Be 4
    }

    It "Blocks an IN-PROGRESS issue when checked out branch does not match" {
        $issues = @(
            [PSCustomObject]@{ number = 4; title = "[DD-AGENT:IN-PROGRESS] Build queue runner"; createdAt = "2026-11-13T05:00:00Z" }
        )
        $decision = Select-QueueTask -Issues $issues -CurrentBranch "main"
        $decision.Action | Should Be "Blocked"
        $decision.Reason | Should Match "does not match"
    }

    It "Prioritizes REVISION over READY task" {
        $issues = @(
            [PSCustomObject]@{ number = 4; title = "[DD-AGENT:READY] Build queue runner"; createdAt = "2026-11-13T03:00:00Z" },
            [PSCustomObject]@{ number = 5; title = "[DD-AGENT:REVISION] Update docs"; createdAt = "2026-11-13T06:00:00Z" }
        )
        $decision = Select-QueueTask -Issues $issues -CurrentBranch "main"
        $decision.Action | Should Be "Execute"
        $decision.TaskType | Should Be "Revision"
        $decision.SelectedIssue.number | Should Be 5
    }

    It "Selects oldest REVISION when multiple revisions exist" {
        $issues = @(
            [PSCustomObject]@{ number = 5; title = "[DD-AGENT:REVISION] Newer revision"; createdAt = "2026-11-13T07:00:00Z" },
            [PSCustomObject]@{ number = 2; title = "[DD-AGENT:REVISION] Older revision"; createdAt = "2026-11-13T02:00:00Z" }
        )
        $decision = Select-QueueTask -Issues $issues -CurrentBranch "main"
        $decision.Action | Should Be "Execute"
        $decision.SelectedIssue.number | Should Be 2
    }

    It "Selects oldest READY task when no revisions exist and on main" {
        $issues = @(
            [PSCustomObject]@{ number = 6; title = "[DD-AGENT:READY] Newer ready"; createdAt = "2026-11-13T08:00:00Z" },
            [PSCustomObject]@{ number = 4; title = "[DD-AGENT:READY] Older ready"; createdAt = "2026-11-13T05:00:00Z" }
        )
        $decision = Select-QueueTask -Issues $issues -CurrentBranch "main"
        $decision.Action | Should Be "Execute"
        $decision.TaskType | Should Be "Ready"
        $decision.SelectedIssue.number | Should Be 4
    }

    It "Blocks READY task if current branch is not main" {
        $issues = @(
            [PSCustomObject]@{ number = 4; title = "[DD-AGENT:READY] Build queue runner"; createdAt = "2026-11-13T05:00:00Z" }
        )
        $decision = Select-QueueTask -Issues $issues -CurrentBranch "feature/some-feature"
        $decision.Action | Should Be "BlockedNotOnMain"
        $decision.Reason | Should Match "must be on 'main'"
    }
}

Describe "QueueWorkerCore - Concurrency Locking" {
    $tempLockDir = Join-Path $env:TEMP ("dd_test_lock_" + [guid]::NewGuid().ToString())
    $tempLockFile = Join-Path $tempLockDir "test_worker.lock"

    It "Acquires an exclusive process file lock" {
        $lock1 = Acquire-WorkerLock -LockFilePath $tempLockFile
        $lock1.Success | Should Be $true
        Test-Path $tempLockFile | Should Be $true

        # Attempt second concurrent acquisition on same file
        $lock2 = Acquire-WorkerLock -LockFilePath $tempLockFile
        $lock2.Success | Should Be $false
        $lock2.ErrorMessage | Should Match "already held"

        # Release first lock
        Release-WorkerLock -LockResult $lock1

        # Subsequent acquisition succeeds
        $lock3 = Acquire-WorkerLock -LockFilePath $tempLockFile
        $lock3.Success | Should Be $true
        Release-WorkerLock -LockResult $lock3

        # Cleanup
        Remove-Item $tempLockDir -Recurse -Force -ErrorAction SilentlyContinue
    }
}

Describe "QueueWorkerCore - Security, Injection & Soft-Denials" {
    It "Treats malicious command strings in issue titles as harmless data" {
        $maliciousTitle = "[DD-AGENT:READY] Task; calc.exe; rm -rf /; `$(whoami) | Out-Null"
        $issues = @(
            [PSCustomObject]@{ number = 99; title = $maliciousTitle; createdAt = "2026-11-13T01:00:00Z" }
        )
        $decision = Select-QueueTask -Issues $issues -CurrentBranch "main"
        $decision.Action | Should Be "Execute"
        $decision.SelectedIssue.title | Should Be $maliciousTitle
    }

    It "Detects headless tool soft-denials" {
        Test-SoftDenial -OutputText "Model requested tool execution, but operation requires an unavailable approval" | Should Be $true
        Test-SoftDenial -OutputText "Error: tool approval required for write_file" | Should Be $true
        Test-SoftDenial -OutputText "The command was denied by user or policy" | Should Be $true
        Test-SoftDenial -OutputText "Execution finished cleanly with report in PR #5" -ExitCode 0 | Should Be $false
    }

    It "Sanitizes tokens and secrets from log text" {
        $dirty = "Auth token: ghp_1234567890abcdef1234567890abcdef1234 and pat github_pat_11AABBCCDDEEFF001122334455 with Bearer eyJhbGciOiJIUzI1NiJ9.test.sig"
        $clean = Sanitize-LogContent -Text $dirty
        $clean | Should Not Match "ghp_1234"
        $clean | Should Not Match "github_pat_"
        $clean | Should Match "\[REDACTED_GH_PAT\]"
        $clean | Should Match "\[REDACTED_GH_FINE_GRAINED_PAT\]"
        $clean | Should Match "Bearer \[REDACTED_TOKEN\]"
    }
}

Describe "QueueWorker Scripts - Dry-Run & Diagnostics Contract" {
    $scriptDir = Join-Path $PSScriptRoot "..\..\scripts\agent_queue"

    It "Install-DigitalDetectiveQueueWorker -DryRun executes without side-effects" {
        $installScript = Join-Path $scriptDir "Install-DigitalDetectiveQueueWorker.ps1"
        $output = & powershell.exe -ExecutionPolicy Bypass -File $installScript -DryRun
        $LASTEXITCODE | Should Be 0
        ($output -join "`n") | Should Match "SCHEDULED TASK INSTALL DRY-RUN"
    }

    It "Uninstall-DigitalDetectiveQueueWorker -DryRun executes without side-effects" {
        $uninstallScript = Join-Path $scriptDir "Uninstall-DigitalDetectiveQueueWorker.ps1"
        $output = & powershell.exe -ExecutionPolicy Bypass -File $uninstallScript -DryRun
        $LASTEXITCODE | Should Be 0
        ($output -join "`n") | Should Match "UNINSTALL DRY-RUN"
    }

    It "Get-DigitalDetectiveQueueWorkerStatus -AsJson returns parseable JSON" {
        $statusScript = Join-Path $scriptDir "Get-DigitalDetectiveQueueWorkerStatus.ps1"
        $jsonRaw = & powershell.exe -ExecutionPolicy Bypass -File $statusScript -AsJson
        $LASTEXITCODE | Should Be 0
        $parsed = ConvertFrom-Json ($jsonRaw -join "`n")
        $parsed.TaskScheduler | Should Not Be $null
        $parsed.Workspace | Should Not Be $null
        $parsed.Executables | Should Not Be $null
    }

    It "Installs, detects idempotency, and uninstalls a test scheduled task cleanly" {
        $testTaskName = "DigitalDetectiveQueueTestRunner_" + [guid]::NewGuid().ToString("N").Substring(0, 8)
        $installScript = Join-Path $scriptDir "Install-DigitalDetectiveQueueWorker.ps1"
        $uninstallScript = Join-Path $scriptDir "Uninstall-DigitalDetectiveQueueWorker.ps1"

        try {
            # 1. Install task
            $outInstall1 = & powershell.exe -ExecutionPolicy Bypass -File $installScript -TaskName $testTaskName
            $LASTEXITCODE | Should Be 0
            $registeredTask = Get-ScheduledTask -TaskName $testTaskName -ErrorAction SilentlyContinue
            $registeredTask | Should Not Be $null

            # 2. Idempotent install (should succeed without error)
            $outInstall2 = & powershell.exe -ExecutionPolicy Bypass -File $installScript -TaskName $testTaskName
            $LASTEXITCODE | Should Be 0
            ($outInstall2 -join "`n") | Should Match "already installed"

            # 3. Uninstall task
            $outUninstall1 = & powershell.exe -ExecutionPolicy Bypass -File $uninstallScript -TaskName $testTaskName
            $LASTEXITCODE | Should Be 0
            (Get-ScheduledTask -TaskName $testTaskName -ErrorAction SilentlyContinue) | Should Be $null

            # 4. Idempotent uninstall (should succeed when task does not exist)
            $outUninstall2 = & powershell.exe -ExecutionPolicy Bypass -File $uninstallScript -TaskName $testTaskName
            $LASTEXITCODE | Should Be 0
            ($outUninstall2 -join "`n") | Should Match "not currently registered"
        }
        finally {
            Unregister-ScheduledTask -TaskName $testTaskName -Confirm:$false -ErrorAction SilentlyContinue
        }
    }
}
