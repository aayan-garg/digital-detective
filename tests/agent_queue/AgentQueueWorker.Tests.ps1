# AgentQueueWorker.Tests.ps1
# Pester unit and contract tests for Digital Detective unattended queue worker.
# Strictly deterministic: no live network calls, no model invocations.

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

Describe "QueueWorkerCore - Queue Task Selection & Exact Branch Matching" {
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

    It "Resumes an IN-PROGRESS issue when checked out branch matches exactly" {
        $issues = @(
            [PSCustomObject]@{ number = 4; title = "[DD-AGENT:IN-PROGRESS] Build queue runner"; createdAt = "2026-11-13T05:00:00Z" }
        )
        $decision = Select-QueueTask -Issues $issues -CurrentBranch "agent/issue-4-windows-queue-runner"
        $decision.Action | Should Be "Resume"
        $decision.SelectedIssue.number | Should Be 4
    }

    It "Rejects an adjacent-number branch (issue-4 vs issue-40 regression test)" {
        $issues = @(
            [PSCustomObject]@{ number = 4; title = "[DD-AGENT:IN-PROGRESS] Build queue runner"; createdAt = "2026-11-13T05:00:00Z" }
        )
        # Branch is issue-40, issue is 4: MUST NOT MATCH
        $decision = Select-QueueTask -Issues $issues -CurrentBranch "agent/issue-40-different-task"
        $decision.Action | Should Be "Blocked"
        $decision.Reason | Should Match "does not match"
    }

    It "Rejects an unanchored number match (issue-4 vs issue-14)" {
        $issues = @(
            [PSCustomObject]@{ number = 4; title = "[DD-AGENT:IN-PROGRESS] Task"; createdAt = "2026-11-13T05:00:00Z" }
        )
        $decision = Select-QueueTask -Issues $issues -CurrentBranch "agent/issue-14-other"
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

Describe "QueueWorkerCore - Post-Run State & PR Verification" {
    $initial = [PSCustomObject]@{ number = 4; title = "[DD-AGENT:IN-PROGRESS] Build runner" }

    It "Validates TASK_COMPLETED_REVIEW_READY when issue is open in [DD-AGENT:REVIEW] with an active open PR" {
        $post = [PSCustomObject]@{ number = 4; title = "[DD-AGENT:REVIEW] Build runner"; state = "open" }
        $pr = [PSCustomObject]@{ number = 6; state = "OPEN"; title = "feat: add runner"; url = "https://github.com/pr/6" }
        $result = Evaluate-PostRunState -InitialIssue $initial -PostIssue $post -AssociatedPr $pr -ExitCode 0 -IsSoftDenied $false
        $result.Outcome | Should Be "TASK_COMPLETED_REVIEW_READY"
        $result.TaskCompleted | Should Be $true
    }

    It "Rejects completion when issue is [DD-AGENT:REVIEW] but associated PR is missing" {
        $post = [PSCustomObject]@{ number = 4; title = "[DD-AGENT:REVIEW] Build runner"; state = "open" }
        $result = Evaluate-PostRunState -InitialIssue $initial -PostIssue $post -AssociatedPr $null -ExitCode 0 -IsSoftDenied $false
        $result.Outcome | Should Be "REVIEW_TITLE_WITHOUT_PR"
        $result.TaskCompleted | Should Be $false
        $result.Reason | Should Match "no associated pull request was found"
    }

    It "Rejects completion when issue is [DD-AGENT:REVIEW] but associated PR is closed or merged" {
        $post = [PSCustomObject]@{ number = 4; title = "[DD-AGENT:REVIEW] Build runner"; state = "open" }
        $pr = [PSCustomObject]@{ number = 6; state = "CLOSED"; title = "feat: add runner"; url = "https://github.com/pr/6" }
        $result = Evaluate-PostRunState -InitialIssue $initial -PostIssue $post -AssociatedPr $pr -ExitCode 0 -IsSoftDenied $false
        $result.Outcome | Should Be "REVIEW_TITLE_PR_NOT_OPEN"
        $result.TaskCompleted | Should Be $false
        $result.Reason | Should Match "instead of OPEN"
    }

    It "Rejects completion and raises policy violation if issue was closed" {
        $post = [PSCustomObject]@{ number = 4; title = "[DD-AGENT:REVIEW] Build runner"; state = "closed" }
        $pr = [PSCustomObject]@{ number = 6; state = "OPEN"; title = "feat: add runner"; url = "https://github.com/pr/6" }
        $result = Evaluate-PostRunState -InitialIssue $initial -PostIssue $post -AssociatedPr $pr -ExitCode 0 -IsSoftDenied $false
        $result.Outcome | Should Be "POLICY_VIOLATION_CLOSED_ISSUE"
        $result.TaskCompleted | Should Be $false
        $result.Reason | Should Match "Policy violation"
    }

    It "Accepts TASK_COMPLETED_DONE when issue is open and marked [DD-AGENT:DONE]" {
        $post = [PSCustomObject]@{ number = 4; title = "[DD-AGENT:DONE] Build runner"; state = "open" }
        $result = Evaluate-PostRunState -InitialIssue $initial -PostIssue $post -AssociatedPr $null -ExitCode 0 -IsSoftDenied $false
        $result.Outcome | Should Be "TASK_COMPLETED_DONE"
        $result.TaskCompleted | Should Be $true
    }

    It "Flags CLI_SUCCEEDED_UNTRANSITIONED when CLI exits 0 but issue remains IN-PROGRESS" {
        $post = [PSCustomObject]@{ number = 4; title = "[DD-AGENT:IN-PROGRESS] Build runner"; state = "open" }
        $result = Evaluate-PostRunState -InitialIssue $initial -PostIssue $post -AssociatedPr $null -ExitCode 0 -IsSoftDenied $false
        $result.Outcome | Should Be "CLI_SUCCEEDED_UNTRANSITIONED"
        $result.TaskCompleted | Should Be $false
        $result.Reason | Should Match "did not transition"
    }

    It "Flags TIMEOUT on exit code 124" {
        $post = [PSCustomObject]@{ number = 4; title = "[DD-AGENT:IN-PROGRESS] Build runner"; state = "open" }
        $result = Evaluate-PostRunState -InitialIssue $initial -PostIssue $post -AssociatedPr $null -ExitCode 124 -IsSoftDenied $false
        $result.Outcome | Should Be "TIMEOUT"
        $result.TaskCompleted | Should Be $false
    }

    It "Flags CLI_PROCESS_FAILED on non-zero exit code" {
        $post = [PSCustomObject]@{ number = 4; title = "[DD-AGENT:IN-PROGRESS] Build runner"; state = "open" }
        $result = Evaluate-PostRunState -InitialIssue $initial -PostIssue $post -AssociatedPr $null -ExitCode 1 -IsSoftDenied $false
        $result.Outcome | Should Be "CLI_PROCESS_FAILED"
        $result.TaskCompleted | Should Be $false
    }

    It "Flags SOFT_DENIED even if CLI process exited with code 0" {
        $post = [PSCustomObject]@{ number = 4; title = "[DD-AGENT:REVIEW] Build runner"; state = "open" }
        $pr = [PSCustomObject]@{ number = 6; state = "OPEN"; title = "feat"; url = "url" }
        $result = Evaluate-PostRunState -InitialIssue $initial -PostIssue $post -AssociatedPr $pr -ExitCode 0 -IsSoftDenied $true
        $result.Outcome | Should Be "SOFT_DENIED"
        $result.TaskCompleted | Should Be $false
    }
}

Describe "QueueWorkerCore - Security, Injection & Representative Stderr Fixtures" {
    It "Treats malicious command strings in issue titles as harmless data" {
        $maliciousTitle = "[DD-AGENT:READY] Task; calc.exe; rm -rf /; `$(whoami) | Out-Null"
        $issues = @(
            [PSCustomObject]@{ number = 99; title = $maliciousTitle; createdAt = "2026-11-13T01:00:00Z" }
        )
        $decision = Select-QueueTask -Issues $issues -CurrentBranch "main"
        $decision.Action | Should Be "Execute"
        $decision.SelectedIssue.title | Should Be $maliciousTitle
    }

    It "Detects representative captured Antigravity CLI stderr fixtures for soft-denials" {
        # Fixture 1: Headless print-mode tool approval denial
        $fixtureHeadlessApproval = @"
[2026-11-13T10:00:00Z] INFO agy::session: Initializing headless session for prompt '/process-digital-detective-queue'
[2026-11-13T10:00:05Z] WARN agy::policy: The model requested execution of 'command', but approval is required in headless mode. The operation was soft-denied.
[2026-11-13T10:00:06Z] INFO agy::turn: Turn finished with soft-denied tool calls.
"@
        Test-SoftDenial -OutputText $fixtureHeadlessApproval | Should Be $true

        # Fixture 2: MCP server approval gate
        $fixtureMcpGate = @"
Error: MCP tool 'github.issue_write' requires user approval. Auto-approval is disabled in non-interactive sessions.
Operation requires an unavailable approval.
"@
        Test-SoftDenial -OutputText $fixtureMcpGate | Should Be $true

        # Fixture 3: Filesystem policy denial
        $fixtureFsPolicy = @"
SecurityError: write_file operation was not authorized by security policy. Tool call not approved.
"@
        Test-SoftDenial -OutputText $fixtureFsPolicy | Should Be $true

        # Fixture 4: Clean successful output
        $fixtureSuccess = @"
{"type":"result","status":"success","message":"Completed task #4 and submitted PR #6."}
"@
        Test-SoftDenial -OutputText $fixtureSuccess -ExitCode 0 | Should Be $false
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

Describe "QueueWorkerCore - Batch Wrapper Execution (.cmd vs .exe)" {
    $tempDir = Join-Path $env:TEMP ("dd_cmd_test_" + [guid]::NewGuid().ToString())
    New-Item -Path $tempDir -ItemType Directory -Force | Out-Null
    $dummyCmd = Join-Path $tempDir "dummy_agy.cmd"
    Set-Content -Path $dummyCmd -Value "@echo off`necho DUMMY_OUTPUT`nexit 0" -Encoding ASCII

    It "Executes a .cmd batch script safely via cmd.exe wrapper" {
        $res = Invoke-AntigravityCli -AgyPath $dummyCmd -Prompt "test" -WorkingDirectory $tempDir -TimeoutSeconds 10
        $res.ExitCode | Should Be 0
        $res.LaunchMode | Should Be "cmd.exe wrapper"
        $res.StandardOutput | Should Match "DUMMY_OUTPUT"
    }

    Remove-Item $tempDir -Recurse -Force -ErrorAction SilentlyContinue
}

Describe "QueueWorker Scripts - Fail-Closed & Empty-Queue Isolation (Deterministic)" {
    $scriptDir = Join-Path $PSScriptRoot "..\..\scripts\agent_queue"
    $workerScript = Join-Path $scriptDir "Invoke-DigitalDetectiveQueueWorker.ps1"
    $mockBinDir = Join-Path $env:TEMP ("dd_mock_bin_" + [guid]::NewGuid().ToString())
    New-Item -Path $mockBinDir -ItemType Directory -Force | Out-Null

    # Create dummy agy.cmd that records invocations of agy -p (headless runs)
    $dummyAgy = Join-Path $mockBinDir "agy.cmd"
    $agyInvocationLog = Join-Path $mockBinDir "agy_invocations.log"
    Set-Content -Path $dummyAgy -Value "@echo off`r`nif `"%1`"==`"--version`" (echo agy version 1.5.8 & exit /b 0)`r`nif `"%1`"==`"-p`" (echo CALLED >> `"$agyInvocationLog`" & exit /b 0)`r`nexit /b 0" -Encoding ASCII

    # Create dummy git.exe (batch alias for test harness)
    $dummyGit = Join-Path $mockBinDir "git.cmd"
    Set-Content -Path $dummyGit -Value "@echo off`necho %* | findstr /i `"rev-parse`" >nul && (echo main & exit /b 0)`necho %* | findstr /i `"status`" >nul && exit /b 0`nexit /b 0" -Encoding ASCII

    It "Fails closed with exit code 2 when required executable is missing" {
        $nonExistentGh = "C:\nonexistent\gh.exe"
        $proc = Start-Process powershell.exe -ArgumentList "-NoProfile -ExecutionPolicy Bypass -File `"$workerScript`" -GhPath `"$nonExistentGh`"" -NoNewWindow -Wait -PassThru
        $proc.ExitCode | Should Be 2
    }

    It "Fails closed with exit code 3 when gh auth status fails" {
        $dummyFailingGh = Join-Path $mockBinDir "gh_fail.cmd"
        Set-Content -Path $dummyFailingGh -Value "@echo off`necho Not logged in & exit /b 1" -Encoding ASCII

        $proc = Start-Process powershell.exe -ArgumentList "-NoProfile -ExecutionPolicy Bypass -File `"$workerScript`" -GhPath `"$dummyFailingGh`" -AgyPath `"$dummyAgy`" -GitPath `"$dummyGit`"" -NoNewWindow -Wait -PassThru
        $proc.ExitCode | Should Be 3
    }

    It "Exits cleanly with code 0 on empty queue without invoking agy -p" {
        $dummyEmptyGh = Join-Path $mockBinDir "gh_empty.cmd"
        Set-Content -Path $dummyEmptyGh -Value "@echo off`nif `"%1`"==`"auth`" (exit /b 0)`nif `"%1`"==`"issue`" (echo [] & exit /b 0)`nexit /b 0" -Encoding ASCII

        if (Test-Path $agyInvocationLog) { Remove-Item $agyInvocationLog -Force }

        $proc = Start-Process powershell.exe -ArgumentList "-NoProfile -ExecutionPolicy Bypass -File `"$workerScript`" -GhPath `"$dummyEmptyGh`" -AgyPath `"$dummyAgy`" -GitPath `"$dummyGit`"" -NoNewWindow -Wait -PassThru
        $proc.ExitCode | Should Be 0
        Test-Path $agyInvocationLog | Should Be $false
    }

    It "Exits cleanly with code 0 on review-pending queue without invoking agy -p" {
        $dummyReviewGh = Join-Path $mockBinDir "gh_review.cmd"
        $reviewJsonFile = Join-Path $mockBinDir "review.json"
        $jsonPayload = @(
            @{
                number    = 3
                title     = "[DD-AGENT:REVIEW] Audit"
                createdAt = "2026-11-13T01:00:00Z"
            }
        ) | ConvertTo-Json -Compress
        Set-Content -Path $reviewJsonFile -Value $jsonPayload -Encoding ASCII

        Set-Content -Path $dummyReviewGh -Value "@echo off`r`nif `"%1`"==`"auth`" (exit /b 0)`r`nif `"%1`"==`"issue`" (type `"$reviewJsonFile`" & exit /b 0)`r`nexit /b 0" -Encoding ASCII

        if (Test-Path $agyInvocationLog) { Remove-Item $agyInvocationLog -Force }

        $proc = Start-Process powershell.exe -ArgumentList "-NoProfile -ExecutionPolicy Bypass -File `"$workerScript`" -GhPath `"$dummyReviewGh`" -AgyPath `"$dummyAgy`" -GitPath `"$dummyGit`"" -NoNewWindow -Wait -PassThru
        $proc.ExitCode | Should Be 0
        Test-Path $agyInvocationLog | Should Be $false
    }

    Remove-Item $mockBinDir -Recurse -Force -ErrorAction SilentlyContinue
}

Describe "QueueWorker Scripts - Dry-Run, Diagnostics & Cleanup Safety Contracts" {
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

    It "Rejects destructive cleanup on an external directory, preserving canary and worker.lock" {
        $uninstallScript = Join-Path $scriptDir "Uninstall-DigitalDetectiveQueueWorker.ps1"
        $dangerousExternalDir = Join-Path $env:TEMP ("dd_external_test_" + [guid]::NewGuid().ToString())
        New-Item -Path $dangerousExternalDir -ItemType Directory -Force | Out-Null

        $canaryFile = Join-Path $dangerousExternalDir "do_not_delete.txt"
        Set-Content -Path $canaryFile -Value "Safe Canary Content"
        $lockCanary = Join-Path $dangerousExternalDir "worker.lock"
        Set-Content -Path $lockCanary -Value "External Lock File"

        try {
            $errFile = [System.IO.Path]::GetTempFileName()
            $proc = Start-Process powershell.exe -ArgumentList "-NoProfile -ExecutionPolicy Bypass -File `"$uninstallScript`" -StateDir `"$dangerousExternalDir`" -RemoveState" -NoNewWindow -Wait -PassThru -RedirectStandardError $errFile
            $proc.ExitCode | Should Be 1

            # Both canary file AND worker.lock must remain untouched
            Test-Path -Path $canaryFile | Should Be $true
            Test-Path -Path $lockCanary | Should Be $true

            $errContent = if (Test-Path -Path $errFile) { Get-Content -Path $errFile -Raw } else { "" }
            $errContent | Should Match "Safety restriction"
            Remove-Item $errFile -Force -ErrorAction SilentlyContinue
        }
        finally {
            Remove-Item $dangerousExternalDir -Recurse -Force -ErrorAction SilentlyContinue
        }
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
