[CmdletBinding()]
param(
    [string[]]$TaskName = @("FLA VPN Runner", "FLA VPN Runner A5"),
    [ValidateRange(1, 120)]
    [int]$TimeoutSeconds = 15
)

$ErrorActionPreference = "Stop"

$selectedTasks = @($TaskName | ForEach-Object {
    Get-ScheduledTask -TaskName $_ -ErrorAction Stop
})
$configMarkers = @($selectedTasks | ForEach-Object {
    $quotedArguments = [regex]::Matches($_.Actions.Arguments, '"([^"]+)"')
    if ($quotedArguments.Count -ge 3) {
        $quotedArguments[2].Groups[1].Value
    }
})

function Test-ConfigMarker([string]$CommandLine) {
    foreach ($marker in $configMarkers) {
        if ($CommandLine -and $CommandLine.Contains($marker)) {
            return $true
        }
    }
    return $false
}

$allProcesses = @(Get-CimInstance Win32_Process)
$managedProcessIds = [System.Collections.Generic.HashSet[int]]::new()

function Add-ProcessTree([int]$ProcessId) {
    if (-not $managedProcessIds.Add($ProcessId)) {
        return
    }
    foreach ($child in $allProcesses | Where-Object { $_.ParentProcessId -eq $ProcessId }) {
        Add-ProcessTree -ProcessId $child.ProcessId
    }
}

$managedRoots = @($allProcesses | Where-Object {
    $_.Name -in @("wscript.exe", "powershell.exe") -and
    (Test-ConfigMarker -CommandLine $_.CommandLine)
})
foreach ($root in $managedRoots) {
    Add-ProcessTree -ProcessId $root.ProcessId
}

foreach ($task in $selectedTasks) {
    if ($task.State -eq "Running") {
        Stop-ScheduledTask -TaskName $task.TaskName
    }
}

$managedProcessIdList = @($managedProcessIds)
[array]::Reverse($managedProcessIdList)
foreach ($processId in $managedProcessIdList) {
    Stop-Process -Id $processId -Force -ErrorAction SilentlyContinue
}

$deadline = (Get-Date).AddSeconds($TimeoutSeconds)
do {
    $tasks = @($TaskName | ForEach-Object { Get-ScheduledTask -TaskName $_ })
    $running = @($tasks | Where-Object { $_.State -eq "Running" })
    $remainingProcesses = @($managedProcessIdList | Where-Object {
        Get-Process -Id $_ -ErrorAction SilentlyContinue
    })
    if ($running.Count -eq 0 -and $remainingProcesses.Count -eq 0) {
        break
    }
    Start-Sleep -Milliseconds 500
} while ((Get-Date) -lt $deadline)

if ($running.Count -gt 0 -or $remainingProcesses.Count -gt 0) {
    $details = ($running | ForEach-Object { "$($_.TaskName)=$($_.State)" }) -join ", "
    throw "Relay tasks did not stop within $TimeoutSeconds seconds: $details; remaining PIDs: $remainingProcesses"
}

$tasks | Select-Object TaskName, State
