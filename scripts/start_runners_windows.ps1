[CmdletBinding()]
param(
    [string[]]$TaskName = @("FLA VPN Runner", "FLA VPN Runner A5"),
    [ValidateRange(1, 120)]
    [int]$TimeoutSeconds = 15
)

$ErrorActionPreference = "Stop"

foreach ($name in $TaskName) {
    $task = Get-ScheduledTask -TaskName $name -ErrorAction Stop
    if ($task.State -ne "Running") {
        Start-ScheduledTask -TaskName $name
    }
}

$deadline = (Get-Date).AddSeconds($TimeoutSeconds)
do {
    $tasks = @($TaskName | ForEach-Object { Get-ScheduledTask -TaskName $_ })
    $pending = @($tasks | Where-Object { $_.State -ne "Running" })
    if ($pending.Count -eq 0) {
        break
    }
    Start-Sleep -Milliseconds 500
} while ((Get-Date) -lt $deadline)

if ($pending.Count -gt 0) {
    $details = ($pending | ForEach-Object { "$($_.TaskName)=$($_.State)" }) -join ", "
    throw "Relay tasks did not start within $TimeoutSeconds seconds: $details"
}

$tasks | Select-Object TaskName, State
