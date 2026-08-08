[CmdletBinding()]
param(
    [string]$TaskName = "FLA VPN Runner",
    [string]$ConfigPath,
    [string]$TokenPath,
    [string]$LogFile,
    [switch]$StartNow
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$runnerScript = Join-Path $PSScriptRoot "run_runner_windows.ps1"
if (-not $ConfigPath) {
    $ConfigPath = Join-Path $repoRoot ".local-secrets\runner-config.json"
}
if (-not $TokenPath) {
    $TokenPath = Join-Path $repoRoot ".local-secrets\runner-token.clixml"
}
if (-not $LogFile) {
    $LogFile = Join-Path $repoRoot ".local-secrets\runner.log"
}

foreach ($requiredPath in @($runnerScript, $ConfigPath, $TokenPath)) {
    if (-not (Test-Path -LiteralPath $requiredPath)) {
        throw "Required Relay file does not exist: $requiredPath"
    }
}

$powerShell = (Get-Command powershell.exe).Source
$arguments = @(
    "-NoProfile"
    "-NonInteractive"
    "-ExecutionPolicy Bypass"
    "-WindowStyle Hidden"
    "-File `"$runnerScript`""
    "-ConfigPath `"$ConfigPath`""
    "-TokenPath `"$TokenPath`""
    "-LogFile `"$LogFile`""
) -join " "

$identity = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
$action = New-ScheduledTaskAction -Execute $powerShell -Argument $arguments -WorkingDirectory $repoRoot
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $identity
$principal = New-ScheduledTaskPrincipal -UserId $identity -LogonType Interactive -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -RestartCount 10 `
    -RestartInterval (New-TimeSpan -Minutes 1) `
    -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -MultipleInstances IgnoreNew

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $action `
    -Trigger $trigger `
    -Principal $principal `
    -Settings $settings `
    -Description "Outbound-only Cloudflare performance queue Relay for the VPN-connected NPU server." `
    -Force | Out-Null

if ($StartNow) {
    Start-ScheduledTask -TaskName $TaskName
}

Write-Host "Scheduled task installed: $TaskName"
Write-Host "Relay config: $ConfigPath"
Write-Host "Relay log: $LogFile"
