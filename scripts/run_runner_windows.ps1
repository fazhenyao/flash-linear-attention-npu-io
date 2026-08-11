[CmdletBinding()]
param(
    [switch]$Once,
    [switch]$Check,
    [string]$ConfigPath,
    [string]$TokenPath,
    [string]$LogFile
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if (-not $ConfigPath) {
    $ConfigPath = Join-Path $repoRoot ".local-secrets\runner-config.json"
}
if (-not $TokenPath) {
    $TokenPath = Join-Path $repoRoot ".local-secrets\runner-token.clixml"
}

if (-not (Test-Path -LiteralPath $ConfigPath)) {
    throw "Runner config does not exist: $ConfigPath"
}
if (-not (Test-Path -LiteralPath $TokenPath)) {
    throw "Protected runner token does not exist: $TokenPath"
}

$config = Get-Content -LiteralPath $ConfigPath -Raw -Encoding UTF8 | ConvertFrom-Json
$python = [string]$config.python
if (-not $python -or -not (Test-Path -LiteralPath $python)) {
    throw "Configured Python executable does not exist: $python"
}
if (-not $config.environment) {
    throw "Runner config must contain an environment object."
}

foreach ($property in $config.environment.PSObject.Properties) {
    [Environment]::SetEnvironmentVariable($property.Name, [string]$property.Value, "Process")
}

$secureToken = Import-Clixml -LiteralPath $TokenPath
$tokenPointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureToken)
$previousToken = [Environment]::GetEnvironmentVariable("RUNNER_TOKEN", "Process")
$runnerArguments = @("-m", "backend.runner_agent")
if ($Once) {
    $runnerArguments += "--once"
}
if ($Check) {
    $runnerArguments += "--check"
}

try {
    $env:RUNNER_TOKEN = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($tokenPointer)
    Push-Location $repoRoot
    try {
        $supervise = -not ($Once -or $Check)
        do {
            if ($LogFile) {
                $logDirectory = Split-Path -Parent $LogFile
                if ($logDirectory) {
                    New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null
                }
                Add-Content -LiteralPath $LogFile -Encoding UTF8 -Value "[$([DateTime]::Now.ToString('s'))] [launcher] starting Runner agent"
                & $python @runnerArguments *>> $LogFile
            }
            else {
                & $python @runnerArguments
            }
            $exitCode = $LASTEXITCODE
            if (-not $supervise) {
                exit $exitCode
            }
            if ($LogFile) {
                Add-Content -LiteralPath $LogFile -Encoding UTF8 -Value "[$([DateTime]::Now.ToString('s'))] [launcher] Runner agent exited with code $exitCode; restarting in 5 seconds"
            }
            Start-Sleep -Seconds 5
        } while ($true)
    }
    finally {
        Pop-Location
    }
}
finally {
    [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($tokenPointer)
    if ($null -eq $previousToken) {
        Remove-Item Env:RUNNER_TOKEN -ErrorAction SilentlyContinue
    }
    else {
        $env:RUNNER_TOKEN = $previousToken
    }
}
