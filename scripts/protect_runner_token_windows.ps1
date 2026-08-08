[CmdletBinding(DefaultParameterSetName = "Prompt")]
param(
    [Parameter(ParameterSetName = "Prompt")]
    [Security.SecureString]$Token,

    [Parameter(Mandatory = $true, ParameterSetName = "File")]
    [string]$PlaintextTokenFile,

    [Parameter(ParameterSetName = "File")]
    [switch]$DeletePlaintextTokenFile,

    [string]$OutputPath
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$secretDirectory = Join-Path $repoRoot ".local-secrets"
if (-not $OutputPath) {
    $OutputPath = Join-Path $secretDirectory "runner-token.clixml"
}

New-Item -ItemType Directory -Path $secretDirectory -Force | Out-Null

if ($PSCmdlet.ParameterSetName -eq "File") {
    $resolvedTokenFile = (Resolve-Path -LiteralPath $PlaintextTokenFile).Path
    try {
        $plaintext = [System.IO.File]::ReadAllText($resolvedTokenFile).Trim()
        if (-not $plaintext) {
            throw "Runner token file is empty."
        }
        $Token = ConvertTo-SecureString $plaintext -AsPlainText -Force
    }
    finally {
        $plaintext = $null
        if ($DeletePlaintextTokenFile -and (Test-Path -LiteralPath $resolvedTokenFile)) {
            Remove-Item -LiteralPath $resolvedTokenFile -Force
        }
    }
}
elseif (-not $Token) {
    $Token = Read-Host "Runner token" -AsSecureString
}

$Token | Export-Clixml -LiteralPath $OutputPath -Force

# DPAPI encrypts the payload; the ACL also limits who can replace or copy the file.
$currentUser = [System.Security.Principal.WindowsIdentity]::GetCurrent().User
$systemUser = New-Object System.Security.Principal.SecurityIdentifier("S-1-5-18")
$acl = New-Object System.Security.AccessControl.FileSecurity
$acl.SetOwner($currentUser)
$acl.SetAccessRuleProtection($true, $false)
$acl.AddAccessRule((New-Object System.Security.AccessControl.FileSystemAccessRule(
    $currentUser,
    [System.Security.AccessControl.FileSystemRights]::FullControl,
    [System.Security.AccessControl.AccessControlType]::Allow
)))
$acl.AddAccessRule((New-Object System.Security.AccessControl.FileSystemAccessRule(
    $systemUser,
    [System.Security.AccessControl.FileSystemRights]::FullControl,
    [System.Security.AccessControl.AccessControlType]::Allow
)))
Set-Acl -LiteralPath $OutputPath -AclObject $acl

Write-Host "Runner token protected for the current Windows user: $OutputPath"
