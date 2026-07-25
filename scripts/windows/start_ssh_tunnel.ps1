[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Remote,
    [int]$SshPort = 22,
    [string]$IdentityFile,
    [int]$LocalLearnerPort = 5600,
    [int]$LocalRegistryPort = 5601,
    [int]$RemoteLearnerPort = 5600,
    [int]$RemoteRegistryPort = 5601,
    [switch]$DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ($Remote.StartsWith("-") -or $Remote -match "\s") {
    throw "-Remote must be an SSH host alias or user@host without whitespace."
}
foreach ($port in @(
    $SshPort,
    $LocalLearnerPort,
    $LocalRegistryPort,
    $RemoteLearnerPort,
    $RemoteRegistryPort
)) {
    if ($port -lt 1 -or $port -gt 65535) {
        throw "All SSH and forwarding ports must be in [1, 65535]."
    }
}
if ($LocalLearnerPort -eq $LocalRegistryPort) {
    throw "Local learner and registry ports must be different."
}

$Ssh = Get-Command "ssh" -ErrorAction SilentlyContinue
if ($null -eq $Ssh) {
    throw "Windows OpenSSH Client is not installed or ssh.exe is not on PATH."
}
$SshCommand = if ($Ssh.CommandType -eq "Application") {
    $Ssh.Source
} else {
    $Ssh.Name
}

$SshArgs = @(
    "-N",
    "-T",
    "-p", "$SshPort",
    "-o", "ExitOnForwardFailure=yes",
    "-o", "ServerAliveInterval=15",
    "-o", "ServerAliveCountMax=3",
    "-L", "127.0.0.1:${LocalLearnerPort}:127.0.0.1:${RemoteLearnerPort}",
    "-L", "127.0.0.1:${LocalRegistryPort}:127.0.0.1:${RemoteRegistryPort}"
)
if (-not [string]::IsNullOrWhiteSpace($IdentityFile)) {
    if (-not (Test-Path $IdentityFile -PathType Leaf)) {
        throw "SSH identity file does not exist: $IdentityFile"
    }
    $SshArgs += @("-i", (Resolve-Path $IdentityFile).Path)
}
$SshArgs += $Remote

if ($DryRun) {
    [ordered]@{
        executable = $SshCommand
        remote = $Remote
        learner_forward = "127.0.0.1:${LocalLearnerPort} -> 127.0.0.1:${RemoteLearnerPort}"
        registry_forward = "127.0.0.1:${LocalRegistryPort} -> 127.0.0.1:${RemoteRegistryPort}"
        note = "The real-time Hollow Knight action loop is not forwarded."
    } | ConvertTo-Json
    exit 0
}

Write-Host (
    "SSH tunnel active in this foreground window. " +
    "Press Ctrl+C to stop both rollout/checkpoint forwards."
)
& $SshCommand @SshArgs
exit $LASTEXITCODE
