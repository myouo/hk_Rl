[CmdletBinding()]
param(
    [string]$HollowKnightRoot,
    [switch]$InstallPythonEnvironment,
    [switch]$InstallModBuildEnvironment,
    [switch]$BuildAndInstallMod
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path

function Require-Command {
    param([Parameter(Mandatory = $true)][string]$Name)

    $command = Get-Command $Name -ErrorAction SilentlyContinue
    if ($null -eq $command) {
        throw "Required command '$Name' was not found on PATH."
    }
    return $command.Name
}

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)][string]$Command,
        [Parameter(Mandatory = $true)][string[]]$Arguments
    )

    & $Command @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "'$Command' exited with code $LASTEXITCODE."
    }
}

function Resolve-GameRoot {
    param([string]$RequestedRoot)

    $candidates = New-Object System.Collections.Generic.List[string]
    if (-not [string]::IsNullOrWhiteSpace($RequestedRoot)) {
        $candidates.Add($RequestedRoot)
    }
    if (-not [string]::IsNullOrWhiteSpace(${env:ProgramFiles(x86)})) {
        $candidates.Add(
            (Join-Path ${env:ProgramFiles(x86)} "Steam\steamapps\common\Hollow Knight")
        )
    }
    if (-not [string]::IsNullOrWhiteSpace($env:ProgramFiles)) {
        $candidates.Add(
            (Join-Path $env:ProgramFiles "Steam\steamapps\common\Hollow Knight")
        )
    }

    foreach ($candidate in $candidates) {
        $managed = Join-Path $candidate "hollow_knight_Data\Managed"
        if (Test-Path (Join-Path $managed "Assembly-CSharp.dll") -PathType Leaf) {
            return (Resolve-Path $candidate).Path
        }
    }

    throw (
        "Hollow Knight was not found. Pass -HollowKnightRoot " +
        "'D:\SteamLibrary\steamapps\common\Hollow Knight'."
    )
}

$ResolvedGameRoot = Resolve-GameRoot $HollowKnightRoot
$ManagedDir = Join-Path $ResolvedGameRoot "hollow_knight_Data\Managed"
$GameExe = Join-Path $ResolvedGameRoot "hollow_knight.exe"

$RequiredAssemblies = @(
    "Assembly-CSharp.dll",
    "UnityEngine.dll",
    "UnityEngine.CoreModule.dll",
    "UnityEngine.IMGUIModule.dll",
    "UnityEngine.Physics2DModule.dll",
    "MMHOOK_Assembly-CSharp.dll",
    "unityscenerepacker.dll",
    "PlayMaker.dll"
)
foreach ($assembly in $RequiredAssemblies) {
    $path = Join-Path $ManagedDir $assembly
    if (-not (Test-Path $path -PathType Leaf)) {
        throw (
            "Missing '$path'. Install/enable the Hollow Knight Modding API " +
            "(Lumafly is recommended on current Windows builds), then retry."
        )
    }
}
if (-not (Test-Path $GameExe -PathType Leaf)) {
    throw "Missing Hollow Knight executable: $GameExe"
}

Push-Location $RepoRoot
try {
$SshPath = Require-Command "ssh"
$CondaPath = Require-Command "conda"
$DotnetPath = $null
if ($BuildAndInstallMod) {
    $DotnetPath = Require-Command "dotnet"
}

if ($InstallPythonEnvironment) {
    Invoke-Checked -Command $CondaPath -Arguments @(
        "env", "update",
        "--name", "hkrl",
        "--file", (Join-Path $RepoRoot "environment.yml"),
        "--prune"
    )
}

if ($InstallModBuildEnvironment -or $BuildAndInstallMod) {
    Invoke-Checked -Command $CondaPath -Arguments @(
        "env", "update",
        "--name", "hkrl-mod-build",
        "--file", (Join-Path $RepoRoot "environment-mod-build.yml"),
        "--prune"
    )
}

$InstalledModDir = $null
if ($BuildAndInstallMod) {
    $SchemaOutput = Join-Path $RepoRoot "mod\HKRLEnvMod\Schema"
    New-Item -ItemType Directory -Force -Path $SchemaOutput | Out-Null
    Invoke-Checked -Command $CondaPath -Arguments @(
        "run", "--name", "hkrl-mod-build",
        "flatc", "--csharp",
        "-o", $SchemaOutput,
        (Join-Path $RepoRoot "schema\hkrl.fbs")
    )

    Invoke-Checked -Command $DotnetPath -Arguments @(
        "build",
        (Join-Path $RepoRoot "mod\HKRLEnvMod\HKRLEnvMod.csproj"),
        "-c", "Release",
        "-p:HollowKnightManaged=$ManagedDir",
        "-p:TreatWarningsAsErrors=true"
    )

    $InstalledModDir = Join-Path $ManagedDir "Mods\HKRLEnvMod"
    New-Item -ItemType Directory -Force -Path $InstalledModDir | Out-Null
    $BuildOutput = Join-Path $RepoRoot "mod\HKRLEnvMod\bin\Release"
    $Timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
    foreach ($fileName in @("HKRLEnvMod.dll", "Google.FlatBuffers.dll")) {
        $source = Join-Path $BuildOutput $fileName
        $destination = Join-Path $InstalledModDir $fileName
        if (-not (Test-Path $source -PathType Leaf)) {
            throw "Expected build output was not found: $source"
        }
        if (Test-Path $destination -PathType Leaf) {
            Copy-Item $destination "$destination.bak-$Timestamp" -Force
        }
        Copy-Item $source $destination -Force
    }
}

$PythonCheck = & $CondaPath run --name hkrl python -c (
    "import sys; print('.'.join(map(str, sys.version_info[:3])))"
) 2>$null
$PythonReady = $LASTEXITCODE -eq 0

[ordered]@{
    repo_root = $RepoRoot
    hollow_knight_root = $ResolvedGameRoot
    managed_dir = $ManagedDir
    game_executable = $GameExe
    modding_api_detected = $true
    openssh_client = $SshPath
    conda = $CondaPath
    python_env_ready = $PythonReady
    python_version = if ($PythonReady) { "$PythonCheck".Trim() } else { $null }
    mod_installed_to = $InstalledModDir
    next = (
        "Set HKRL_AUTH_TOKEN, start the remote learner stack, then run " +
        "start_ssh_tunnel.ps1 and start_game_worker.ps1."
    )
} | ConvertTo-Json -Depth 3
} finally {
    Pop-Location
}
