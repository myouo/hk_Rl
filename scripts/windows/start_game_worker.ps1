[CmdletBinding()]
param(
    [string]$Config = "configs/train/windows_game_worker.yaml",
    [string]$Task = "configs/tasks/gruz_mother.yaml",
    [string[]]$Tasks = @(),
    [string]$WorkerId = "windows-game-0",
    [int]$EnvPort = 5555,
    [int]$LearnerPort = 5600,
    [int]$RegistryPort = 5601,
    [int]$Steps = 0,
    [string]$HollowKnightExe,
    [int]$GameStartupTimeoutSeconds = 60,
    [switch]$SkipLiveEnvCheck
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($env:HKRL_AUTH_TOKEN)) {
    throw (
        "HKRL_AUTH_TOKEN is not set. Set it to the same value used by the " +
        "remote learner before launching Hollow Knight and this worker."
    )
}
foreach ($port in @($EnvPort, $LearnerPort, $RegistryPort)) {
    if ($port -lt 1 -or $port -gt 65535) {
        throw "Env, learner, and registry ports must be in [1, 65535]."
    }
}
if ($Steps -lt 0) {
    throw "-Steps must be non-negative; 0 means run continuously."
}
if ($GameStartupTimeoutSeconds -lt 1) {
    throw "-GameStartupTimeoutSeconds must be positive."
}

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Conda = Get-Command "conda" -ErrorAction SilentlyContinue
if ($null -eq $Conda) {
    throw "conda was not found. Run prepare_game_pc.ps1 first."
}
$CondaCommand = if ($Conda.CommandType -eq "Application") {
    $Conda.Source
} else {
    $Conda.Name
}

function Resolve-RepoPath {
    param([Parameter(Mandatory = $true)][string]$Path)

    $candidate = if ([System.IO.Path]::IsPathRooted($Path)) {
        $Path
    } else {
        Join-Path $RepoRoot $Path
    }
    if (-not (Test-Path $candidate -PathType Leaf)) {
        throw "Required file does not exist: $candidate"
    }
    return (Resolve-Path $candidate).Path
}

function Invoke-CondaPython {
    param([Parameter(Mandatory = $true)][string[]]$Arguments)

    & $CondaCommand run --name hkrl python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code $LASTEXITCODE."
    }
}

$ConfigPath = Resolve-RepoPath $Config
$TaskPath = Resolve-RepoPath $Task
$TaskPaths = @()
foreach ($item in $Tasks) {
    $TaskPaths += Resolve-RepoPath $item
}

$env:HKRL_HOST = "127.0.0.1"
$env:HKRL_PORT = "$EnvPort"

if (-not [string]::IsNullOrWhiteSpace($HollowKnightExe)) {
    $ResolvedGameExe = Resolve-RepoPath $HollowKnightExe
    Start-Process -FilePath $ResolvedGameExe -WorkingDirectory (
        Split-Path $ResolvedGameExe -Parent
    ) | Out-Null

    $Deadline = (Get-Date).AddSeconds($GameStartupTimeoutSeconds)
    do {
        Start-Sleep -Milliseconds 500
        $Client = New-Object System.Net.Sockets.TcpClient
        try {
            $Connected = $Client.ConnectAsync("127.0.0.1", $EnvPort).Wait(250)
        } catch {
            $Connected = $false
        } finally {
            $Client.Dispose()
        }
    } while (-not $Connected -and (Get-Date) -lt $Deadline)
    if (-not $Connected) {
        throw "HKRLEnvMod did not listen on 127.0.0.1:$EnvPort before timeout."
    }
}

Push-Location $RepoRoot
try {
    $WorkerBase = @(
        (Join-Path $RepoRoot "scripts\run_worker.py"),
        "--config", $ConfigPath,
        "--task", $TaskPath,
        "--env-host", "127.0.0.1",
        "--env-port", "$EnvPort",
        "--learner", "127.0.0.1:$LearnerPort",
        "--registry", "http://127.0.0.1:$RegistryPort/",
        "--worker-id", $WorkerId,
        "--batch-dir", (Join-Path $RepoRoot "runs\windows-worker\batches"),
        "--heartbeat-jsonl", (Join-Path $RepoRoot "runs\windows-worker\heartbeats.jsonl")
    )
    if ($TaskPaths.Count -gt 0) {
        $WorkerBase += "--tasks"
        $WorkerBase += $TaskPaths
    }

    # This fetches index.jsonl through the SSH tunnel and validates model/layout
    # wiring without touching the live game.
    $DryRunArgs = $WorkerBase + @("--dry-run")
    $DryRunOutput = @(
        & $CondaCommand run --name hkrl python @DryRunArgs
    )
    if ($LASTEXITCODE -ne 0) {
        throw "Worker dry-run failed with exit code $LASTEXITCODE."
    }
    $DryRunOutput | ForEach-Object { Write-Host $_ }
    $DryRunJson = $DryRunOutput |
        Where-Object { -not [string]::IsNullOrWhiteSpace("$_") } |
        Select-Object -Last 1
    if ([string]::IsNullOrWhiteSpace("$DryRunJson")) {
        throw "Worker dry-run produced no JSON summary."
    }
    try {
        $DryRunSummary = $DryRunJson | ConvertFrom-Json
    } catch {
        throw "Worker dry-run did not end with a JSON summary."
    }
    if (
        -not ($DryRunSummary.PSObject.Properties.Name -contains "latest_checkpoint") -or
        [int]$DryRunSummary.latest_checkpoint -lt 1
    ) {
        throw (
            "The remote registry has no startup checkpoint. Start the remote " +
            "learner stack and wait for checkpoint_v000001.pt before collecting."
        )
    }

    if (-not $SkipLiveEnvCheck) {
        Invoke-CondaPython -Arguments @(
            (Join-Path $RepoRoot "scripts\check_env.py"),
            "--config", $ConfigPath,
            "--task", $TaskPath,
            "--host", "127.0.0.1",
            "--port", "$EnvPort"
        )
    }

    $WorkerArgs = $WorkerBase
    if ($Steps -gt 0) {
        $WorkerArgs += @("--steps", "$Steps")
    }
    Invoke-CondaPython -Arguments $WorkerArgs
} finally {
    Pop-Location
}
