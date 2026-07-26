# Windows Game PC + SSH Remote Learner

This is the retained Windows compatibility deployment. The primary production
path is now [`linux_ssh_deployment.md`](./linux_ssh_deployment.md). On Windows,
the desktop owns
the game and every latency-sensitive operation; SSH carries only completed
rollouts and versioned checkpoints.

```text
Windows Game PC
  Hollow Knight + HKRLEnvMod
       ▲  loopback TCP :5555 (obs/action/reset)
       ▼
  GameWorker + local policy inference
       │
       ├─ rollout batches ──> 127.0.0.1:5600 ─┐
       └─ checkpoints <──── 127.0.0.1:5601 ─┐ │
                                             SSH local forwards
Remote GPU                                   │ │
  127.0.0.1:5600  authenticated batch intake┘ │
  127.0.0.1:5601  authenticated registry <────┘
  APPO learner + checkpoint storage
```

SSH is deployment/transport infrastructure. It is deliberately absent from
`observation -> local policy -> action -> game`; losing the tunnel may delay an
upload or weight refresh, but must not inject network jitter into game control.

## 1. Before running the scripts

Windows Game PC:

- Install Hollow Knight and a current compatible HK Modding API (Lumafly is the
  normal Windows installer/manager).
- Have a playable save with Godhome / Hall of Gods available.
- Install Miniconda, .NET SDK, Git, and Windows OpenSSH Client.
- Clone this repository locally.

Remote GPU:

- Clone the same repository revision.
- Create a Python 3.10 environment.
- Install the CUDA-enabled PyTorch build appropriate for that GPU/driver first,
  then install this package with
  `pip install -e "python[dev,logging,distributed]"`.
- Configure SSH public-key login from the Windows machine.

Use one high-entropy `HKRL_AUTH_TOKEN` on both machines. Do not put it in YAML,
Git, command-line arguments, or logs.

## 2. Prepare and install the Windows side

Run PowerShell from the repository root:

```powershell
Set-ExecutionPolicy -Scope Process Bypass

.\scripts\windows\prepare_game_pc.ps1 `
  -HollowKnightRoot "D:\SteamLibrary\steamapps\common\Hollow Knight" `
  -InstallPythonEnvironment `
  -InstallModBuildEnvironment `
  -BuildAndInstallMod
```

The script validates the game/Modding API assemblies, creates or updates the
`hkrl` and `hkrl-mod-build` Conda environments, generates C# FlatBuffers
bindings with `flatc 23.5.26`, builds against the **real installed game
assemblies**, and installs these files:

```text
hollow_knight_Data\Managed\Mods\HKRLEnvMod\HKRLEnvMod.dll
hollow_knight_Data\Managed\Mods\HKRLEnvMod\Google.FlatBuffers.dll
```

Existing DLLs are backed up with a timestamp before replacement. Close Hollow
Knight before rebuilding, otherwise Windows may lock the loaded DLL.

To perform checks only, omit the three switches:

```powershell
.\scripts\windows\prepare_game_pc.ps1 `
  -HollowKnightRoot "D:\SteamLibrary\steamapps\common\Hollow Knight"
```

## 3. Start the remote learner stack

On the remote GPU, from the repository root:

```bash
scripts/remote/bootstrap_learner_env.sh
export HKRL_AUTH_TOKEN='replace-with-the-shared-random-token'
export HKRL_PYTHON_BIN=/path/to/hkrl-learner/bin/python

scripts/remote/start_learner_stack.sh
```

For an operator-friendly walkthrough, open
[`notebooks/remote_gpu_training.ipynb`](../notebooks/remote_gpu_training.ipynb)
with the `Python (hkrl-learner)` kernel. The notebook defaults to a non-mutating
inspection mode and contains the service start/stop, monitoring, resume, and
Windows evaluation handoff cells.
For a blank training host, upload
[`notebooks/one_click_clone_setup.ipynb`](../notebooks/one_click_clone_setup.ipynb)
first. Its execute mode clones the repository without deleting or overwriting
existing paths. On a regular training host it calls the same GPU bootstrap,
registers the kernel, creates a permission-restricted token file, and validates
the composed SSH learner config. On Kaggle it instead reuses the current
Python/PyTorch environment, skips the SSH token, and runs only the offline Phase
8 smoke; it is not a replacement for the stable SSH-reachable learner host used
by the live Windows deployment.

The launcher starts:

- APPO rollout intake on `127.0.0.1:5600`;
- read-only checkpoint registry HTTP on `127.0.0.1:5601`;
- three compatible task layouts by default.

Both services remain loopback-only. The registry serves only `index.jsonl` and
`checkpoint_v*.pt`, requires a Bearer token, and the worker still verifies every
checkpoint SHA-256 before loading it.

To train a smaller task set:

```bash
scripts/remote/start_learner_stack.sh \
  configs/tasks/gruz_mother.yaml \
  configs/tasks/hornet_protector.yaml
```

The remote role uses
[`configs/train/ssh_remote_learner.yaml`](../configs/train/ssh_remote_learner.yaml).
It requires `learner.device: cuda`, so an accidentally CPU-only PyTorch
installation fails at startup instead of silently running a long CPU training
job.

## 4. Open the SSH forwards from Windows

In PowerShell terminal A:

```powershell
.\scripts\windows\start_ssh_tunnel.ps1 `
  -Remote "gpu-user@gpu-host" `
  -IdentityFile "$HOME\.ssh\id_ed25519"
```

This terminal intentionally stays in the foreground. It forwards:

```text
Windows 127.0.0.1:5600 -> remote 127.0.0.1:5600  rollout batches
Windows 127.0.0.1:5601 -> remote 127.0.0.1:5601  checkpoints
```

Use `-DryRun` to inspect the resolved topology without connecting. The first
real SSH connection may ask you to verify the remote host key.

## 5. Launch the game and GameWorker

In PowerShell terminal B, set the same token. If
`-HollowKnightExe` is supplied, the script launches the game so the mod inherits
the token and loopback listener settings:

```powershell
$env:HKRL_AUTH_TOKEN = "replace-with-the-shared-random-token"

.\scripts\windows\start_game_worker.ps1 `
  -HollowKnightExe "D:\SteamLibrary\steamapps\common\Hollow Knight\hollow_knight.exe" `
  -Task "configs/tasks/gruz_mother.yaml" `
  -WorkerId "windows-game-0" `
  -Steps 2048
```

Omit `-Steps` (or use `0`) for a continuous worker. For round-robin tasks:

```powershell
.\scripts\windows\start_game_worker.ps1 `
  -Task "configs/tasks/gruz_mother.yaml" `
  -Tasks @(
    "configs/tasks/gruz_mother.yaml",
    "configs/tasks/hornet_protector.yaml",
    "configs/tasks/mantis_lords.yaml"
  ) `
  -WorkerId "windows-game-0"
```

The worker launcher first:

1. fetches the checkpoint index through the SSH tunnel;
2. validates config/model/task layout with `--dry-run`;
3. sends a live `PING` to the local mod;
4. starts local inference and rollout collection.

It uses
[`configs/train/windows_game_worker.yaml`](../configs/train/windows_game_worker.yaml).
Rollouts are also spooled under `runs/windows-worker/batches` and heartbeats
under `runs/windows-worker/heartbeats.jsonl` for recovery evidence.

If Hollow Knight is already open, it must have inherited `HKRL_AUTH_TOKEN`,
`HKRL_HOST=127.0.0.1`, and the matching `HKRL_PORT` when it started. Restart it
through the script if unsure.

## 6. Live acceptance gates

CI stubs and offline smoke do not prove game compatibility. Before a long run,
verify all of these on Windows:

```text
[ ] Real-assembly mod build succeeds with 0 errors.
[ ] Mod log says "In-mod PlayerAction input injection is active."
[ ] scripts/check_env.py reports ok=true and the expected schema version.
[ ] A short scripted/random rollout visibly moves, jumps, attacks, casts,
    focuses, and releases nail-art input without keyboard/gamepad assistance.
[ ] RESET reaches RUNNING repeatedly without stale input or reward events.
[ ] Remote learner accepts a batch and publishes a newer checkpoint.
[ ] Windows worker downloads it and reports the newer policy/checkpoint version.
[ ] Fixed-seed evaluator produces per-boss win rate, damage taken, TTK, and
    invalid-action ratio; training reward alone is not accepted as performance.
```

Keep manual keyboard/gamepad controls idle during an agent rollout. On any input
binding failure, stop the run and capture the Modding API log; the injector
reports a single compatibility error instead of throwing through the game
update loop.

## 7. What must be supplied for an actual remote deployment

The repository-side preparation is complete, but a real SSH deployment needs
machine-specific values that must not be guessed:

- Windows Hollow Knight installation path;
- confirmation that Modding API and a Godhome-capable save work;
- remote SSH target (`user@host` or SSH alias) and key path;
- remote repository path and Python environment path;
- remote GPU/driver/CUDA details for the correct PyTorch build.

No public learner or registry port is required; keep the remote firewall closed
and use the loopback SSH forwards above.
