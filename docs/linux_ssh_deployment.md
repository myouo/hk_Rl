# Linux Game Host + SSH Remote Learner

This is the primary production topology. Linux owns Hollow Knight, the
HKRLEnvMod environment server, the Gym environment, and local policy inference.
The remote GPU receives only complete rollout batches and publishes versioned
checkpoints.

```text
Linux game host
  Hollow Knight (native or Steam/Proton) + HKRLEnvMod
       ▲ 127.0.0.1:5555 FlatBuffers step/reset
       ▼
  GameWorker + local CPU inference
       │
       ├─ 127.0.0.1:5600 ─SSH─> remote learner 127.0.0.1:5600
       └─ 127.0.0.1:5601 <─SSH─ remote registry 127.0.0.1:5601

Remote GPU
  APPO learner + authenticated checkpoint registry
```

The real-time `observation -> policy -> action -> game` path never crosses SSH.
Tunnel loss can delay a rollout upload or checkpoint refresh, but cannot add
remote-network latency to input.

## 1. Supported game version and Modding API

As of July 2026, Hollow Knight's current Steam build is not supported by the
Modding API. Select Steam:

```text
Hollow Knight -> Properties -> Game Versions & Betas
              -> 1.5.78.11833 (Previous version)
```

Wait for the downgrade to finish before installing or launching mods. Lumafly's
[compatibility notice](https://github.com/TheMulhima/Lumafly/issues/239) names
`1.5.78.11833` as the last supported version. Use the cross-platform
[Lumafly](https://github.com/TheMulhima/Lumafly) manager to enable the Modding
API. Do not force the API onto the unsupported current build; it crashes at
startup.

The host OS and game binary are separate compatibility decisions. Inspect the
game root before installing the API:

| Game executable | Runtime | Required API v77 archive |
| --- | --- | --- |
| `hollow_knight.x86_64` | native Linux | `moddingapi.v77.linux.zip` |
| `hollow_knight.exe` | Steam/Proton | `moddingapi.v77.windows.zip` |

Both archives come from the official
[HK Modding API v77 release](https://github.com/hk-modding/api/releases/tag/1.5.78.11833-77).
The Linux-host script rejects a mismatched native library:
`libunityscenerepacker.so` for native Linux,
`unityscenerepacker.dll` for Proton. In particular, a Linux host running
`hollow_knight.exe` must use the Windows archive. The scripts search the default
Steam library and every path in `libraryfolders.vdf`; use `--game-root` for a
nonstandard or non-Steam install.

## 2. Prepare Python and install HKRLEnvMod

From the repository root:

```bash
scripts/linux/prepare_game_pc.sh \
  --install-python-environment \
  --install-mod-build-environment \
  --build-and-install-mod
```

If the `hkrl` and `hkrl-mod-build` environments already exist, do not request
an environment update:

```bash
scripts/linux/prepare_game_pc.sh --build-and-install-mod
```

The script verifies the supported Steam branch and the real Modding API
assemblies, creates/updates the `hkrl` and `hkrl-mod-build` Conda environments,
generates C# FlatBuffers bindings with `flatc 23.5.26`, builds against the real
game assemblies, backs up existing mod DLLs, and installs:

```text
hollow_knight_Data/Managed/Mods/HKRLEnvMod/HKRLEnvMod.dll
hollow_knight_Data/Managed/Mods/HKRLEnvMod/Google.FlatBuffers.dll
```

With no install flags it performs a non-mutating readiness check:

```bash
scripts/linux/prepare_game_pc.sh
```

Close Hollow Knight before replacing DLLs.

## 3. Authentication without Steam launch-option secrets

The Mod normally reads `HKRL_HOST`, `HKRL_PORT`, and `HKRL_AUTH_TOKEN` from its
process environment. An already-running Steam client does not reliably inherit
new shell variables when launching a Proton game. On Linux,
`start_game_worker.sh` therefore atomically writes:

```text
hollow_knight_Data/Managed/Mods/HKRLEnvMod/hkrl-runtime.conf
```

with mode `0600`. The Mod reads this file only as a fallback; environment
variables remain higher priority. Token values are never logged. Do not commit,
copy, or relax permissions on this file.

Create the local worker token file without printing the token:

```bash
install -d -m 700 "$HOME/.config/hkrl"
umask 077
{
  printf 'HKRL_AUTH_TOKEN='
  ssh -p 30262 root@region-9.autodl.pro \
    "sed -n 's/^HKRL_AUTH_TOKEN=//p' /root/.config/hkrl/learner.env"
} > "$HOME/.config/hkrl/worker.env"
chmod 600 "$HOME/.config/hkrl/worker.env"
```

The file contains one shell assignment and is sourced by the combined launcher.

## 4. Start the complete Linux flow

The remote learner must already be running. Then one foreground command starts
the SSH tunnel, launches the game when necessary, validates the remote registry,
PINGs the live Mod, downloads the initial checkpoint, and starts continuous
local rollout collection:

```bash
scripts/linux/start_training_stack.sh \
  --remote root@region-9.autodl.pro \
  --ssh-port 30262 \
  --identity-file "$HOME/.ssh/id_ed25519" \
  --auth-file "$HOME/.config/hkrl/worker.env" \
  --task configs/tasks/gruz_mother.yaml
```

Use `--steps 2048` for a bounded acceptance run. Use `--no-launch-game` when the
game is already open with the matching runtime configuration.

To inspect the resolved topology without writing config, launching the game, or
opening SSH:

```bash
scripts/linux/start_training_stack.sh \
  --remote root@region-9.autodl.pro \
  --ssh-port 30262 \
  --identity-file "$HOME/.ssh/id_ed25519" \
  --game-root "$HOME/.local/share/Steam/steamapps/common/Hollow Knight" \
  --dry-run
```

The worker uses one PyTorch intra-op thread by default. Batch-size-one local
inference is normally faster and produces less game-frame jitter this way. Tune
with `start_game_worker.sh --inference-threads N` only after measuring SPS.

## 5. Separate-terminal operation

Tunnel terminal:

```bash
scripts/linux/start_ssh_tunnel.sh \
  --remote root@region-9.autodl.pro \
  --ssh-port 30262 \
  --identity-file "$HOME/.ssh/id_ed25519"
```

Game/worker terminal:

```bash
set -a
source "$HOME/.config/hkrl/worker.env"
set +a

scripts/linux/start_game_worker.sh \
  --launch-game \
  --task configs/tasks/gruz_mother.yaml \
  --worker-id linux-game-0
```

Local recovery evidence is written under:

```text
runs/linux-worker/batches/
runs/linux-worker/heartbeats.jsonl
runs/linux-worker/game.log or steam-launch.log
```

## 6. Acceptance gates

Before a long run, verify:

```text
[ ] Steam branch is 1.5.78.11833 and download state is complete.
[ ] Modding API native library matches the executable (Linux native or Proton).
[ ] Real-assembly mod build succeeds with zero warnings/errors.
[ ] Mod log contains "In-mod PlayerAction input injection is active."
[ ] Mod log reports loopback listener and auth=enabled.
[ ] scripts/check_env.py returns ok=true with the expected schema version.
[ ] RESET repeatedly reaches RUNNING without stale inputs/events.
[ ] A bounded worker run uploads a batch to the remote learner.
[ ] Learner policy/checkpoint version increases and the worker downloads it.
[ ] Fixed-seed per-boss evaluation reports win rate, damage, TTK, and invalid actions.
```

The game must contain a Godhome-capable save, and the intended boss must be
available. No offline test can substitute for these live gates.
