# Release

Phase 8 releases are evidence-driven. A release is not ready until local gates,
offline distributed artifacts, remote CI, and game-machine checks have all been
recorded.

## 1. Local Gates

Run the Python quality gate first:

```bash
make check
```

Then produce the offline Phase 8 artifacts:

```bash
make phase8-smoke
make phase8-dashboard
make phase8-profile
make phase8-release-checklist
make phase8-release-evidence
make phase8-verify-release-evidence
```

The generated files under `runs/` are ignored by git and should be attached to a
release note or CI artifact store when useful. `evidence.json` records the
release artifact paths, byte sizes, and sha256 hashes; `evidence-verification.json`
records the result of re-hashing those files and checking the manifest aggregate
counts. The verifier also rejects absolute, non-normalized, or duplicate
artifact paths, plus unsupported `manifest_version` values. Offline artifacts
are always included in the hash manifest; live eval artifacts are included when
they exist locally:

```text
runs/phase8-smoke/summary.json
runs/phase8-smoke/dashboard.html
runs/phase8-smoke/dashboard.json
runs/phase8-smoke/profile.md
runs/phase8-smoke/profile.json
runs/release/checklist.md
runs/release/checklist.json
runs/eval.json
runs/eval-report.md
runs/eval-report.json
```

The same command also writes `runs/release/evidence.md`,
`runs/release/evidence.json`, and `runs/release/evidence-verification.json` as
the manifest and verification reports.

## 2. Remote CI

After pushing, confirm the latest `main` run is green and matches the release
commit:

```bash
gh run list --branch main --limit 1
```

The CI gate currently covers the Python package and generated FlatBuffers
bindings through `make check`.

## 3. Game Machine Gates

These gates require a machine with Hollow Knight, HKRLEnvMod dependencies, and
the HK Modding API configured:

```bash
dotnet build mod/HKRLEnvMod/HKRLEnvMod.csproj
python scripts/train.py --config configs/train/ppo_mlp.yaml \
  --task configs/tasks/gruz_mother.yaml --smoke
python scripts/run_eval.py --policy scripted \
  --tasks configs/tasks/gruz_mother.yaml --episodes 5 \
  --output runs/eval.json
make phase8-eval-report
```

For multi-instance evaluation, provide one live mod TCP port per intended worker:

```bash
python scripts/run_eval.py --policy scripted \
  --tasks configs/tasks/gruz_mother.yaml configs/tasks/hornet_protector.yaml \
  --episodes 5 --eval-workers 2 --ports 5555 5556 \
  --output runs/eval.json
make phase8-eval-report
```

`phase8-eval-report` turns the fixed-seed evaluator JSON into
`runs/eval-report.json` and `runs/eval-report.md`, including per-task win rates
and regression deltas when the eval output includes a `regression` section.

## 4. Security Review

Before a LAN release, verify:

- `security.bind_scope` is `localhost` or LAN-scoped as intended.
- `security.require_token` is true for non-loopback services.
- `HKRL_AUTH_TOKEN` is configured where required.
- Checkpoint registries remain local, LAN, or authenticated HTTP(S), and workers
  keep sha256 verification enabled.
- No service is intentionally exposed to a public network.

## 5. Checklist Artifact

Use the checklist renderer for a release record:

```bash
python scripts/render_release_checklist.py \
  --version phase8 \
  --git-sha "$(git rev-parse HEAD)" \
  --output-json runs/release/checklist.json \
  --output-md runs/release/checklist.md
```

After all local artifacts are generated, render the hash manifest:

```bash
python scripts/render_release_evidence.py \
  --version phase8 \
  --git-sha "$(git rev-parse HEAD)" \
  --output-json runs/release/evidence.json \
  --output-md runs/release/evidence.md
```

Verify the manifest before attaching artifacts:

```bash
python scripts/verify_release_evidence.py \
  --manifest runs/release/evidence.json \
  --output-json runs/release/evidence-verification.json
```

`make phase8-release-evidence` runs the offline Phase 8 artifact pipeline and
writes the checklist, evidence manifest, and verification report with the current
git SHA.

The checklist is a release record, not an automated certification. The game
machine gates still need to be executed on a configured Hollow Knight host.
