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
counts. When `runs/release/evidence.md` exists, the verifier also checks that
its release title, manifest metadata including `manifest_version`, and artifact
table exactly match `evidence.json` without missing, reordered, or extra
artifact rows. The verifier also rejects absolute, non-normalized, or duplicate
artifact paths, non-object artifact entries, missing or malformed full-length
`git_sha` values, mismatched release commit SHAs when `--git-sha` is provided,
unsupported release `version` or `manifest_version` values, and Phase 8 smoke
summaries that do not report `ok=true` with non-negative coordinator
`worker_count`, `active_worker_count`, and `sps` metrics plus learner, worker,
task, checkpoint, worker-id, and worker-row sections. Smoke worker rows must
include an `alive` flag and non-negative worker `sps`/`worker_crash_count`
metrics. Dashboard JSON must include health, learner, metrics, task, and worker
sections with well-formed task and worker rows, while dashboard HTML must have
the HKRL dashboard title, learner/worker/task sections, and the JSON task/worker
rows.
Profile JSON must come from the Phase 8 smoke source and include metrics,
findings, and workers with well-formed finding and worker rows, while profile
Markdown must have the HKRL profile title, worker table, and the JSON worker
rows. Checklist JSON must be a Phase 8 checklist with every required gate,
well-formed check rows, a matching checklist `git_sha`, and a matching blocking
check count; checklist Markdown must have the HKRL release title, matching
`git_sha`, and every required gate ID. It also requires every offline Phase 8
artifact below to be listed in the manifest. Live eval artifacts are included
only when the full live eval group exists locally; if any live eval artifact is
listed, all three live eval artifacts must be listed. When the eval report JSON
is listed, verification also requires a `run_eval` report with well-formed
`summary`, `tasks`, and `findings` sections, task rows with
`task_id`/`metrics_valid`, matching valid/malformed task counts, unique task IDs,
at least one valid task row, and no critical eval findings; eval report Markdown
must have the HKRL eval title, summary/task sections, and the JSON task rows so
hash-valid but failed fixed-seed reports cannot pass as release evidence:

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
and regression deltas when the eval output includes a `regression` section. If a
task omits `win_rate` or reports an invalid value, the report uses
`per_boss_win_rate` as the canonical win-rate fallback. Non-object per-task
metric payloads are rendered as critical findings so malformed eval evidence
does not pass as ordinary zero-valued performance. Summary win-rate metrics are
computed over valid task rows and report separate valid/malformed task counts.
If no valid task metric rows remain, the report emits an additional critical
finding and the live eval evidence should be regenerated before release.
Malformed regression deltas are also critical findings; they are left out of
worst-regression summaries rather than coerced into a non-regression value.

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
  --git-sha "$(git rev-parse HEAD)" \
  --output-json runs/release/evidence-verification.json
```

`make phase8-release-evidence` runs the offline Phase 8 artifact pipeline and
writes the checklist, evidence manifest, and verification report with the current
git SHA.

The checklist is a release record, not an automated certification. The game
machine gates still need to be executed on a configured Hollow Knight host.
