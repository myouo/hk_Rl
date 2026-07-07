# HK-RL — Hollow Knight Reinforcement Learning

[![CI](https://github.com/myouo/hk_Rl/actions/workflows/ci.yml/badge.svg)](https://github.com/myouo/hk_Rl/actions/workflows/ci.yml)

> 从 Hollow Knight Mod 获取结构化游戏状态，构建可扩展、高性能的强化学习环境，
> 训练小骑士在 Godhome / Hall of Gods 中击败 Boss。

本仓库是 [`hollow_knight_rl_prd.md`](./hollow_knight_rl_prd.md) 的工程落地。**先读 PRD，再读 [`AGENTS.md`](./AGENTS.md)。**

---

## 一句话定位

> 一个 **game-state agent**（读取结构化游戏内部状态，非视觉人类 agent），
> 通过 *Mod-as-Environment-Server* + *本地推理 / 远程训练* + *entity-attention + recurrent* 模型，
> 在单 Boss → 多敌方单位 → 线性多 Boss 流程上逐级提升。

## 架构 60 秒概览

```text
Hollow Knight + HKRLEnvMod (C#)        Game PC                  Remote GPU
  ├─ Observation 采集                   ┌──────────────┐         ┌──────────────┐
  ├─ Action 应用 (FixedUpdate)   <───>  │ GameWorker   │ ──────> │ Learner      │
  ├─ Reward events 上报                 │ 本地推理      │ rollout │ PPO/APPO     │
  └─ Clean Episode Lifecycle           │ Gym Env      │ <────── │ checkpoint   │
       │ FlatBuffers over TCP          └──────────────┘ weights └──────────────┘
       └────────────────────────────────────┘
```

**六大不变量**（详见 [`docs/architecture.md`](./docs/architecture.md)）：

1. 实时 action loop 永不跨远程网络。
2. `schema/hkrl.fbs` 是 observation/action/protocol 的唯一真相源。
3. Transport 可插拔（当前 live mod 使用 TCP；shared-memory 是显式 opt-in 原型）。
4. Config 驱动 + 组件注册表。
5. Clean Episode Lifecycle，杜绝 reset 污染。
6. 模型 encoder/attention/memory/heads 解耦，全程 entity_mask。

## 关键技术决策（ADR）

| 决策 | 选型 | ADR |
|---|---|---|
| RL 框架 | 自研 PyTorch PPO | [`docs/adr/0001`](./docs/adr/0001-rl-framework-pytorch.md) |
| 序列化 | FlatBuffers 单一真相源 | [`docs/adr/0002`](./docs/adr/0002-serialization-flatbuffers.md) |
| Mod 框架 | HK Modding API | [`docs/adr/0003`](./docs/adr/0003-mod-framework-hk-modding-api.md) |
| 推理/训练 | 本地推理 + 远程训练解耦 | [`docs/adr/0004`](./docs/adr/0004-local-inference-remote-training.md) |

## 目录导航

```text
schema/    ★ FlatBuffers 单一真相源（先改这里，再 codegen）
mod/       Hollow Knight C# Mod（HKRLEnvMod，Environment Server）
python/    hkrl Python 包（env / models / training / worker / learner / ...）
configs/   task 与 train YAML 配置
docs/      规范文档与 ADR
scripts/   codegen / train / worker / learner / eval 入口
```

## 快速开始

```bash
# 1. 创建并启用 Conda 开发环境
conda env create -f environment.yml
conda env create -f environment-mod-build.yml   # C# schema/mod build 使用 flatc 23.5.26
conda activate hkrl

# 2. 生成 schema 绑定并运行本地质量门禁
make check

# 3. 在游戏机启动 Hollow Knight + HKRLEnvMod
# 可选：覆盖 mod TCP 监听地址；多实例评估/worker 时给每个游戏实例不同端口
export HKRL_HOST=127.0.0.1
export HKRL_PORT=5555
# 可选：设置后 mod 会启用 TCP auth；Python env client 会自动发送同名 token
export HKRL_AUTH_TOKEN=dev-secret

# 4. 先做轻量接入检查：只 PING mod，不重置场景
python scripts/check_env.py \
  --config configs/train/ppo_mlp.yaml \
  --task configs/tasks/gruz_mother.yaml \
  --host 127.0.0.1 \
  --port 5555

# 5. 本地 smoke（需要 Hollow Knight + HKRLEnvMod 正在监听 TCP）
python scripts/train.py \
  --config configs/train/ppo_mlp.yaml \
  --task configs/tasks/gruz_mother.yaml \
  --smoke \
  --host 127.0.0.1 \
  --port 5555 \
  --steps 100 \
  --metrics runs/smoke.jsonl

# 6. 固定 seed 评估（同样需要本地 HKRLEnvMod TCP）
python scripts/run_eval.py \
  --policy scripted \
  --tasks configs/tasks/gruz_mother.yaml \
  --episodes 5 \
  --seeds 0 1 2 \
  --eval-workers 1 \
  --ports 5555 \
  --replay-jsonl runs/eval-replay.jsonl \
  --output runs/eval.json

# 评估本地训练产物时可直接指定 registry 目录，自动加载 latest checkpoint
python scripts/run_eval.py \
  --policy mlp \
  --checkpoint-dir checkpoints \
  --tasks configs/tasks/gruz_mother.yaml \
  --episodes 5 \
  --eval-workers 1 \
  --ports 5555

# 评估 config 注册的模型（如 attention+GRU recurrent checkpoint）
python scripts/run_eval.py \
  --policy model \
  --train-config configs/train/ppo_attention_gru.yaml \
  --checkpoint-dir checkpoints_gru \
  --tasks configs/tasks/gruz_mother.yaml \
  --episodes 5 \
  --eval-workers 1 \
  --ports 5555

# 7. 本地 PPO/RecurrentPPO 训练（需要本地 HKRLEnvMod TCP）
python scripts/train.py \
  --config configs/train/ppo_mlp.yaml \
  --task configs/tasks/gruz_mother.yaml \
  --host 127.0.0.1 \
  --port 5555 \
  --updates 1 \
  --metrics runs/train.jsonl \
  --checkpoint-dir checkpoints

# attention+GRU recurrent PPO 使用 sequence/burn-in buffer
python scripts/train.py \
  --config configs/train/ppo_attention_gru.yaml \
  --task configs/tasks/gruz_mother.yaml \
  --host 127.0.0.1 \
  --port 5555 \
  --updates 1 \
  --metrics runs/train_gru.jsonl \
  --checkpoint-dir checkpoints_gru

# 指标也可写 CSV
python scripts/train.py \
  --config configs/train/ppo_mlp.yaml \
  --task configs/tasks/gruz_mother.yaml \
  --smoke \
  --host 127.0.0.1 \
  --port 5555 \
  --metrics runs/smoke.csv \
  --metrics-kind csv

# 8. 分布式入口 dry-run（不连接真实游戏，用于验证配置/任务/worker 编排）
make phase8-smoke
make phase8-dashboard   # writes runs/phase8-smoke/dashboard.html + dashboard.json
make phase8-profile     # writes runs/phase8-smoke/profile.md + profile.json
make phase8-release-checklist
make phase8-release-evidence
make phase8-verify-release-evidence

# 等价的显式入口会串联 learner/worker/coordinator 的离线 wiring 检查
python scripts/run_phase8_smoke.py \
  --config configs/train/remote_learner.yaml \
  --tasks configs/tasks/gruz_mother.yaml configs/tasks/hornet_protector.yaml

python scripts/render_phase8_dashboard.py \
  --summary runs/phase8-smoke/summary.json \
  --output-html runs/phase8-smoke/dashboard.html \
  --output-json runs/phase8-smoke/dashboard.json

python scripts/render_profile_report.py \
  --summary runs/phase8-smoke/summary.json \
  --output-json runs/phase8-smoke/profile.json \
  --output-md runs/phase8-smoke/profile.md

python scripts/render_release_checklist.py \
  --version phase8 \
  --output-json runs/release/checklist.json \
  --output-md runs/release/checklist.md

python scripts/render_release_evidence.py \
  --version phase8 \
  --output-json runs/release/evidence.json \
  --output-md runs/release/evidence.md

python scripts/verify_release_evidence.py \
  --manifest runs/release/evidence.json \
  --output-json runs/release/evidence-verification.json

python scripts/run_coordinator.py \
  --config configs/train/remote_learner.yaml \
  --tasks configs/tasks/gruz_mother.yaml configs/tasks/hornet_protector.yaml \
  --eval-metrics runs/eval.json \
  --heartbeat-jsonl runs/worker-heartbeats.jsonl \
  --dry-run

python scripts/run_learner.py \
  --config configs/train/remote_learner.yaml \
  --tasks configs/tasks/gruz_mother.yaml configs/tasks/hornet_protector.yaml \
  --checkpoint-dir checkpoints

# learner 启动时会立即发布或恢复 registry latest checkpoint；
# 先启动/恢复 learner，再把该 checkpoint registry 暴露给 worker。

# 远程 batch intake smoke：learner 接收 1 个 TCP rollout batch 后更新一次
export HKRL_AUTH_TOKEN=dev-secret
python scripts/run_learner.py \
  --config configs/train/remote_learner.yaml \
  --tasks configs/tasks/gruz_mother.yaml \
  --bind 127.0.0.1:5600 \
  --intake-count 1 \
  --checkpoint-dir checkpoints

# 长跑 learner 使用 --serve-forever 持续接收 rollout，并在 accepted batch 后更新/发布 checkpoint
python scripts/run_learner.py \
  --config configs/train/remote_learner.yaml \
  --tasks configs/tasks/gruz_mother.yaml \
  --bind 127.0.0.1:5600 \
  --serve-forever \
  --checkpoint-dir checkpoints

# worker 在本地推理/采样，rollout 满后上传到 learner；也可同时写 --batch-dir 作为本地 spool
# checkpoint registry 可用只读 HTTP(S) 暴露；本地 smoke 可先运行：
python -m http.server 8000 --directory checkpoints

python scripts/run_worker.py \
  --config configs/train/remote_learner.yaml \
  --task configs/tasks/gruz_mother.yaml \
  --env-host 127.0.0.1 \
  --env-port 5555 \
  --learner 127.0.0.1:5600 \
  --registry http://127.0.0.1:8000/ \
  --heartbeat-jsonl runs/worker-heartbeats.jsonl \
  --steps 2048
```

`--checkpoint-dir` 会写入 `CheckpointRegistry` 格式的 `index.jsonl` 与
`checkpoint_v*.pt`，包含 `policy_version`、step、sha256 等元数据。registry
中的 checkpoint 路径是相对路径。learner 在空 registry 启动时会发布初始
policy version 0 checkpoint；重启时会加载 latest checkpoint 并继续使用其中的
policy_version。可用本地路径、`file://` 或 HTTP(S) 目录提供给 worker 的
`--registry`，worker 会在首个 rollout 前加载 latest 并验证 sha256。

发布前检查见 [`docs/release.md`](./docs/release.md)，其中区分了 CI 可验证的
Python/离线分布式门禁和必须在 Hollow Knight 机器上执行的 mod/live smoke 门禁；
`make phase8-release-evidence` 会生成包含 sha256 的发布证据 manifest 并立即校验。

## CI

GitHub Actions 在 `push` / `pull_request` 到 `main` 时使用
[`environment.yml`](./environment.yml) 创建 `hkrl` Conda 环境，并执行：
`make check`。该目标会先运行 `make gen-schema`，再执行
`make format-check`、`make lint`、`make typecheck`、`make test`。
Python bindings 使用 `environment.yml` 中的当前 `flatc` 生成；C# bindings
使用 `environment-mod-build.yml` 中固定为 23.5.26 的 `flatc` 生成，以匹配 C#
mod 的 `Google.FlatBuffers` runtime，避免 Python 检查生成出无法被 HKRLEnvMod
编译的 C# bindings。`make check` 会自动发现 `hkrl-mod-build` 环境；也可以显式传
`FLATC_CS=/path/to/flatc-23.5.26`。

云端 CI 包含 Python 包检查和 C# mod 编译检查。C# workflow 会生成 FlatBuffers
绑定，并用 `mod/ci-stubs/` 里的最小 Hollow Knight / Unity / Modding API stub
assemblies 编译 `HKRLEnvMod`，用于持续捕获 C# 语法、项目引用和 schema 绑定问题。
真实游戏程序集兼容性仍需按 [`docs/mod_dev.md`](./docs/mod_dev.md) 在配置
Hollow Knight Managed assemblies 与 HK Modding API 的机器上验证。

## Git Hooks

本仓库提供可版本化的 pre-commit hook。首次 clone 后执行：

```bash
make install-hooks
```

之后每次 `git commit` 前会运行 `make check`，也就是先生成 FlatBuffers
bindings，再执行格式检查、lint、typecheck 和 tests。如果本机有 `hkrl` Conda
环境，hook 会自动通过 `conda run -n hkrl` 执行检查；如果同时存在
`hkrl-mod-build` 环境，`make check` 会自动使用其中固定为 23.5.26 的 `flatc`
生成 C# schema bindings，避免与 mod runtime 版本漂移。

> ⚠️ Mod 的 C# 编译需 Hollow Knight 程序集与 HK Modding API，本机通常不具备，详见 [`docs/mod_dev.md`](./docs/mod_dev.md)。
> 需要真实游戏进程的 smoke / eval / training 命令必须在运行 HKRLEnvMod 的机器上执行。

## 当前状态

Roadmap 已推进到 **Phase 8**：本地 Gym env、step/reset lifecycle、TCP live
transport、reward events、action mask、entity observation、PPO/RecurrentPPO/APPO、
worker/learner/checkpoint/coordinator/evaluator 等核心路径均已落地并由
`make check` 覆盖。

本仓库当前仍有三个需要真实外部环境或后续实现验证的边界：HKRLEnvMod 的 C# 编译
依赖本机 Hollow Knight Managed 程序集与 HK Modding API；端到端
smoke/eval/training 依赖运行中的 Hollow Knight + HKRLEnvMod TCP 服务；
shared-memory transport 目前是 Python 进程内原型，不是可接入当前 mod 的 live
transport。

## License

见 [`LICENSE`](./LICENSE)。
