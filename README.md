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
       │ FlatBuffers over TCP/SHM      └──────────────┘ weights └──────────────┘
       └────────────────────────────────────┘
```

**六大不变量**（详见 [`docs/architecture.md`](./docs/architecture.md)）：

1. 实时 action loop 永不跨远程网络。
2. `schema/hkrl.fbs` 是 observation/action/protocol 的唯一真相源。
3. Transport 可插拔（TCP / shared-memory）。
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
conda activate hkrl

# 2. 生成 schema 绑定并运行本地质量门禁
make check

# 3. 本地 smoke（需要 Hollow Knight + HKRLEnvMod 正在监听 TCP）
python scripts/train.py \
  --config configs/train/ppo_mlp.yaml \
  --task configs/tasks/gruz_mother.yaml \
  --smoke \
  --steps 100 \
  --metrics runs/smoke.jsonl

# 4. 固定 seed 评估（同样需要本地 HKRLEnvMod TCP）
python scripts/run_eval.py \
  --policy scripted \
  --tasks configs/tasks/gruz_mother.yaml \
  --episodes 5 \
  --seeds 0 1 2 \
  --replay-jsonl runs/eval-replay.jsonl \
  --output runs/eval.json

# 评估本地训练产物时可直接指定 registry 目录，自动加载 latest checkpoint
python scripts/run_eval.py \
  --policy mlp \
  --checkpoint-dir checkpoints \
  --tasks configs/tasks/gruz_mother.yaml \
  --episodes 5

# 评估 config 注册的模型（如 attention+GRU recurrent checkpoint）
python scripts/run_eval.py \
  --policy model \
  --train-config configs/train/ppo_attention_gru.yaml \
  --checkpoint-dir checkpoints_gru \
  --tasks configs/tasks/gruz_mother.yaml \
  --episodes 5

# 5. 本地 PPO/RecurrentPPO 训练（需要本地 HKRLEnvMod TCP）
python scripts/train.py \
  --config configs/train/ppo_mlp.yaml \
  --task configs/tasks/gruz_mother.yaml \
  --updates 1 \
  --metrics runs/train.jsonl \
  --checkpoint-dir checkpoints

# attention+GRU recurrent PPO 使用 sequence/burn-in buffer
python scripts/train.py \
  --config configs/train/ppo_attention_gru.yaml \
  --task configs/tasks/gruz_mother.yaml \
  --updates 1 \
  --metrics runs/train_gru.jsonl \
  --checkpoint-dir checkpoints_gru

# 指标也可写 CSV
python scripts/train.py \
  --config configs/train/ppo_mlp.yaml \
  --task configs/tasks/gruz_mother.yaml \
  --smoke \
  --metrics runs/smoke.csv \
  --metrics-kind csv

# 6. 分布式入口 dry-run（不连接真实游戏，用于验证配置/任务/worker 编排）
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

# 远程 batch intake smoke：learner 接收 1 个 TCP rollout batch 后更新一次
export HKRL_AUTH_TOKEN=dev-secret
python scripts/run_learner.py \
  --config configs/train/remote_learner.yaml \
  --tasks configs/tasks/gruz_mother.yaml \
  --bind 127.0.0.1:5600 \
  --intake-count 1 \
  --checkpoint-dir checkpoints

# worker 在本地推理/采样，rollout 满后上传到 learner；也可同时写 --batch-dir 作为本地 spool
# checkpoint registry 可用只读 HTTP(S) 暴露；本地 smoke 可先运行：
python -m http.server 8000 --directory checkpoints

python scripts/run_worker.py \
  --config configs/train/remote_learner.yaml \
  --task configs/tasks/gruz_mother.yaml \
  --learner 127.0.0.1:5600 \
  --registry http://127.0.0.1:8000/ \
  --heartbeat-jsonl runs/worker-heartbeats.jsonl \
  --steps 2048
```

`--checkpoint-dir` 会写入 `CheckpointRegistry` 格式的 `index.jsonl` 与
`checkpoint_v*.pt`，包含 `policy_version`、step、sha256 等元数据。registry
中的 checkpoint 路径是相对路径，可用本地路径、`file://` 或 HTTP(S) 目录提供给
worker 的 `--registry`，worker 会在加载前验证 sha256。

## CI

GitHub Actions 在 `push` / `pull_request` 到 `main` 时使用
[`environment.yml`](./environment.yml) 创建 `hkrl` Conda 环境，并执行：
`make check`。该目标会先运行 `make gen-schema`，再执行
`make format-check`、`make lint`、`make typecheck`、`make test`。

云端 CI 当前只覆盖 Python 包。C# mod 构建需要本机 Hollow Knight Managed
程序集与 HK Modding API 路径，按 [`docs/mod_dev.md`](./docs/mod_dev.md) 在
配置游戏安装的机器上验证。

## Git Hooks

本仓库提供可版本化的 pre-commit hook。首次 clone 后执行：

```bash
make install-hooks
```

之后每次 `git commit` 前会运行 `make check`，也就是先生成 FlatBuffers
bindings，再执行格式检查、lint、typecheck 和 tests。如果本机有 `hkrl` Conda
环境，hook 会自动通过 `conda run -n hkrl` 执行检查。

> ⚠️ Mod 的 C# 编译需 Hollow Knight 程序集与 HK Modding API，本机通常不具备，详见 [`docs/mod_dev.md`](./docs/mod_dev.md)。
> 需要真实游戏进程的 smoke / eval / training 命令必须在运行 HKRLEnvMod 的机器上执行。

## 当前状态

Roadmap 已推进到 **Phase 8**：本地 Gym env、step/reset lifecycle、TCP/SHM
transport、reward events、action mask、entity observation、PPO/RecurrentPPO/APPO、
worker/learner/checkpoint/coordinator/evaluator 等核心路径均已落地并由
`make check` 覆盖。

本仓库当前仍有两个需要真实外部环境验证的边界：HKRLEnvMod 的 C# 编译依赖本机
Hollow Knight Managed 程序集与 HK Modding API；端到端 smoke/eval/training 依赖
运行中的 Hollow Knight + HKRLEnvMod TCP 服务。

## License

见 [`LICENSE`](./LICENSE)。
