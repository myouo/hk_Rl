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

## 快速开始（占位骨架阶段）

```bash
# 1. 创建并启用 Conda 开发环境
conda env create -f environment.yml
conda activate hkrl

# 2. 生成 schema 绑定并运行本地质量门禁
make check

# 3. 本地 smoke（待 Phase 2 实现）
python scripts/train.py --config configs/train/ppo_mlp.yaml
```

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

> ⚠️ 当前为**接口级占位骨架**：目录、签名、文档与协议已就位，具体实现按 [Roadmap](./AGENTS.md#roadmap) Phase 0→8 推进。
> Mod 的 C# 编译需 Hollow Knight 程序集与 HK Modding API，本机通常不具备，详见 [`docs/mod_dev.md`](./docs/mod_dev.md)。

## 当前状态

**Phase 0 — 调研与环境准备**。下一步优先级（P0）：双向 step/reset 协议、Gymnasium Env、Clean episode lifecycle、action mask、reward events、单 Boss baseline。

## License

见 [`LICENSE`](./LICENSE)。
