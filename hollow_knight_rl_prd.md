# Hollow Knight 强化学习项目 PRD

> 版本：v0.1  
> 日期：2026-06-07  
> 目标：基于 Hollow Knight Mod 获取结构化游戏状态，构建可扩展的强化学习环境，支持单 Boss、多敌方单位、线性多 Boss 流程、本地推理 + 远程训练，以及长期的 entity encoder + recurrent memory 高上限模型。

---

## 1. 项目背景与定位

现有 `Vitao2/Hollow-Knight-Neural-Network` 已经验证了一个方向：通过 C# Unity Mod 从游戏中提取结构化状态，再用 Python PPO 模型训练小骑士对战 Hornet Protector。它的架构是：

```text
Hollow Knight + C# Mod
  -> named pipe 输出状态
  -> Python PPO 推理/训练
  -> vgamepad 外部输入动作
```

该项目对原型验证非常有价值，但长期扩展存在限制：

1. 主要目标是 Hornet Protector 单 Boss。
2. 状态空间是固定 player + single boss vector。
3. Boss 数据虽遍历 `BossSceneController.Instance.bosses`，但最终写入单个 `bx/by/bossHp/bossState` 字段，无法表达多个敌方单位。
4. 动作输出是 multi-binary buttons，缺少动作互斥、tap/hold/release、持续时间、action mask。
5. Mod 只做单向状态导出，不是完整 environment server。
6. Python 脚本把推理、训练、reward、vgamepad 控制耦合在一起。
7. reset 生命周期较粗糙，容易污染训练数据。
8. 4 帧堆叠可以解决一部分速度/短期历史问题，但无法充分表达 Boss 前摇、冷却、投射物轨迹、回血窗口等长期时序。

因此，本项目不应直接在 Vitao2 架构上堆功能，而应从第一版就重构为：

```text
HKRLMod Environment Server
  <-> Transport Protocol
  <-> Local GameWorker / Gymnasium Env
  <-> Remote Learner
  <-> Coordinator / Evaluator / Metrics
```

---

## 2. 项目目标

### 2.1 MVP 目标

在 Godhome/Hall of Gods 中，基于结构化状态训练模型击败单个 Boss，例如 Hornet Protector 或 Gruz Mother。

MVP 成功标准：

- Mod 能稳定输出 player + boss observation。
- Python 能通过标准 `env.reset()` / `env.step(action)` 与游戏交互。
- random policy 可连续跑 1000 个 episode 不崩溃。
- PPO baseline 能在单 Boss 上学习到稳定正收益。
- 每个 episode 有完整日志：reward、damage dealt、damage taken、win/loss、episode length、SPS、reset status。

### 2.2 中期目标

支持同时多个敌方单位与复杂场景对象：

- 多 Boss 或多敌人同时存在。
- 投射物、召唤物、危险区域、平台边缘、陷阱。
- 可变数量 entity list。
- action mask。
- macro actions。
- recurrent policy。

### 2.3 长期目标

支持线性多 Boss 流程与分布式训练：

- 一个 episode 内连续挑战多个 Boss。
- 或者通过 curriculum 在多个 Boss/task 之间采样。
- 本地推理 + 远程训练。
- 多 GameWorker 并行采样。
- entity encoder + attention + recurrent memory。
- 自动评估与灾难性遗忘监控。

---

## 3. “多 Boss”的两层含义与处理方案

用户提出的“多 Boss”有两种含义，它们需要不同设计。

### 3.1 同时多个敌方单位

例如：

- 螳螂领主多人同时出现。
- 骨钉兄弟。
- 神驯者 + 坐骑。
- Boss + 召唤物。
- Boss + 弹幕 + hazard。

此时必须修改 Mod 状态空间。固定字段：

```text
boss_x, boss_y, boss_hp, boss_state
```

不再够用。必须改成 entity list：

```text
entities[]:
  entity_id
  entity_type
  team
  prefab_hash
  scene_hash
  pos
  vel
  hp
  max_hp
  fsm_state
  phase
  hurtbox
  hitbox_active
  damage
  ttl
  threat_flags
```

同时需要：

- `entity_mask`：标记哪些 slot 有效。
- `type_embedding`：区分 boss、enemy、projectile、hazard、platform。
- `team`：enemy / neutral / player-created projectile。
- `relative_position`：建议以玩家为中心归一化坐标。
- `stable_entity_id`：同一个对象在多帧中保持身份一致。
- `priority_top_k`：实体过多时按威胁度或距离裁剪。

推荐第一版容量：

```text
max_bosses = 4
max_enemies = 8
max_projectiles = 32
max_hazards = 16
```

如果为了简化，可以合并为：

```text
max_entities = 64
```

并用 `entity_type` 区分类型。

### 3.2 线性流程挑战多位 Boss

例如：

```text
Boss A -> Boss B -> Boss C -> Boss D
```

这更像 curriculum / task sequence / long-horizon RL。状态空间不一定需要“同时多个 Boss”，但需要：

- `task_id` / `boss_id` / `arena_id`。
- `stage_index`：当前是流程中的第几战。
- `remaining_boss_count`。
- boss transition event。
- episode 内是否保留血量/魂量。
- 是否在 Boss 间清空 RNN hidden state。

处理方式有两种：

#### 方案 A：多 Boss 分开训练 + curriculum

```text
episode = 单个 Boss
task sampler 按 Boss 采样
每个 Boss 独立评估胜率
```

优点：

- 稳定。
- 易 debug。
- PPO 训练更简单。
- 适合前中期。

#### 方案 B：一个 episode 连续打多个 Boss

```text
episode = Boss A + Boss B + Boss C + ...
sub_episode = 每个 Boss 小阶段
```

优点：

- 接近最终目标。
- 能学习资源管理、回血策略、跨战斗长期规划。

缺点：

- horizon 很长。
- credit assignment 难。
- reset/debug 成本高。
- PPO 数据方差大。

建议路线：

```text
先做方案 A，再做方案 B。
```

---

## 4. 最终架构

### 4.1 总体架构

```text
+--------------------------------------------------------------+
|                      Remote GPU Server                        |
|                                                              |
|  +------------------+     +-------------------------------+  |
|  | Learner          |<--->| Checkpoint / Policy Registry   |  |
|  | PPO/APPO/IMPALA  |     +-------------------------------+  |
|  +------------------+                ^                       |
|           ^                           |                       |
|           | rollout batches           | weights               |
|           v                           v                       |
|  +------------------+     +-------------------------------+  |
|  | Coordinator      |<--->| Metrics / Evaluator / Logger   |  |
|  +------------------+     +-------------------------------+  |
+----------------------------^---------------------------------+
                             |
                             | TCP/gRPC/Ray/ZeroMQ
                             |
+----------------------------v---------------------------------+
|                         Game PC                               |
|                                                              |
|  +------------------+     +-------------------------------+  |
|  | GameWorker       |<--->| HKRLMod inside Hollow Knight   |  |
|  | Local inference  |     | Env Server                     |  |
|  | Gym Env wrapper  |     | Observation / Action / Reset   |  |
|  +------------------+     +-------------------------------+  |
+--------------------------------------------------------------+
```

### 4.2 组件职责

#### HKRLMod

运行在 Hollow Knight 进程内，职责：

- 采集 observation。
- 接收 action。
- 在主线程按 tick 应用动作。
- hook reward events。
- 管理 reset lifecycle。
- 管理 time scale / fixed timestep。
- 输出 action mask。
- 输出 debug info。
- 保证网络线程不阻塞 Unity 主线程。

#### Transport

推荐：

- 本机 MVP：TCP localhost 或 duplex named pipe。
- 跨机器：TCP/gRPC/ZeroMQ。
- 分布式：Ray actor 或自定义 coordinator + gRPC。

消息编码：

- MVP：MessagePack。
- 稳定长期：Protobuf / FlatBuffers。
- 不建议长期使用 JSON，因为体积和解析成本较高。

协议要求：

- 每条消息带 `schema_version`。
- 每条 step 带 `tick_id`。
- 支持 heartbeat。
- 支持 reset ack。
- 支持超时与重连。
- 支持 policy_version。
- 支持任务切换命令。

#### GameWorker

运行在 Game PC 本地，职责：

- 本地推理。
- 包装成 Gymnasium-like Env。
- 与 HKRLMod 通信。
- 管理 rollout buffer。
- 上传 trajectory batch 到远程 learner。
- 拉取最新 checkpoint。
- 处理 crash/reconnect/reset failure。
- 记录本地 metrics。

核心原则：

```text
实时 action loop 不跨远程网络。
```

也就是：

```text
obs -> 本地 policy -> action -> game
```

远程 GPU 只做大 batch training。

#### Learner

运行在远程 GPU 机器，职责：

- 收集 trajectory batches。
- 根据 policy_version 管理 on-policy 数据。
- 更新模型。
- 发布 checkpoint。
- 记录 loss、entropy、KL、value loss、explained variance 等训练指标。

#### Coordinator

职责：

- 管理多个 GameWorker。
- 分配 task/boss。
- curriculum sampling。
- checkpoint registry。
- 失败恢复。
- 训练/评估隔离。
- 指标聚合。

#### Evaluator

职责：

- 固定种子/固定任务评估。
- 每个 Boss 独立胜率。
- 录像或 replay。
- 防止训练 reward 上升但真实胜率下降。
- 检测灾难性遗忘。

---

## 5. Mod 重写方案

### 5.1 前置条件

需要掌握：

- C#。
- Unity MonoBehaviour 生命周期：Awake、Start、Update、FixedUpdate、Coroutine。
- Harmony patch。
- BepInEx 或 Hollow Knight Modding API。
- Hollow Knight 内部类：`HeroController`、`PlayerData`、`HealthManager`、`BossSceneController`、`GameManager`、`PlayMakerFSM`。
- ILSpy / dnSpy 查看 Assembly-CSharp。
- 多线程基础：lock、ConcurrentQueue、ring buffer。
- 游戏主线程限制：不要在网络线程直接访问/修改 Unity 对象。

### 5.2 模块目录

```text
HKRLEnvMod/
  HKRLEnvMod.cs

  Transport/
    TcpServer.cs
    MessageCodec.cs
    Protocol.cs
    Heartbeat.cs

  Env/
    StepController.cs
    EpisodeLifecycle.cs
    ResetManager.cs
    SimControl.cs
    SceneController.cs

  Observation/
    ObservationCollector.cs
    GlobalObserver.cs
    PlayerObserver.cs
    EntityObserver.cs
    BossObserver.cs
    ProjectileObserver.cs
    HazardObserver.cs
    EntityRegistry.cs

  Action/
    ActionApplier.cs
    InputInjector.cs
    ActionMasker.cs
    MacroActionScheduler.cs

  Rewards/
    RewardEventBuffer.cs
    DamageHooks.cs
    HealHooks.cs
    DeathHooks.cs
    SceneHooks.cs

  Debug/
    Overlay.cs
    Logger.cs
    SnapshotRecorder.cs
```

### 5.3 主线程与网络线程设计

错误做法：

```text
网络线程收到 action 后直接操作 HeroController。
```

正确做法：

```text
NetworkThread:
  receive StepRequest
  enqueue request

FixedUpdate MainThread:
  dequeue latest action
  apply action
  collect observation
  collect reward events
  write snapshot to response queue

NetworkThread:
  send StepResponse
```

### 5.4 Step 协议

```text
StepRequest:
  schema_version
  env_id
  tick_id
  command
  action
  action_repeat
  policy_version
  client_time

StepResponse:
  schema_version
  env_id
  tick_id
  server_tick
  observation
  reward_events
  action_mask
  terminated
  truncated
  lifecycle_state
  info
```

`command`：

```text
STEP
RESET
PAUSE
RESUME
SET_TASK
SET_TIMESCALE
PING
```

### 5.5 Observation Schema

#### GlobalState

```text
scene_hash
arena_id
task_id
difficulty
time_in_episode
time_scale
fixed_delta_time
stage_index
episode_id
```

#### PlayerState

```text
pos_x, pos_y
vel_x, vel_y
hp, max_hp
soul, max_soul
facing
on_ground
wall_sliding
jumping
falling
dashing
shadow_dashing
invulnerable
invuln_timer
attack_lock_timer
cast_lock_timer
focus_state
dash_cooldown
double_jump_available
can_attack
can_cast
can_focus
```

#### EntityState

```text
entity_id
entity_type
team
prefab_hash
fsm_name_hash
fsm_state_hash
pos_x, pos_y
rel_x, rel_y
vel_x, vel_y
hp, max_hp
hurtbox_center_x, hurtbox_center_y
hurtbox_size_x, hurtbox_size_y
hitbox_active
damage
ttl
phase
threat_score
flags
```

#### Entity Types

```text
PLAYER = 0
BOSS = 1
ENEMY = 2
PROJECTILE = 3
HAZARD = 4
PLATFORM = 5
PICKUP = 6
EFFECT = 7
UNKNOWN = 255
```

### 5.6 Reward Events

Mod 不直接写死最终 reward，只上报事件：

```text
DamageDealt:
  target_entity_id
  amount
  damage_type

DamageTaken:
  source_entity_id
  amount

Heal:
  amount

SoulGained:
  amount

BossKilled:
  entity_id

PlayerDeath:
  reason

SceneChanged:
  from_scene
  to_scene

InvalidAction:
  action_id
  reason

Stagger:
  entity_id
```

Python reward 函数组合：

```text
reward =
  + 1.0  * damage_dealt
  - 8.0  * damage_taken
  + 0.5  * soul_gained
  + 2.0  * heal_amount
  + 100  * boss_kill
  - 100  * player_death
  - 0.001 * time_step
  - 0.01 * invalid_action
  + optional shaping
```

### 5.7 Clean Episode Lifecycle

状态机：

```text
IDLE
  -> RESET_REQUESTED
  -> FREEZE_INPUT
  -> CLEAR_EVENTS
  -> LOAD_SCENE
  -> WAIT_SCENE_READY
  -> WAIT_PLAYER_READY
  -> WAIT_BOSS_READY
  -> RESTORE_PLAYER_STATE
  -> CLEAR_PROJECTILES
  -> COUNTDOWN
  -> RUNNING
  -> TERMINATING
  -> REPORT_DONE
  -> CLEANUP
  -> IDLE
```

关键要求：

- reset 不应混入旧 episode 的 reward event。
- reset 完成后才允许 step。
- 每个 episode 有唯一 `episode_id`。
- 死亡、胜利、场景切换都进入 `TERMINATING`。
- `done` 之后不再收集新 reward。
- reset failure 必须返回错误码，而不是继续训练。

---

## 6. 动作空间设计

### 6.1 问题

Hollow Knight 的动作不是简单按钮组合。关键语义包括：

- 短跳 / 长跳。
- 攻击方向：前、上、下。
- 下劈 pogo。
- dash timing。
- 施法方向。
- focus hold。
- nail art hold/release。
- 攻击/冲刺/施法硬直。
- 动作缓冲和取消窗口。
- 冷却与资源限制。

如果继续用纯 MultiBinary，会产生大量无效组合：

```text
left + right
up + down
focus + dash
attack + focus
dash while cooldown
cast without soul
jump while unavailable
```

### 6.2 推荐动作空间

```text
movement_x: Discrete(3)
  0 = left
  1 = neutral
  2 = right

aim_y: Discrete(3)
  0 = down
  1 = neutral
  2 = up

buttons:
  jump_tap
  jump_hold
  dash
  attack
  cast
  focus_hold
  dream_nail
  nail_art_hold
  nail_art_release

duration:
  1 / 2 / 4 / 8 ticks
```

### 6.3 Action Mask

```text
dash_cooldown > 0 -> mask dash
soul < cast_cost -> mask cast/focus
attack_lock > 0 -> mask attack/cast/dash
not grounded and no double_jump -> mask jump
focusing -> mask attack/dash/cast
left/right mutually exclusive
up/down mutually exclusive
```

### 6.4 Macro Actions

早期训练可以加入宏动作：

```text
approach
retreat
jump_attack
pogo
dash_away
dash_through
cast_forward
cast_up
focus_when_safe
short_hop
long_jump
```

推荐策略：

```text
Phase 1: macro-heavy
Phase 2: macro + primitive mixed
Phase 3: mostly primitive with learned duration
```

---

## 7. 模型结构：Entity Encoder + Attention + Recurrent Memory

### 7.1 为什么使用 Attention

Boss 环境复杂度会随任务增加：

- 单 Boss：固定向量足够。
- 多 Boss：需要比较多个威胁。
- 弹幕：需要关注最近/最快/最危险 projectile。
- 召唤物：需要在 Boss 与小怪之间切换目标。
- Hazard：需要空间避障。
- 线性流程：需要长期资源管理。

Attention 的优势：

- 可处理可变数量实体。
- 能动态决定关注 Boss、投射物、危险区域还是回血窗口。
- 可以配合 entity mask。
- 对多敌方单位比固定拼接向量更自然。

### 7.2 模型结构

```text
global_emb = MLP(GlobalState)
player_emb = MLP(PlayerState)

for entity in entities:
  type_emb = Embedding(entity_type)
  id_emb   = optional Embedding(entity_id / boss_id)
  feat_emb = MLP(entity_features)
  entity_emb = type_emb + feat_emb + optional id_emb

entity_context = TransformerEncoder(entity_embs, entity_mask)
或者
entity_context = CrossAttention(query=player_emb, key/value=entity_embs)

memory_input = concat(
  global_emb,
  player_emb,
  entity_context,
  prev_action_emb,
  prev_reward
)

memory_out, next_hidden = GRU/LSTM(memory_input, prev_hidden)

policy_heads:
  movement_x_head
  aim_y_head
  button_heads
  duration_head
  macro_head optional

value_head:
  V(memory_out)
```

### 7.3 复杂度控制

Transformer attention 是 O(N²)，但 Hollow Knight 单屏实体数量通常可控。建议：

```text
max_entities = 64
attention_layers = 2
hidden_dim = 128 or 256
heads = 4
```

若实体过多：

1. 保留所有 Boss。
2. 保留最近/最快/威胁最高的 projectiles。
3. 保留最近 hazards。
4. 其他 entities 聚合成 summary token。

### 7.4 Recurrent Memory

需要 RNN 的原因：

- Boss 前摇持续时间。
- dash 冷却和 invulnerability 窗口。
- 自己刚按了什么动作。
- 投射物轨迹历史。
- 回血窗口判断。
- 线性 Boss 流程中的资源管理。

训练 recurrent policy 时必须存：

```text
hidden_state
episode_start
sequence_length
action_mask
prev_action
prev_reward
policy_version
```

使用 truncated BPTT：

```text
sequence_length = 32 or 64
burn_in = optional 8
```

---

## 8. 本地推理 + 远程训练

### 8.1 推荐流程

```text
GameWorker:
  load latest policy
  while running:
    obs = env.current_obs
    action, logprob, value, hidden = local_policy(obs, hidden)
    next_obs, reward, done, info = env.step(action)
    buffer.add(...)
    if buffer full:
      upload rollout batch
    if new checkpoint:
      load weights
```

```text
Learner:
  receive rollout batches
  filter by policy_version
  update PPO/APPO/IMPALA
  publish checkpoint
```

### 8.2 为什么不做远程实时推理

不推荐：

```text
Game PC obs -> Remote GPU -> action -> Game PC
```

原因：

- 网络延迟会污染动作时序。
- jitter 会让同一策略表现不稳定。
- 动作游戏需要稳定 tick。
- GPU 推理单样本不一定比本地 CPU/GPU 快。
- 远程推理只有在批量多环境推理时才有意义。

### 8.3 数据格式

RolloutBatch：

```text
obs_global
obs_player
obs_entities
entity_mask
actions
log_probs
values
rewards
dones
truncateds
action_masks
prev_actions
rnn_states
episode_ids
task_ids
policy_version
```

---

## 9. 根源性问题与解决方案

### 9.1 部分可观测性

问题：如果 observation 不包含 hitbox、hurtbox、冷却、前摇、投射物 ttl，环境对 agent 就不是完整马尔可夫状态。

解决：

- 优先补显式状态。
- 添加 cooldown timers、lock timers、hitbox flags。
- 添加 entity history 或 RNN。
- 不依赖 frame stacking 解决所有记忆问题。

### 9.2 动作语义不完整

问题：按钮维度不等于操作语义，tap/hold/release/duration 对 Hollow Knight 很关键。

解决：

- 使用 hybrid action。
- 显式 duration。
- 加 action mask。
- 加 macro actions。
- Mod 内部注入动作，减少 vgamepad 时序不确定性。

### 9.3 Reset 污染

问题：死亡/胜利后的旧事件、场景加载、Boss 未就绪都会污染训练数据。

解决：

- Clean Episode Lifecycle。
- reset ack。
- lifecycle_state。
- episode_id。
- event buffer clear。
- ready checks。

### 9.4 Reward hacking

问题：距离 shaping、伤害奖励、回血奖励可能导致策略刷中间奖励，不追求胜利。

解决：

- 终局奖励远大于中间 shaping。
- 分离 reward events 与 reward function。
- 每次训练保留无 shaping 评估指标。
- 以 win rate / damage ratio 作为核心评估，不只看 reward。

### 9.5 PPO 与异步采样冲突

问题：远程多 worker 会产生旧 policy 数据，PPO 对 stale data 敏感。

解决：

- 每条 batch 带 policy_version。
- 同步 PPO：固定版本收集，统一 update。
- 异步训练改 APPO/IMPALA。
- 过旧 rollout 丢弃或降权。

### 9.6 SPS 而非 FPS

问题：游戏高帧率不等于训练高效率。

解决：

- 指标以 samples per second 为主。
- 控制 `Time.timeScale` 与 `fixedDeltaTime`。
- 使用 action_repeat。
- 多实例并行。
- 减少渲染质量。
- 优化 reset 耗时。

### 9.7 泛化与灾难性遗忘

问题：多 Boss 训练时，模型可能学会新 Boss 后忘记旧 Boss。

解决：

- task_id/boss_id embedding。
- balanced task sampler。
- old task replay。
- 每个 Boss 独立 evaluation。
- curriculum 难度递增。
- shared encoder + task adapter。
- 必要时使用 mixture-of-experts。

### 9.8 内部信息过强

问题：读取 FSM/hitbox 是超人类信息，会让 agent 过拟合内部状态。

解决：

- 明确项目定位：game-state agent，不是视觉人类 agent。
- 分层实验：
  - privileged state。
  - reduced state。
  - human-visible state。
- 最终报告分别评估。

### 9.9 Mod 脆弱性

问题：游戏版本、Mod API、FSM 名称、scene 加载都可能变化。

解决：

- schema_version。
- mod version lock。
- observation health checks。
- fallback entity fields。
- 单元测试：进入场景、读取 boss、reset、死亡、击杀。
- 关键 hook 加 try/catch 和 debug overlay。

### 9.10 安全与远程通信

问题：远程训练涉及网络服务和模型文件同步。

解决：

- 只绑定局域网或 localhost tunnel。
- token 鉴权。
- 不暴露公网端口。
- checkpoint 签名或 hash 校验。
- worker 只执行白名单 command。

---

## 10. Roadmap

### Phase 0：调研与环境准备

目标：

- 跑通 Hollow Knight Mod 开发环境。
- 能编译并加载一个 Hello World mod。
- 读懂 Vitao2 的状态导出逻辑。
- 确定使用 BepInEx 还是 HK Modding API。

Todo：

- [ ] 安装 Hollow Knight Steam 版。
- [ ] 安装 BepInEx 或 Modding API。
- [ ] 配置 C# IDE。
- [ ] 用 ILSpy/dnSpy 查看 Assembly-CSharp。
- [ ] 写 Hello World mod。
- [ ] 在日志中输出玩家位置与场景名。

Milestone：

- [ ] 启动游戏后 mod 成功加载。
- [ ] 日志能输出 player position。
- [ ] 游戏关闭时无异常。

### Phase 1：HKRLMod v0 Environment Server

目标：

- 双向通信。
- player + single boss observation。
- step/reset 协议。
- random policy 跑通。

Todo：

- [ ] 实现 TCP server。
- [ ] 实现 MessagePack/Protobuf 编码。
- [ ] 实现 StepRequest/StepResponse。
- [ ] 实现 ObservationCollector。
- [ ] 实现 PlayerObserver。
- [ ] 实现 BossObserver。
- [ ] 实现 RewardEventBuffer。
- [ ] 实现 ResetManager v0。
- [ ] 实现 debug overlay。

Milestone：

- [ ] Python 连接 mod。
- [ ] 每个 FixedUpdate 能返回 observation。
- [ ] Python 可发送 action。
- [ ] 1000 steps 不断线。
- [ ] reset 10 次成功。

### Phase 2：Python Gymnasium Env

目标：

- 标准化 `reset()` / `step()`。
- 本地 action loop。
- 可接 PPO。

Todo：

- [ ] `hkrl/env.py`。
- [ ] `hkrl/transport.py`。
- [ ] `hkrl/protocol.py`。
- [ ] `hkrl/reward.py`。
- [ ] `hkrl/spaces.py`。
- [ ] `hkrl/wrappers.py`。
- [ ] random policy。
- [ ] scripted policy。
- [ ] episode logger。

Milestone：

- [ ] `check_env` 或自定义 smoke test 通过。
- [ ] random policy 连续 100 episode。
- [ ] 每个 episode 有完整 CSV/JSONL 日志。

### Phase 3：单 Boss PPO Baseline

目标：

- Hornet Protector 或 Gruz Mother 单 Boss 可学习。

Todo：

- [ ] MLP Actor-Critic。
- [ ] PPO rollout buffer。
- [ ] action mask v0。
- [ ] reward shaping v0。
- [ ] checkpoint save/load。
- [ ] TensorBoard/WandB 日志。
- [ ] 固定评估脚本。

Milestone：

- [ ] 胜率 > 30%。
- [ ] invalid action ratio < 5%。
- [ ] reset failure < 1%。
- [ ] SPS 可稳定记录。

### Phase 4：实体化观测

目标：

- 支持同时多个敌方单位。
- 支持 projectiles/hazards。

Todo：

- [ ] EntityRegistry。
- [ ] stable_entity_id。
- [ ] bosses[]。
- [ ] enemies[]。
- [ ] projectiles[]。
- [ ] hazards[]。
- [ ] entity_mask。
- [ ] top-k priority filtering。
- [ ] entity debug overlay。

Milestone：

- [ ] 多 Boss 场景能看到所有 Boss。
- [ ] projectile 数量与位置合理。
- [ ] entity_id 跨帧稳定。
- [ ] 不同场景 schema 不变。

### Phase 5：Attention + Recurrent Policy

目标：

- entity encoder + attention。
- GRU/LSTM memory。
- recurrent PPO。

Todo：

- [ ] Entity MLP encoder。
- [ ] type embedding。
- [ ] attention encoder。
- [ ] player/global fusion。
- [ ] GRU/LSTM。
- [ ] recurrent rollout buffer。
- [ ] sequence training。
- [ ] hidden state reset logic。
- [ ] ablation：MLP vs attention vs attention+GRU。

Milestone：

- [ ] 多敌方单位任务胜率高于 MLP baseline。
- [ ] 复杂弹幕/召唤任务中 attention+GRU 表现更稳。
- [ ] sequence training 无 shape/mask bug。

### Phase 6：本地推理 + 远程训练

目标：

- GameWorker 本地推理。
- Learner 远程训练。
- checkpoint 同步。

Todo：

- [ ] Rollout uploader。
- [ ] Checkpoint registry。
- [ ] policy_version。
- [ ] learner server。
- [ ] worker heartbeat。
- [ ] stale rollout filtering。
- [ ] graceful reconnect。

Milestone：

- [ ] Game PC 无 GPU 也能本地推理。
- [ ] Remote GPU 能持续训练。
- [ ] worker 可热加载新权重。
- [ ] 网络断开后可恢复。

### Phase 7：多 Boss Curriculum

目标：

- 多 Boss 任务泛化。
- 防止灾难性遗忘。

Todo：

- [ ] task config。
- [ ] boss_id embedding。
- [ ] balanced task sampler。
- [ ] per-boss evaluation。
- [ ] curriculum scheduler。
- [ ] old-task replay。
- [ ] checkpoint regression tests。

Milestone：

- [ ] 3 个 Boss 胜率均稳定。
- [ ] 新增 Boss 不导致旧 Boss 胜率大幅下降。
- [ ] 线性 Boss 流程 v0 跑通。

### Phase 8：多实例扩展与工程化

目标：

- 多 GameWorker。
- 多机器采样。
- 高 SPS。
- 完整监控。

Todo：

- [ ] Coordinator。
- [ ] worker registry。
- [ ] crash recovery。
- [ ] metrics dashboard。
- [ ] eval worker pool。
- [ ] automated smoke tests。
- [ ] profile mod CPU / network / Python inference。
- [ ] release docs。

Milestone：

- [ ] 2-4 个 GameWorker 并行。
- [ ] SPS 线性提升或接近线性提升。
- [ ] 单 worker 崩溃不影响训练。
- [ ] 每日 regression eval 自动产出报告。

---

## 11. 推荐仓库结构

```text
hk-rl/
  mod/
    HKRLEnvMod/
      HKRLEnvMod.cs
      Transport/
      Env/
      Observation/
      Action/
      Rewards/
      Debug/

  python/
    hkrl/
      env.py
      protocol.py
      transport.py
      spaces.py
      reward.py
      wrappers.py
      models/
        mlp.py
        entity_attention.py
        recurrent_policy.py
      training/
        ppo.py
        recurrent_ppo.py
        appo.py
      worker/
        game_worker.py
        rollout_buffer.py
        checkpoint_client.py
      learner/
        learner_server.py
        checkpoint_registry.py
      eval/
        evaluator.py
        scripted_policies.py
      utils/
        logging.py
        config.py

  configs/
    tasks/
      hornet_protector.yaml
      gruz_mother.yaml
      mantis_lords.yaml
    train/
      ppo_mlp.yaml
      ppo_attention_gru.yaml
      remote_learner.yaml

  docs/
    PRD.md
    protocol.md
    mod_dev.md
    reward_design.md
    troubleshooting.md

  scripts/
    run_worker.ps1
    run_learner.sh
    run_eval.py
```

---

## 12. 配置示例

### 12.1 Task Config

```yaml
task_id: hornet_protector_attuned
scene: GG_Hornet_1
difficulty: attuned
time_limit_seconds: 180
player:
  hp: max
  soul: 66
  charms: default
reward:
  boss_damage: 1.0
  player_damage: -8.0
  boss_kill: 100.0
  player_death: -100.0
  time_penalty: -0.001
observation:
  max_entities: 64
  include_fsm_state: true
  include_hitbox: true
action:
  action_repeat: 2
  enable_macro_actions: true
```

### 12.2 Training Config

```yaml
algorithm: recurrent_ppo
gamma: 0.995
gae_lambda: 0.95
clip_range: 0.2
learning_rate: 3e-4
rollout_steps: 2048
minibatch_size: 256
epochs: 4
sequence_length: 32
entropy_coef: 0.01
value_coef: 0.5
max_grad_norm: 0.5
policy:
  entity_hidden: 128
  attention_layers: 2
  attention_heads: 4
  rnn_type: gru
  rnn_hidden: 256
```

---

## 13. 指标体系

必须记录：

```text
episode_reward
win_rate
episode_length
damage_dealt
damage_taken
heal_count
death_reason
invalid_action_ratio
action_entropy
policy_kl
value_loss
policy_loss
explained_variance
SPS
reset_success_rate
reset_duration
worker_crash_count
per_boss_win_rate
per_boss_damage_ratio
```

更重要的是：

```text
训练 reward != 真实能力
```

核心评估应优先看：

1. per-boss win rate。
2. damage taken。
3. time to kill。
4. invalid action ratio。
5. 泛化到未训练 Boss 的表现。
6. old task regression。

---

## 14. 实现优先级

### P0 必须做

- 双向 step/reset 协议。
- Gymnasium Env。
- Clean episode lifecycle。
- action mask。
- reward events。
- 单 Boss baseline。

### P1 强烈建议

- entity list。
- local inference + remote training。
- recurrent policy。
- per-boss evaluation。
- metrics dashboard。

### P2 长期增强

- attention encoder。
- macro/primitive mixed action。
- curriculum。
- multiple GameWorkers。
- APPO/IMPALA。
- linear boss sequence。

### P3 可选实验

- imitation learning / behavior cloning。
- human replay recorder。
- mixture-of-experts。
- reduced-state vs privileged-state 对比。
- vision + state hybrid。

---

## 15. 最终建议

项目最关键的不是“神经网络要多大”，而是：

1. Mod 必须变成 environment server。
2. Observation 必须从固定 Boss 向 entity list 升级。
3. Action 必须表达 tap/hold/duration/mask。
4. Episode lifecycle 必须干净。
5. 本地推理和远程训练必须解耦。
6. 训练必须保留 per-task evaluation，防止 reward hacking 和灾难性遗忘。
7. Attention + recurrent memory 是长期上限方案，但应该在稳定环境和实体化观测之后再上。

推荐总路线：

```text
先把环境做对，再把模型做大。
先把单 Boss 跑稳，再做多敌方单位。
先做本地 worker，再接远程 learner。
先做 entity list，再做 attention。
先做 clean reset，再做 curriculum。
```

---

## 16. 参考资料

- Vitao2/Hollow-Knight-Neural-Network：C# mod + named pipe + Python PPO + vgamepad 的现有实现。  
  https://github.com/Vitao2/Hollow-Knight-Neural-Network
- Hollow Knight Modding API：Hollow Knight 的 Modding API/loader，使用 MonoMod，并提供 examples/docs。  
  https://github.com/hk-modding/api
- BepInEx plugin tutorial：BepInEx 插件通过 `BepInPlugin` 注解类，编译为 .NET DLL 后放入 `BepInEx/plugins`。  
  https://docs.bepinex.dev/articles/dev_guide/plugin_tutorial/index.html
- Gymnasium Env API：自定义环境需要 action_space、observation_space，并实现 reset/step。  
  https://gymnasium.farama.org/api/env/
- Ray RLlib scaling guide：RLlib 通过 Ray actors 扩展采样与学习吞吐。  
  https://docs.ray.io/en/latest/rllib/scaling-guide.html
- Ray actors：Ray actor 是有状态 worker/service，适合表达 worker、learner、coordinator。  
  https://docs.ray.io/en/latest/ray-core/actors.html
- Unity Time.timeScale：调整 timeScale 会影响游戏时间与物理 fixed timestep 的关系。  
  https://docs.unity3d.com/6000.4/Documentation/ScriptReference/Time-timeScale.html
- Unity fixedDeltaTime：fixedDeltaTime 控制固定时间步长，例如 0.01 表示每秒 100 个 fixed updates。  
  https://docs.unity.cn/Manual/TimeFrameManagement.html
- SB3 Contrib Recurrent PPO：提供 LSTM 版本 PPO。  
  https://sb3-contrib.readthedocs.io/en/master/modules/ppo_recurrent.html
- Stable-Baselines3 VecEnv：vectorized environments 可将多个独立环境堆成 batch。  
  https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html
