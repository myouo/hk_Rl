# Model Architecture

> Implements PRD §7. Code: `hkrl/models/`. The long-term performance ceiling of
> the project. **Build the environment right first, then scale the model** (PRD §15).

## 1. Why attention + recurrence

| Task complexity | Needs |
|---|---|
| single boss | fixed vector suffices |
| multi-boss / summons | compare/attend over variable entities |
| projectile bullet-hell | focus on nearest/fastest/most-dangerous |
| hazards | spatial avoidance |
| linear boss sequence | long-horizon resource management |

Attention handles a **variable number of entities**, dynamically decides what to
attend to, and composes with `entity_mask`. Recurrence captures temporal state
that explicit features can't fully cover (boss wind-up, trajectory history,
"what did I just press", heal-window timing).

## 2. Forward pass

```text
global_emb = MLP(GlobalState)
player_emb = MLP(PlayerState)

for entity in entities:                       # masked over entity_mask
    type_emb = Embedding(entity_type)
    id_emb   = Embedding(entity_id / boss_id) # optional
    feat_emb = MLP(entity_features)
    entity_emb = type_emb + feat_emb (+ id_emb)

entity_context = TransformerEncoder(entity_embs, key_padding_mask=~entity_mask)
#   or  CrossAttention(query=player_emb, key/value=entity_embs, mask=entity_mask)

memory_input = concat(global_emb, player_emb, entity_context,
                      prev_action_emb, prev_reward)
memory_out, next_hidden = GRU/LSTM(memory_input, prev_hidden)

# policy heads (one per action component, mask-aware):
movement_x_head, aim_y_head, button_heads, duration_head, macro_head(optional)
value_head = V(memory_out)
```

Module mapping:

| Code | Role |
|---|---|
| `models/encoders.py` | `GlobalEncoder`, `PlayerEncoder`, `EntityEncoder`, type/id embeddings |
| `models/entity_attention.py` | masked Transformer encoder / cross-attention, summary-token aggregation |
| `models/recurrent_policy.py` | GRU/LSTM memory + `ActorCritic` assembling encoders+attention+heads |
| `models/heads.py` | per-component policy heads (mask → -inf), value head |
| `models/mlp.py` | non-recurrent MLP baseline (Phase 3) |
| `models/base.py` | `ActorCritic` ABC: `forward / act / evaluate_actions / initial_state` |

## 3. Complexity control (PRD §7.3)

Attention is O(N²) but on-screen entities are bounded.

```text
max_entities = 64, attention_layers = 2, hidden_dim = 128 or 256, heads = 4
```

When entities exceed capacity (priority, applied mod-side via `threat_score`):
keep all bosses → nearest/fastest/most-threatening projectiles → nearest hazards
→ aggregate the rest into one **summary token**.

## 4. Recurrent memory & training

Store per-step for truncated BPTT:

```text
hidden_state, episode_start, sequence_length, action_mask,
prev_action, prev_reward, policy_version
```

```text
sequence_length = 32 or 64,  burn_in = optional 8
```

`initial_state()` returns zeroed hidden state; it is reset at episode boundaries
(and optionally between bosses in a linear sequence — config-controlled). The
recurrent buffer (`training/recurrent_buffer.py`) handles sequence chunking,
masking of padded timesteps, and burn-in.

## 5. Performance

- `entity_mask` everywhere (attention key-padding + masked pooling); never let
  padded slots leak gradient/signal.
- Reserve `torch.compile` and AMP (mixed precision) on the training path.
- Batch sequences contiguously to keep GPU utilization high.

## 6. Ablations (PRD Phase 5)

`MLP` vs `attention` vs `attention+GRU`. Plus observation-tier ablations
(privileged/reduced/human-visible — [`observation_schema.md`](./observation_schema.md) §7).
