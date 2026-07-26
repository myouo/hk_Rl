using System.Collections.Generic;
using HKRLEnvMod.Observation;

namespace HKRLEnvMod.Rewards
{
    /// <summary>
    /// Derives core reward events from consecutive structured observations.
    /// This complements hook-based events and gives the environment a robust
    /// baseline signal even when a game hook is unavailable.
    /// </summary>
    public sealed class ObservationRewardTracker
    {
        private readonly Dictionary<int, TrackedEntity> _entities = new();
        private bool _hasPrevious;
        private int _playerHp;
        private int _playerSoul;

        public void Reset()
        {
            _hasPrevious = false;
            _playerHp = 0;
            _playerSoul = 0;
            _entities.Clear();
        }

        public void Update(ObservationSnapshot observation, RewardEventBuffer rewards)
        {
            if (!_hasPrevious)
            {
                Capture(observation);
                return;
            }

            EmitPlayerEvents(observation.Player, rewards);
            EmitEntityEvents(observation.Entities, rewards);
            Capture(observation);
        }

        private void EmitPlayerEvents(PlayerObservation player, RewardEventBuffer rewards)
        {
            var hpDelta = player.Hp - _playerHp;
            if (hpDelta < 0)
            {
                rewards.AddIfAbsent(HKRL.RewardEventKind.DamageTaken, amount: -hpDelta);
            }
            else if (hpDelta > 0)
            {
                rewards.AddIfAbsent(HKRL.RewardEventKind.Heal, amount: hpDelta);
            }

            var soulDelta = player.Soul - _playerSoul;
            if (soulDelta > 0)
            {
                rewards.Add(HKRL.RewardEventKind.SoulGained, amount: soulDelta);
            }

            if (_playerHp > 0 && player.Hp <= 0)
            {
                rewards.AddIfAbsent(HKRL.RewardEventKind.PlayerDeath);
            }
        }

        private void EmitEntityEvents(
            IReadOnlyList<EntityObservation> entities,
            RewardEventBuffer rewards)
        {
            var observed = new HashSet<int>();
            var hasNewBossIdentity = false;
            for (var i = 0; i < entities.Count; i++)
            {
                var entity = entities[i];
                observed.Add(entity.EntityId);
                if (entity.EntityType == HKRL.EntityType.Boss
                    && !_entities.ContainsKey(entity.EntityId))
                {
                    hasNewBossIdentity = true;
                }
                if (entity.MaxHp <= 0 || !_entities.TryGetValue(entity.EntityId, out var previous))
                {
                    continue;
                }

                if (entity.Hp < previous.Hp)
                {
                    rewards.AddIfAbsent(
                        HKRL.RewardEventKind.DamageDealt,
                        entity.EntityId,
                        previous.Hp - entity.Hp);
                }

                if (previous.EntityType == HKRL.EntityType.Boss
                    && previous.Hp > 0
                    && entity.Hp <= 0)
                {
                    rewards.AddIfAbsent(HKRL.RewardEventKind.BossKilled, entity.EntityId);
                }
            }

            // Several bosses destroy/disable their HealthManager immediately on
            // death, so no hp=0 frame exists. Treat disappearance of a previously
            // tracked boss as the fallback kill signal. StepController only ends
            // the episode after no living boss entities remain, which preserves
            // multi-boss fights.
            foreach (var pair in _entities)
            {
                if (pair.Value.EntityType == HKRL.EntityType.Boss
                    && pair.Value.Hp > 0
                    && !hasNewBossIdentity
                    && !observed.Contains(pair.Key))
                {
                    rewards.AddIfAbsent(HKRL.RewardEventKind.BossKilled, pair.Key);
                }
            }
        }

        private void Capture(ObservationSnapshot observation)
        {
            _playerHp = observation.Player.Hp;
            _playerSoul = observation.Player.Soul;
            _entities.Clear();
            for (var i = 0; i < observation.Entities.Count; i++)
            {
                var entity = observation.Entities[i];
                if (entity.MaxHp <= 0)
                {
                    continue;
                }

                _entities[entity.EntityId] = new TrackedEntity(entity.Hp, entity.EntityType);
            }

            _hasPrevious = true;
        }

        private readonly struct TrackedEntity
        {
            public TrackedEntity(int hp, HKRL.EntityType entityType)
            {
                Hp = hp;
                EntityType = entityType;
            }

            public int Hp { get; }
            public HKRL.EntityType EntityType { get; }
        }
    }
}
