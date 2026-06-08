using System;
using System.Collections.Generic;

namespace HKRLEnvMod.Observation
{
    public readonly struct EntityObservation
    {
        public EntityObservation(
            int entityId,
            HKRL.EntityType entityType,
            HKRL.Team team,
            int prefabHash = 0,
            int fsmNameHash = 0,
            int fsmStateHash = 0,
            float posX = 0.0f,
            float posY = 0.0f,
            float relX = 0.0f,
            float relY = 0.0f,
            float velX = 0.0f,
            float velY = 0.0f,
            int hp = 0,
            int maxHp = 0,
            float hurtboxCenterX = 0.0f,
            float hurtboxCenterY = 0.0f,
            float hurtboxSizeX = 0.0f,
            float hurtboxSizeY = 0.0f,
            bool hitboxActive = false,
            int damage = 0,
            float ttl = 0.0f,
            int phase = 0,
            float threatScore = 0.0f,
            uint flags = 0)
        {
            EntityId = entityId;
            EntityType = entityType;
            Team = team;
            PrefabHash = prefabHash;
            FsmNameHash = fsmNameHash;
            FsmStateHash = fsmStateHash;
            PosX = posX;
            PosY = posY;
            RelX = relX;
            RelY = relY;
            VelX = velX;
            VelY = velY;
            Hp = hp;
            MaxHp = maxHp;
            HurtboxCenterX = hurtboxCenterX;
            HurtboxCenterY = hurtboxCenterY;
            HurtboxSizeX = hurtboxSizeX;
            HurtboxSizeY = hurtboxSizeY;
            HitboxActive = hitboxActive;
            Damage = damage;
            Ttl = ttl;
            Phase = phase;
            ThreatScore = threatScore;
            Flags = flags;
        }

        public int EntityId { get; }
        public HKRL.EntityType EntityType { get; }
        public HKRL.Team Team { get; }
        public int PrefabHash { get; }
        public int FsmNameHash { get; }
        public int FsmStateHash { get; }
        public float PosX { get; }
        public float PosY { get; }
        public float RelX { get; }
        public float RelY { get; }
        public float VelX { get; }
        public float VelY { get; }
        public int Hp { get; }
        public int MaxHp { get; }
        public float HurtboxCenterX { get; }
        public float HurtboxCenterY { get; }
        public float HurtboxSizeX { get; }
        public float HurtboxSizeY { get; }
        public bool HitboxActive { get; }
        public int Damage { get; }
        public float Ttl { get; }
        public int Phase { get; }
        public float ThreatScore { get; }
        public uint Flags { get; }
    }

    public sealed class ObservationSnapshot
    {
        public ObservationSnapshot(
            GlobalObservation global,
            PlayerObservation player,
            IReadOnlyList<EntityObservation> entities,
            IReadOnlyList<bool> entityMask)
        {
            Global = global;
            Player = player;
            Entities = entities;
            EntityMask = entityMask;
        }

        public GlobalObservation Global { get; }
        public PlayerObservation Player { get; }
        public IReadOnlyList<EntityObservation> Entities { get; }
        public IReadOnlyList<bool> EntityMask { get; }
    }

    /// <summary>
    /// Assembles a full Observation snapshot each tick from the sub-observers
    /// (global/player/entities). Output maps to HKRL.Observation (schema/hkrl.fbs).
    /// Semantics: docs/observation_schema.md. Main-thread only.
    /// </summary>
    public sealed class ObservationCollector
    {
        private readonly GlobalObserver _global = new();
        private readonly PlayerObserver _player = new();
        private readonly EntityObserver _entities = new();

        /// <summary>Collect a snapshot; returns data for MessageCodec to encode.</summary>
        public ObservationSnapshot Collect(int taskId = 0, ulong episodeId = 0)
        {
            var player = _player.Read();
            var entities = _entities.Collect(player);
            return new ObservationSnapshot(
                _global.Read(taskId, episodeId),
                player,
                entities,
                BuildEntityMask(entities.Count));
        }

        private static IReadOnlyList<bool> BuildEntityMask(int count)
        {
            if (count <= 0)
            {
                return Array.Empty<bool>();
            }

            var mask = new bool[count];
            for (var i = 0; i < mask.Length; i++)
            {
                mask[i] = true;
            }

            return mask;
        }
    }
}
