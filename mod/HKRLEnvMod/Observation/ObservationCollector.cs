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
            var player = ReadPlayerSafe();
            var entities = ReadEntitiesSafe(player);
            return new ObservationSnapshot(
                ReadGlobalSafe(taskId, episodeId),
                player,
                entities,
                BuildEntityMask(entities.Count));
        }

        private PlayerObservation ReadPlayerSafe()
        {
            try
            {
                return _player.Read();
            }
            catch (Exception exception)
            {
                global::HKRLEnvMod.Debug.Logger.Error("Failed to read player observation", exception);
                return DefaultPlayer();
            }
        }

        private IReadOnlyList<EntityObservation> ReadEntitiesSafe(PlayerObservation player)
        {
            try
            {
                return _entities.Collect(player);
            }
            catch (Exception exception)
            {
                global::HKRLEnvMod.Debug.Logger.Error("Failed to read entity observations", exception);
                return Array.Empty<EntityObservation>();
            }
        }

        private GlobalObservation ReadGlobalSafe(int taskId, ulong episodeId)
        {
            try
            {
                return _global.Read(taskId, episodeId);
            }
            catch (Exception exception)
            {
                global::HKRLEnvMod.Debug.Logger.Error("Failed to read global observation", exception);
                return new GlobalObservation(
                    sceneHash: 0,
                    arenaId: 0,
                    taskId: taskId,
                    difficulty: 0,
                    timeInEpisode: 0.0f,
                    timeScale: 0.0f,
                    fixedDeltaTime: 0.0f,
                    stageIndex: 0,
                    episodeId: episodeId);
            }
        }

        private static PlayerObservation DefaultPlayer()
        {
            return new PlayerObservation(
                0.0f,
                0.0f,
                0.0f,
                0.0f,
                hp: 1,
                maxHp: 1,
                soul: 0,
                maxSoul: 99,
                facing: 1,
                onGround: false,
                doubleJumpAvailable: false,
                canAttack: false,
                canCast: false,
                canFocus: false);
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
