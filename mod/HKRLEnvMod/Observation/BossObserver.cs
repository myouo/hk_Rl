using System.Collections.Generic;
using UnityEngine;

namespace HKRLEnvMod.Observation
{
    /// <summary>
    /// Reads boss entities via BossSceneController / HealthManager / PlayMakerFSM
    /// (hp, fsm_state_hash, phase, hitbox/hurtbox). Used standalone for the single-
    /// boss baseline (Phase 1/3) and by EntityObserver for multi-entity (Phase 4).
    /// </summary>
    public sealed class BossObserver
    {
        public void ReadInto(
            ICollection<EntityObservation> entities,
            EntityRegistry registry,
            ISet<int> aliveInstanceIds,
            PlayerObservation player)
        {
            if (entities == null)
            {
                throw new System.ArgumentNullException(nameof(entities));
            }
            if (registry == null)
            {
                throw new System.ArgumentNullException(nameof(registry));
            }
            if (aliveInstanceIds == null)
            {
                throw new System.ArgumentNullException(nameof(aliveInstanceIds));
            }

            foreach (var health in BossLocator.FindActiveBosses())
            {
                if (health == null || !health.isActiveAndEnabled)
                {
                    continue;
                }

                var gameObject = health.gameObject;
                var instanceId = gameObject.GetInstanceID();
                if (aliveInstanceIds.Contains(instanceId))
                {
                    continue;
                }

                aliveInstanceIds.Add(instanceId);
                entities.Add(EntityReadHelpers.BuildEntity(
                    health,
                    registry,
                    player,
                    HKRL.EntityType.Boss,
                    HKRL.Team.Enemy,
                    baseThreat: 100.0f,
                    flags: 1u << 4));
            }
        }
    }
}
