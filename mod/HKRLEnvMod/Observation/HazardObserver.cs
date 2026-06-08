namespace HKRLEnvMod.Observation
{
    using System.Collections.Generic;
    using UnityEngine;

    /// <summary>
    /// Reads static/dynamic hazards (spikes, danger zones, platform edges) for
    /// spatial avoidance (docs/observation_schema.md §3). Maps to HKRL.EntityState
    /// with entity_type = Hazard.
    /// </summary>
    public sealed class HazardObserver
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

            foreach (var collider in Object.FindObjectsOfType<Collider2D>())
            {
                if (collider == null || !collider.enabled || collider.gameObject == null)
                {
                    continue;
                }

                var gameObject = collider.gameObject;
                if (!IsLikelyHazard(gameObject))
                {
                    continue;
                }

                var instanceId = gameObject.GetInstanceID();
                if (aliveInstanceIds.Contains(instanceId))
                {
                    continue;
                }

                aliveInstanceIds.Add(instanceId);
                entities.Add(EntityReadHelpers.BuildEntity(
                    collider,
                    registry,
                    player,
                    HKRL.EntityType.Hazard,
                    HKRL.Team.Neutral,
                    baseThreat: 20.0f,
                    damage: 1,
                    flags: 1u << 0));
            }
        }

        private static bool IsLikelyHazard(GameObject gameObject)
        {
            if (EntityReadHelpers.NameContains(
                gameObject,
                "hazard",
                "spike",
                "acid",
                "lava",
                "thorn",
                "pit",
                "death"))
            {
                return true;
            }

            return gameObject.GetComponent("HazardRespawnTrigger") != null
                || gameObject.GetComponent("DamageHero") != null;
        }
    }
}
