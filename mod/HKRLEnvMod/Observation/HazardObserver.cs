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
        private readonly List<Collider2D> _sceneCandidates = new();
        private int _sceneHandle = int.MinValue;

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

            RefreshSceneCandidates();
            foreach (Collider2D collider in _sceneCandidates)
            {
                if (collider == null
                    || !collider.enabled
                    || collider.gameObject == null
                    || !collider.gameObject.activeInHierarchy)
                {
                    continue;
                }

                var gameObject = collider.gameObject;
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

        private void RefreshSceneCandidates()
        {
            var scene = UnityEngine.SceneManagement.SceneManager.GetActiveScene();
            if (scene.handle == _sceneHandle)
            {
                return;
            }

            _sceneHandle = scene.handle;
            _sceneCandidates.Clear();
            foreach (Collider2D collider in Object.FindObjectsOfType<Collider2D>())
            {
                if (collider == null
                    || collider.gameObject == null
                    || collider.gameObject.scene.handle != scene.handle
                    || !IsLikelyHazard(collider.gameObject))
                {
                    continue;
                }

                _sceneCandidates.Add(collider);
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
