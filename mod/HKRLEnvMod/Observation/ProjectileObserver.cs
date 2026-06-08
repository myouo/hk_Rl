namespace HKRLEnvMod.Observation
{
    using System.Collections.Generic;
    using UnityEngine;

    /// <summary>
    /// Reads projectiles/bullets (pos, vel, ttl, damage, hitbox). Feeds top-k
    /// threat filtering for bullet-hell phases (docs/model_architecture.md §3).
    /// </summary>
    public sealed class ProjectileObserver
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

            foreach (var transform in Object.FindObjectsOfType<Transform>())
            {
                if (transform == null
                    || transform.gameObject == null
                    || !transform.gameObject.activeInHierarchy)
                {
                    continue;
                }

                var gameObject = transform.gameObject;
                if (!IsLikelyProjectile(gameObject))
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
                    transform,
                    registry,
                    player,
                    HKRL.EntityType.Projectile,
                    GuessTeam(gameObject),
                    baseThreat: 30.0f,
                    damage: GuessDamage(gameObject),
                    ttl: 1.0f,
                    flags: 1u << 3));
            }
        }

        private static bool IsLikelyProjectile(GameObject gameObject)
        {
            if (EntityReadHelpers.NameContains(
                gameObject,
                "projectile",
                "bullet",
                "shot",
                "spike",
                "orb",
                "fireball",
                "acid"))
            {
                return true;
            }

            return gameObject.GetComponent("Projectile") != null
                || gameObject.GetComponent("DamageEnemies") != null
                || gameObject.GetComponent("DamageHero") != null;
        }

        private static HKRL.Team GuessTeam(GameObject gameObject)
        {
            return gameObject.GetComponent("DamageEnemies") != null
                ? HKRL.Team.PlayerCreated
                : HKRL.Team.Enemy;
        }

        private static int GuessDamage(GameObject gameObject)
        {
            return gameObject.GetComponent("DamageHero") != null ? 1 : 0;
        }
    }
}
