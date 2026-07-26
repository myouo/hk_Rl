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
        private const float CandidateRefreshIntervalSeconds = 0.1f;

        private readonly List<ProjectileCandidate> _candidates = new();
        private readonly HashSet<int> _candidateIds = new();
        private int _sceneHandle = int.MinValue;
        private float _nextCandidateRefreshTime;

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

            RefreshCandidates();
            foreach (ProjectileCandidate candidate in _candidates)
            {
                Component component = candidate.Component;
                if (component == null
                    || component.gameObject == null
                    || !component.gameObject.activeInHierarchy)
                {
                    continue;
                }

                var gameObject = component.gameObject;
                var instanceId = gameObject.GetInstanceID();
                if (aliveInstanceIds.Contains(instanceId))
                {
                    continue;
                }

                aliveInstanceIds.Add(instanceId);
                entities.Add(EntityReadHelpers.BuildEntity(
                    component,
                    registry,
                    player,
                    HKRL.EntityType.Projectile,
                    candidate.Team,
                    baseThreat: 30.0f,
                    damage: candidate.Damage,
                    ttl: 1.0f,
                    flags: 1u << 3));
            }
        }

        private void RefreshCandidates()
        {
            var scene = UnityEngine.SceneManagement.SceneManager.GetActiveScene();
            float now = Time.unscaledTime;
            if (scene.handle == _sceneHandle && now < _nextCandidateRefreshTime)
            {
                return;
            }

            _sceneHandle = scene.handle;
            _nextCandidateRefreshTime = now + CandidateRefreshIntervalSeconds;
            _candidates.Clear();
            _candidateIds.Clear();

            foreach (global::DamageHero damageHero in
                Object.FindObjectsOfType<global::DamageHero>())
            {
                AddCandidate(damageHero, scene.handle);
            }
            foreach (global::DamageEnemies damageEnemies in
                Object.FindObjectsOfType<global::DamageEnemies>())
            {
                AddCandidate(damageEnemies, scene.handle);
            }
        }

        private void AddCandidate(Component component, int sceneHandle)
        {
            if (component == null || component.gameObject == null)
            {
                return;
            }

            GameObject gameObject = component.gameObject;
            if (gameObject.scene.handle != sceneHandle)
            {
                return;
            }

            int instanceId = gameObject.GetInstanceID();
            if (!_candidateIds.Add(instanceId))
            {
                return;
            }

            bool damagesEnemies = gameObject.GetComponent<global::DamageEnemies>() != null;
            bool damagesHero = gameObject.GetComponent<global::DamageHero>() != null;
            _candidates.Add(new ProjectileCandidate(
                component,
                damagesEnemies ? HKRL.Team.PlayerCreated : HKRL.Team.Enemy,
                damagesHero ? 1 : 0));
        }

        private readonly struct ProjectileCandidate
        {
            public ProjectileCandidate(Component component, HKRL.Team team, int damage)
            {
                Component = component;
                Team = team;
                Damage = damage;
            }

            public Component Component { get; }
            public HKRL.Team Team { get; }
            public int Damage { get; }
        }
    }
}
