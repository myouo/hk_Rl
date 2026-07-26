using System;
using System.Collections;
using System.Collections.Generic;
using System.Reflection;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace HKRLEnvMod.Observation
{
    /// <summary>
    /// Resolves live boss HealthManagers from BossSceneController first, with a
    /// conservative name fallback for scene variants whose controller has not
    /// populated its boss list yet. Reflection keeps this compatible with minor
    /// Modding API/game field-layout changes.
    /// </summary>
    internal static class BossLocator
    {
        private const BindingFlags InstanceFlags =
            BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic;
        private const BindingFlags StaticFlags =
            BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic;
        private static readonly List<global::HealthManager> BossBuffer = new();
        private static readonly HashSet<int> SeenInstanceIds = new();
        private static readonly List<global::HealthManager> ConfiguredBossBuffer = new();
        private static readonly HashSet<int> ConfiguredSeenInstanceIds = new();
        private static readonly List<global::HealthManager> SceneBossCandidateCache = new();
        private static readonly HashSet<int> SceneBossCandidateIds = new();
        private static int _candidateSceneHandle = int.MinValue;
        private static float _nextCandidateScanAt;

        public static IReadOnlyList<global::HealthManager> FindActiveBosses()
        {
            BossBuffer.Clear();
            SeenInstanceIds.Clear();
            TryReadBossSceneController(
                BossBuffer,
                SeenInstanceIds,
                requireActive: true);
            if (BossBuffer.Count > 0)
            {
                return BossBuffer;
            }

            AddSceneBossCandidates(
                BossBuffer,
                SeenInstanceIds,
                requireActive: true);
            if (BossBuffer.Count > 0)
            {
                return BossBuffer;
            }

            foreach (var health in UnityEngine.Object.FindObjectsOfType<global::HealthManager>())
            {
                if (IsActive(health) && IsLikelyBoss(health.gameObject))
                {
                    AddHealthManager(health, BossBuffer, SeenInstanceIds);
                }
            }

            return BossBuffer;
        }

        /// <summary>
        /// Returns the Boss HealthManagers registered by the active arena
        /// controller, including objects that are deliberately inactive until the
        /// Hero crosses the arena's ordinary battle trigger. This is a lifecycle
        /// check only; observations continue to expose active Bosses.
        /// </summary>
        public static IReadOnlyList<global::HealthManager> FindConfiguredBosses()
        {
            ConfiguredBossBuffer.Clear();
            ConfiguredSeenInstanceIds.Clear();
            TryReadBossSceneController(
                ConfiguredBossBuffer,
                ConfiguredSeenInstanceIds,
                requireActive: false);
            if (ConfiguredBossBuffer.Count == 0)
            {
                AddSceneBossCandidates(
                    ConfiguredBossBuffer,
                    ConfiguredSeenInstanceIds,
                    requireActive: false);
            }
            return ConfiguredBossBuffer;
        }

        private static void AddSceneBossCandidates(
            ICollection<global::HealthManager> bosses,
            ISet<int> seen,
            bool requireActive)
        {
            Scene scene =
                UnityEngine.SceneManagement.SceneManager.GetActiveScene();
            if (!scene.isLoaded)
            {
                return;
            }

            float now = Time.unscaledTime;
            bool sceneChanged = scene.handle != _candidateSceneHandle;
            bool cacheHasLiveCandidate = false;
            foreach (global::HealthManager candidate in SceneBossCandidateCache)
            {
                if (candidate != null
                    && candidate.gameObject != null
                    && candidate.gameObject.scene.handle == scene.handle)
                {
                    cacheHasLiveCandidate = true;
                    break;
                }
            }

            if (sceneChanged
                || (!cacheHasLiveCandidate && now >= _nextCandidateScanAt))
            {
                RefreshSceneBossCandidates(scene, now);
            }

            foreach (global::HealthManager candidate in SceneBossCandidateCache)
            {
                AddHealthManager(candidate, bosses, seen, requireActive);
            }
        }

        private static void RefreshSceneBossCandidates(Scene scene, float now)
        {
            _candidateSceneHandle = scene.handle;
            _nextCandidateScanAt = now + 0.5f;
            SceneBossCandidateCache.Clear();
            SceneBossCandidateIds.Clear();
            foreach (global::HealthManager health in
                     Resources.FindObjectsOfTypeAll<global::HealthManager>())
            {
                if (health == null
                    || health.gameObject == null
                    || health.gameObject.scene.handle != scene.handle)
                {
                    continue;
                }

                int instanceId = health.gameObject.GetInstanceID();
                if (SceneBossCandidateIds.Add(instanceId))
                {
                    SceneBossCandidateCache.Add(health);
                }
            }
        }

        private static void TryReadBossSceneController(
            ICollection<global::HealthManager> bosses,
            ISet<int> seen,
            bool requireActive)
        {
            try
            {
                Type? controllerType = typeof(global::HealthManager).Assembly.GetType(
                    "BossSceneController");
                if (controllerType == null)
                {
                    return;
                }

                object? controller = ReadFirstMember(
                    controllerType,
                    null,
                    StaticFlags,
                    "Instance",
                    "instance");
                if (controller == null)
                {
                    return;
                }

                object? values = ReadFirstMember(
                    controllerType,
                    controller,
                    InstanceFlags,
                    "bosses",
                    "Bosses");
                if (values is not IEnumerable enumerable)
                {
                    return;
                }

                foreach (object? value in enumerable)
                {
                    AddBossValue(value, bosses, seen, requireActive);
                }
            }
            catch (Exception exception)
            {
                global::HKRLEnvMod.Debug.Logger.Error(
                    "Failed to read BossSceneController boss list",
                    exception);
            }
        }

        private static void AddBossValue(
            object? value,
            ICollection<global::HealthManager> bosses,
            ISet<int> seen,
            bool requireActive)
        {
            switch (value)
            {
                case null:
                    return;
                case global::HealthManager health:
                    AddHealthManager(health, bosses, seen, requireActive);
                    return;
                case GameObject gameObject:
                    AddFromGameObject(gameObject, bosses, seen, requireActive);
                    return;
                case Component component:
                    AddFromGameObject(component.gameObject, bosses, seen, requireActive);
                    return;
            }

            Type type = value.GetType();
            object? nested = ReadFirstMember(
                type,
                value,
                InstanceFlags,
                "boss",
                "Boss",
                "gameObject",
                "GameObject");
            if (!ReferenceEquals(nested, value))
            {
                AddBossValue(nested, bosses, seen, requireActive);
            }
        }

        private static void AddFromGameObject(
            GameObject gameObject,
            ICollection<global::HealthManager> bosses,
            ISet<int> seen,
            bool requireActive)
        {
            if (gameObject == null)
            {
                return;
            }

            foreach (var health in gameObject.GetComponentsInChildren<global::HealthManager>(true))
            {
                AddHealthManager(health, bosses, seen, requireActive);
            }
        }

        private static void AddHealthManager(
            global::HealthManager? health,
            ICollection<global::HealthManager> bosses,
            ISet<int> seen,
            bool requireActive = true)
        {
            if (health == null
                || health.gameObject == null
                || (requireActive && !IsActive(health)))
            {
                return;
            }

            int instanceId = health.gameObject.GetInstanceID();
            if (seen.Add(instanceId))
            {
                bosses.Add(health);
            }
        }

        private static bool IsActive(global::HealthManager? health)
        {
            return health != null
                && health.isActiveAndEnabled
                && health.gameObject != null
                && health.gameObject.activeInHierarchy;
        }

        private static bool IsLikelyBoss(GameObject gameObject)
        {
            if (gameObject == null)
            {
                return false;
            }

            if (EntityReadHelpers.NameContains(
                gameObject,
                "boss",
                "hornet",
                "gruz",
                "giant fly",
                "mantis lord"))
            {
                return true;
            }

            Transform? parent = gameObject.transform.parent;
            while (parent != null)
            {
                if (EntityReadHelpers.NameContains(parent.gameObject, "boss"))
                {
                    return true;
                }

                parent = parent.parent;
            }

            return false;
        }

        private static object? ReadFirstMember(
            Type type,
            object? target,
            BindingFlags flags,
            params string[] names)
        {
            foreach (string name in names)
            {
                FieldInfo? field = type.GetField(name, flags);
                if (field != null)
                {
                    return field.GetValue(target);
                }

                PropertyInfo? property = type.GetProperty(name, flags);
                if (property != null && property.GetIndexParameters().Length == 0)
                {
                    return property.GetValue(target, null);
                }
            }

            return null;
        }
    }
}
