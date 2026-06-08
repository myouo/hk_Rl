using System;
using System.Reflection;
using UnityEngine;

namespace HKRLEnvMod.Observation
{
    internal static class EntityReadHelpers
    {
        public static EntityObservation BuildEntity(
            Component component,
            EntityRegistry registry,
            PlayerObservation player,
            HKRL.EntityType entityType,
            HKRL.Team team,
            float baseThreat,
            int damage = 0,
            float ttl = 0.0f,
            uint flags = 0)
        {
            if (component == null)
            {
                throw new ArgumentNullException(nameof(component));
            }
            if (registry == null)
            {
                throw new ArgumentNullException(nameof(registry));
            }

            var transform = component.transform;
            var position = transform.position;
            var velocity = ReadVelocity(component.gameObject);
            var bounds = ReadBounds(component.gameObject);
            var hp = ReadIntMember(component, "hp", "HP", "health");
            var maxHp = ReadIntMember(component, "maxHp", "MaxHp", "MaxHP", "maxHealth");
            var fsmHash = ReadFsmStateHash(component.gameObject);
            var relX = position.x - player.PosX;
            var relY = position.y - player.PosY;
            var distance = Mathf.Max(0.01f, Mathf.Sqrt((relX * relX) + (relY * relY)));
            var threatScore = baseThreat + (1.0f / distance) + velocity.magnitude * 0.05f;

            return new EntityObservation(
                registry.GetStableId(component.gameObject.GetInstanceID()),
                entityType,
                team,
                prefabHash: StableHash(component.gameObject.name),
                fsmNameHash: 0,
                fsmStateHash: fsmHash,
                posX: position.x,
                posY: position.y,
                relX: relX,
                relY: relY,
                velX: velocity.x,
                velY: velocity.y,
                hp: hp,
                maxHp: maxHp,
                hurtboxCenterX: bounds.center.x,
                hurtboxCenterY: bounds.center.y,
                hurtboxSizeX: bounds.size.x,
                hurtboxSizeY: bounds.size.y,
                hitboxActive: HasEnabledCollider(component.gameObject),
                damage: damage,
                ttl: ttl,
                phase: 0,
                threatScore: threatScore,
                flags: flags);
        }

        public static int StableHash(string value)
        {
            unchecked
            {
                var hash = 23;
                foreach (var ch in value ?? string.Empty)
                {
                    hash = (hash * 31) + ch;
                }

                return hash;
            }
        }

        public static bool NameContains(GameObject gameObject, params string[] needles)
        {
            var name = gameObject == null ? string.Empty : gameObject.name;
            foreach (var needle in needles)
            {
                if (name.IndexOf(needle, StringComparison.OrdinalIgnoreCase) >= 0)
                {
                    return true;
                }
            }

            return false;
        }

        private static Vector2 ReadVelocity(GameObject gameObject)
        {
            var body = gameObject.GetComponent<Rigidbody2D>();
            return body != null ? body.velocity : Vector2.zero;
        }

        private static Bounds ReadBounds(GameObject gameObject)
        {
            var collider = gameObject.GetComponent<Collider2D>();
            if (collider != null)
            {
                return collider.bounds;
            }

            return new Bounds(gameObject.transform.position, Vector3.zero);
        }

        private static bool HasEnabledCollider(GameObject gameObject)
        {
            var colliders = gameObject.GetComponents<Collider2D>();
            foreach (var collider in colliders)
            {
                if (collider != null && collider.enabled)
                {
                    return true;
                }
            }

            return false;
        }

        private static int ReadFsmStateHash(GameObject gameObject)
        {
            var components = gameObject.GetComponents<Component>();
            foreach (var component in components)
            {
                if (component == null || component.GetType().Name != "PlayMakerFSM")
                {
                    continue;
                }

                var activeStateName = ReadFsmStateName(component);
                if (!string.IsNullOrEmpty(activeStateName))
                {
                    return StableHash(activeStateName);
                }
            }

            return 0;
        }

        private static string ReadFsmStateName(Component? fsm)
        {
            if (fsm == null)
            {
                return string.Empty;
            }

            var type = fsm.GetType();
            var activeStateName = type.GetProperty("ActiveStateName")?.GetValue(
                fsm,
                null) as string;
            if (!string.IsNullOrEmpty(activeStateName))
            {
                return activeStateName;
            }

            var activeState = type.GetProperty("ActiveState")?.GetValue(fsm, null);
            var name = activeState?.GetType().GetProperty("Name")?.GetValue(
                activeState,
                null) as string;
            return name ?? string.Empty;
        }

        private static int ReadIntMember(Component component, params string[] names)
        {
            var type = component.GetType();
            const BindingFlags flags = BindingFlags.Instance
                | BindingFlags.Public
                | BindingFlags.NonPublic;
            foreach (var name in names)
            {
                var field = type.GetField(name, flags);
                if (field != null)
                {
                    return ConvertToInt(field.GetValue(component));
                }

                var property = type.GetProperty(name, flags);
                if (property != null)
                {
                    return ConvertToInt(property.GetValue(component, null));
                }
            }

            return 0;
        }

        private static int ConvertToInt(object? value)
        {
            if (value == null)
            {
                return 0;
            }

            try
            {
                return Convert.ToInt32(value);
            }
            catch (FormatException)
            {
                return 0;
            }
            catch (InvalidCastException)
            {
                return 0;
            }
            catch (OverflowException)
            {
                return 0;
            }
        }
    }
}
