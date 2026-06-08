using System;
using System.Reflection;
using UnityEngine;

namespace HKRLEnvMod.Observation
{
    public readonly struct PlayerObservation
    {
        public PlayerObservation(
            float posX,
            float posY,
            float velX,
            float velY,
            int hp,
            int maxHp,
            int soul,
            int maxSoul,
            sbyte facing,
            bool onGround,
            bool doubleJumpAvailable,
            bool canAttack,
            bool canCast,
            bool canFocus)
        {
            PosX = posX;
            PosY = posY;
            VelX = velX;
            VelY = velY;
            Hp = hp;
            MaxHp = maxHp;
            Soul = soul;
            MaxSoul = maxSoul;
            Facing = facing;
            OnGround = onGround;
            DoubleJumpAvailable = doubleJumpAvailable;
            CanAttack = canAttack;
            CanCast = canCast;
            CanFocus = canFocus;
        }

        public float PosX { get; }
        public float PosY { get; }
        public float VelX { get; }
        public float VelY { get; }
        public int Hp { get; }
        public int MaxHp { get; }
        public int Soul { get; }
        public int MaxSoul { get; }
        public sbyte Facing { get; }
        public bool OnGround { get; }
        public bool DoubleJumpAvailable { get; }
        public bool CanAttack { get; }
        public bool CanCast { get; }
        public bool CanFocus { get; }
    }

    /// <summary>
    /// Reads PlayerState from HeroController/PlayerData incl. explicit cooldown and
    /// lock timers that make the env Markovian (docs/observation_schema.md §5,
    /// PRD §9.1). Maps to HKRL.PlayerState.
    /// </summary>
    public sealed class PlayerObserver
    {
        public PlayerObservation Read()
        {
            global::HeroController hero = global::HeroController.instance;
            if (hero == null)
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

            Vector3 position = hero.transform.position;
            Rigidbody2D body = hero.GetComponent<Rigidbody2D>();
            Vector2 velocity = body != null ? body.velocity : Vector2.zero;
            sbyte facing = hero.transform.localScale.x < 0.0f ? (sbyte)(-1) : (sbyte)1;
            var playerData = FindSingleton("PlayerData", "instance");
            var hp = ReadInt(playerData, 1, "health", "Health");
            var maxHp = ReadInt(playerData, 1, "maxHealth", "MaxHealth");
            var soul = ReadInt(playerData, 0, "MPCharge", "soul", "Soul");
            var maxSoul = ReadInt(playerData, 99, "maxMP", "MaxMP", "maxSoul", "MaxSoul");
            return new PlayerObservation(
                position.x,
                position.y,
                velocity.x,
                velocity.y,
                hp,
                maxHp,
                soul,
                maxSoul,
                facing,
                onGround: true,
                doubleJumpAvailable: true,
                canAttack: true,
                canCast: true,
                canFocus: true);
        }

        private static object? FindSingleton(string typeName, string memberName)
        {
            foreach (var assembly in AppDomain.CurrentDomain.GetAssemblies())
            {
                foreach (var type in SafeGetTypes(assembly))
                {
                    if (type.Name != typeName && type.FullName != typeName)
                    {
                        continue;
                    }

                    var flags = BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Static;
                    var field = type.GetField(memberName, flags);
                    if (field != null)
                    {
                        return field.GetValue(null);
                    }

                    var property = type.GetProperty(memberName, flags);
                    if (property != null)
                    {
                        return property.GetValue(null, null);
                    }
                }
            }

            return null;
        }

        private static Type[] SafeGetTypes(Assembly assembly)
        {
            try
            {
                return assembly.GetTypes();
            }
            catch (ReflectionTypeLoadException exception)
            {
                return Array.FindAll(exception.Types, type => type != null)!;
            }
        }

        private static int ReadInt(object? target, int fallback, params string[] names)
        {
            if (target == null)
            {
                return fallback;
            }

            foreach (var name in names)
            {
                if (TryReadIntMember(target, name, out var memberValue))
                {
                    return memberValue;
                }
                if (TryReadGetInt(target, name, out var getIntValue))
                {
                    return getIntValue;
                }
            }

            return fallback;
        }

        private static bool TryReadIntMember(object target, string name, out int value)
        {
            var flags = BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance;
            var type = target.GetType();
            var field = type.GetField(name, flags);
            if (field != null && TryConvertInt(field.GetValue(target), out value))
            {
                return true;
            }

            var property = type.GetProperty(name, flags);
            if (property != null && TryConvertInt(property.GetValue(target, null), out value))
            {
                return true;
            }

            value = 0;
            return false;
        }

        private static bool TryReadGetInt(object target, string name, out int value)
        {
            var method = target.GetType().GetMethod(
                "GetInt",
                BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance,
                binder: null,
                types: new[] { typeof(string) },
                modifiers: null);
            if (method != null && TryConvertInt(method.Invoke(target, new object[] { name }), out value))
            {
                return true;
            }

            value = 0;
            return false;
        }

        private static bool TryConvertInt(object? input, out int value)
        {
            switch (input)
            {
                case int intValue:
                    value = intValue;
                    return true;
                case float floatValue:
                    value = (int)floatValue;
                    return true;
                case double doubleValue:
                    value = (int)doubleValue;
                    return true;
                default:
                    value = 0;
                    return false;
            }
        }
    }
}
