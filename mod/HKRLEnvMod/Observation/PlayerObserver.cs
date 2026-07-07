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
            bool canFocus,
            bool wallSliding = false,
            bool jumping = false,
            bool falling = false,
            bool dashing = false,
            bool shadowDashing = false,
            bool invulnerable = false,
            float invulnTimer = 0.0f,
            float attackLockTimer = 0.0f,
            float castLockTimer = 0.0f,
            byte focusState = 0,
            float dashCooldown = 0.0f)
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
            WallSliding = wallSliding;
            Jumping = jumping;
            Falling = falling;
            Dashing = dashing;
            ShadowDashing = shadowDashing;
            Invulnerable = invulnerable;
            InvulnTimer = invulnTimer;
            AttackLockTimer = attackLockTimer;
            CastLockTimer = castLockTimer;
            FocusState = focusState;
            DashCooldown = dashCooldown;
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
        public bool WallSliding { get; }
        public bool Jumping { get; }
        public bool Falling { get; }
        public bool Dashing { get; }
        public bool ShadowDashing { get; }
        public bool Invulnerable { get; }
        public float InvulnTimer { get; }
        public float AttackLockTimer { get; }
        public float CastLockTimer { get; }
        public byte FocusState { get; }
        public float DashCooldown { get; }
    }

    /// <summary>
    /// Reads PlayerState from HeroController/PlayerData incl. explicit cooldown and
    /// lock timers that make the env Markovian (docs/observation_schema.md §5,
    /// PRD §9.1). Maps to HKRL.PlayerState.
    /// </summary>
    public sealed class PlayerObserver
    {
        private static Type? _playerDataType;
        private static bool _playerDataTypeSearched;

        public PlayerObservation Read()
        {
            global::HeroController? hero = global::HeroController.instance;
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
            Rigidbody2D? body = hero.GetComponent<Rigidbody2D>();
            Vector2 velocity = body != null ? body.velocity : Vector2.zero;
            sbyte facing = hero.transform.localScale.x < 0.0f ? (sbyte)(-1) : (sbyte)1;
            var playerData = FindSingleton("PlayerData", "instance");
            var hp = ReadInt(playerData, 1, "health", "Health");
            var maxHp = ReadInt(playerData, 1, "maxHealth", "MaxHealth");
            var soul = ReadInt(playerData, 0, "MPCharge", "soul", "Soul");
            var maxSoul = ReadInt(playerData, 99, "maxMP", "MaxMP", "maxSoul", "MaxSoul");
            var focusing = ReadBool(hero, false, "cState.focusing", "focusing", "Focusing");
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
                onGround: ReadBool(hero, true, "cState.onGround", "onGround", "OnGround"),
                doubleJumpAvailable: ReadBool(
                    hero,
                    true,
                    "cState.doubleJumpAvailable",
                    "doubleJumpAvailable",
                    "DoubleJumpAvailable"),
                canAttack: ReadBool(hero, true, "canAttack", "CanAttack"),
                canCast: ReadBool(hero, true, "canCast", "CanCast"),
                canFocus: ReadBool(hero, true, "canFocus", "CanFocus"),
                wallSliding: ReadBool(
                    hero,
                    false,
                    "cState.wallSliding",
                    "wallSliding",
                    "WallSliding"),
                jumping: ReadBool(hero, false, "cState.jumping", "jumping", "Jumping"),
                falling: ReadBool(hero, false, "cState.falling", "falling", "Falling"),
                dashing: ReadBool(hero, false, "cState.dashing", "dashing", "Dashing"),
                shadowDashing: ReadBool(
                    hero,
                    false,
                    "cState.shadowDashing",
                    "shadowDashing",
                    "ShadowDashing"),
                invulnerable: ReadBool(
                    hero,
                    false,
                    "cState.invulnerable",
                    "invulnerable",
                    "Invulnerable"),
                invulnTimer: ReadFloat(
                    hero,
                    0.0f,
                    "invulnerableTimer",
                    "invulnTimer",
                    "invuln_timer"),
                attackLockTimer: ReadFloat(
                    hero,
                    0.0f,
                    "attackLockTimer",
                    "attack_cooldown",
                    "attackCooldownTimer"),
                castLockTimer: ReadFloat(
                    hero,
                    0.0f,
                    "castLockTimer",
                    "cast_cooldown",
                    "spellControl.timer"),
                focusState: ClampByte(ReadInt(hero, focusing ? 1 : 0, "focusState", "FocusState")),
                dashCooldown: ReadFloat(
                    hero,
                    0.0f,
                    "dashCooldown",
                    "dashCooldownTimer",
                    "dash_cooldown"));
        }

        private static object? FindSingleton(string typeName, string memberName)
        {
            var type = FindType(typeName);
            if (type == null)
            {
                return null;
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

            return null;
        }

        private static Type? FindType(string typeName)
        {
            if (typeName == "PlayerData" && _playerDataTypeSearched)
            {
                return _playerDataType;
            }

            foreach (var assembly in AppDomain.CurrentDomain.GetAssemblies())
            {
                foreach (var type in SafeGetTypes(assembly))
                {
                    if (type.Name == typeName || type.FullName == typeName)
                    {
                        if (typeName == "PlayerData")
                        {
                            _playerDataType = type;
                            _playerDataTypeSearched = true;
                        }

                        return type;
                    }
                }
            }

            if (typeName == "PlayerData")
            {
                _playerDataType = null;
                _playerDataTypeSearched = true;
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
            if (TryReadMemberPath(target, name, out var rawValue) && TryConvertInt(rawValue, out value))
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

        private static bool ReadBool(object? target, bool fallback, params string[] names)
        {
            if (target == null)
            {
                return fallback;
            }

            foreach (var name in names)
            {
                if (TryReadMemberPath(target, name, out var value) && TryConvertBool(value, out var result))
                {
                    return result;
                }
            }

            return fallback;
        }

        private static float ReadFloat(object? target, float fallback, params string[] names)
        {
            if (target == null)
            {
                return fallback;
            }

            foreach (var name in names)
            {
                if (TryReadMemberPath(target, name, out var value) && TryConvertFloat(value, out var result))
                {
                    return result;
                }
                if (TryReadGetFloat(target, name, out var getFloatValue))
                {
                    return getFloatValue;
                }
            }

            return fallback;
        }

        private static bool TryReadGetFloat(object target, string name, out float value)
        {
            var method = target.GetType().GetMethod(
                "GetFloat",
                BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance,
                binder: null,
                types: new[] { typeof(string) },
                modifiers: null);
            if (method != null && TryConvertFloat(method.Invoke(target, new object[] { name }), out value))
            {
                return true;
            }

            value = 0.0f;
            return false;
        }

        private static bool TryReadMemberPath(object target, string path, out object? value)
        {
            value = target;
            var parts = path.Split('.');
            for (var i = 0; i < parts.Length; i++)
            {
                if (value == null || !TryReadRawMember(value, parts[i], out value))
                {
                    value = null;
                    return false;
                }
            }

            return true;
        }

        private static bool TryReadRawMember(object target, string name, out object? value)
        {
            var flags = BindingFlags.Public
                | BindingFlags.NonPublic
                | BindingFlags.Instance
                | BindingFlags.Static;
            var type = target.GetType();
            var field = type.GetField(name, flags);
            if (field != null)
            {
                try
                {
                    value = field.GetValue(target);
                    return true;
                }
                catch (Exception)
                {
                    value = null;
                    return false;
                }
            }

            var property = type.GetProperty(name, flags);
            if (property != null && property.GetIndexParameters().Length == 0)
            {
                try
                {
                    value = property.GetValue(target, null);
                    return true;
                }
                catch (Exception)
                {
                    value = null;
                    return false;
                }
            }

            value = null;
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

        private static bool TryConvertBool(object? input, out bool value)
        {
            switch (input)
            {
                case bool boolValue:
                    value = boolValue;
                    return true;
                case int intValue:
                    value = intValue != 0;
                    return true;
                case byte byteValue:
                    value = byteValue != 0;
                    return true;
                default:
                    value = false;
                    return false;
            }
        }

        private static bool TryConvertFloat(object? input, out float value)
        {
            switch (input)
            {
                case float floatValue:
                    value = floatValue;
                    return true;
                case double doubleValue:
                    value = (float)doubleValue;
                    return true;
                case int intValue:
                    value = intValue;
                    return true;
                default:
                    value = 0.0f;
                    return false;
            }
        }

        private static byte ClampByte(int value)
        {
            if (value < 0)
            {
                return 0;
            }
            if (value > byte.MaxValue)
            {
                return byte.MaxValue;
            }

            return (byte)value;
        }
    }
}
