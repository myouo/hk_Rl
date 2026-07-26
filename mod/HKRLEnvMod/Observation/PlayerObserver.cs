using System;
using System.Reflection;
using HKRLEnvMod.Env;
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
            float dashCooldown = 0.0f,
            bool canDash = true,
            bool canDreamNail = true,
            bool canNailCharge = true,
            bool hasSpell = true,
            int actorStateHash = 0,
            uint actionFlags = 0,
            int spellFsmStateHash = 0,
            int dreamNailFsmStateHash = 0,
            int nailArtsFsmStateHash = 0,
            float nailChargeTimer = 0.0f,
            uint appliedInputButtons = 0)
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
            CanDash = canDash;
            CanDreamNail = canDreamNail;
            CanNailCharge = canNailCharge;
            HasSpell = hasSpell;
            ActorStateHash = actorStateHash;
            ActionFlags = actionFlags;
            SpellFsmStateHash = spellFsmStateHash;
            DreamNailFsmStateHash = dreamNailFsmStateHash;
            NailArtsFsmStateHash = nailArtsFsmStateHash;
            NailChargeTimer = nailChargeTimer;
            AppliedInputButtons = appliedInputButtons;
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
        public bool CanDash { get; }
        public bool CanDreamNail { get; }
        public bool CanNailCharge { get; }
        public bool HasSpell { get; }
        public int ActorStateHash { get; }
        public uint ActionFlags { get; }
        public int SpellFsmStateHash { get; }
        public int DreamNailFsmStateHash { get; }
        public int NailArtsFsmStateHash { get; }
        public float NailChargeTimer { get; }
        public uint AppliedInputButtons { get; }
    }

    /// <summary>
    /// Reads PlayerState from HeroController/PlayerData incl. explicit cooldown and
    /// lock timers that make the env Markovian (docs/observation_schema.md §5,
    /// PRD §9.1). Maps to HKRL.PlayerState.
    /// </summary>
    public sealed class PlayerObserver
    {
        private int _actionFsmHeroInstanceId;
        private PlayMakerFSM? _spellControlFsm;
        private PlayMakerFSM? _dreamNailFsm;
        private PlayMakerFSM? _nailArtsFsm;

        /// <summary>
        /// True only after Hollow Knight's gameplay loop will consume PlayerAction
        /// input. An active Hero can exist during scene/boss intros while the
        /// acceptingInput gate is still false.
        /// </summary>
        public static bool IsReadyForControl(global::HeroController? hero)
        {
            if (hero == null
                || hero.gameObject == null
                || !hero.gameObject.activeInHierarchy)
            {
                return false;
            }

            Rigidbody2D? body = hero.GetComponent<Rigidbody2D>();
            Collider2D? collider = hero.GetComponent<Collider2D>();
            string heroState = ReadText(hero, string.Empty, "hero_state");
            return EpisodeReadiness.IsHeroReady(
                active: hero.gameObject.activeInHierarchy,
                acceptingInput: ReadBool(
                    hero,
                    false,
                    "acceptingInput",
                    "AcceptingInput"),
                controlRelinquished: ReadBool(
                    hero,
                    true,
                    "controlReqlinquished",
                    "controlRelinquished"),
                gameplayState: !string.IsNullOrEmpty(heroState)
                    && !string.Equals(
                        heroState,
                        "no_input",
                        StringComparison.OrdinalIgnoreCase),
                transitioning: ReadBool(hero, true, "cState.transitioning"),
                transitionState: ReadText(hero, string.Empty, "transitionState"),
                hasBody: body != null,
                gravityScale: body?.gravityScale ?? 0.0f,
                bodyKinematic: body?.isKinematic ?? true,
                bodySimulated: body?.simulated ?? false,
                positionConstraintsFree: HasFreePositionConstraints(body),
                hasCollider: collider != null,
                colliderEnabled: collider?.enabled ?? false,
                tilemapTestActive: ReadBool(hero, false, "tilemapTestActive"),
                groundedJumpReady: !hero.cState.onGround
                    || ReadBool(hero, false, "CanJump"));
        }

        public static string DescribeControlReadiness(global::HeroController? hero)
        {
            if (hero == null || hero.gameObject == null)
            {
                return "hero=<unavailable>";
            }

            Rigidbody2D? body = hero.GetComponent<Rigidbody2D>();
            Collider2D? collider = hero.GetComponent<Collider2D>();
            string gravityScale =
                body == null ? "<missing>" : body.gravityScale.ToString("F3");
            string bodyKinematic =
                body == null ? "<missing>" : body.isKinematic.ToString();
            string bodySimulated =
                body == null ? "<missing>" : body.simulated.ToString();
            string bodyConstraints =
                body == null ? "<missing>" : body.constraints.ToString();
            string colliderEnabled =
                collider == null ? "<missing>" : collider.enabled.ToString();
            return "hero_active="
                + $"{hero.gameObject.activeInHierarchy}, "
                + $"accepting_input={ReadBool(hero, false, "acceptingInput", "AcceptingInput")}, "
                + "control_relinquished="
                + $"{ReadBool(hero, true, "controlReqlinquished", "controlRelinquished")}, "
                + $"hero_state={ReadText(hero, "<missing>", "hero_state")}, "
                + $"transitioning={ReadBool(hero, true, "cState.transitioning")}, "
                + $"transition_state={ReadText(hero, "<missing>", "transitionState")}, "
                + $"gravity_scale={gravityScale}, "
                + $"body_kinematic={bodyKinematic}, "
                + $"body_simulated={bodySimulated}, "
                + $"body_constraints={bodyConstraints}, "
                + $"position_constraints_free={HasFreePositionConstraints(body)}, "
                + $"collider_enabled={colliderEnabled}, "
                + $"tilemap_test_active={ReadBool(hero, false, "tilemapTestActive")}, "
                + $"can_jump={ReadBool(hero, false, "CanJump")}, "
                + $"bouncing={hero.cState.bouncing}, "
                + $"shroom_bouncing={hero.cState.shroomBouncing}";
        }

        public PlayerObservation Read(uint appliedInputButtons = 0)
        {
            global::HeroController? hero = global::HeroController.SilentInstance;
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
                    canFocus: false,
                    canDash: false,
                    canDreamNail: false,
                    canNailCharge: false,
                    hasSpell: false);
            }

            Vector3 position = hero.transform.position;
            Rigidbody2D? body = hero.GetComponent<Rigidbody2D>();
            Vector2 velocity = body != null ? body.velocity : Vector2.zero;
            sbyte facing = hero.transform.localScale.x < 0.0f ? (sbyte)(-1) : (sbyte)1;
            global::PlayerData? playerData = global::PlayerData.instance;
            global::HeroControllerStates states = hero.cState;
            var hp = playerData?.health ?? 1;
            var maxHp = playerData?.maxHealth ?? 1;
            var soul = playerData?.MPCharge ?? 0;
            var maxSoul = playerData?.maxMP ?? 99;
            var focusing = states.focusing;
            ActionTrace actionTrace = ReadActionTrace(hero);
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
                onGround: states.onGround,
                doubleJumpAvailable: ReadBool(
                    hero,
                    false,
                    "CanDoubleJump"),
                canAttack: ReadBool(hero, false, "CanAttack"),
                canCast: ReadBool(hero, false, "CanCast"),
                canFocus: ReadBool(hero, false, "CanFocus"),
                wallSliding: states.wallSliding,
                jumping: states.jumping,
                falling: states.falling,
                dashing: states.dashing,
                shadowDashing: states.shadowDashing,
                invulnerable: states.invulnerable,
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
                    "dash_cooldown"),
                canDash: ReadBool(hero, false, "CanDash"),
                canDreamNail: ReadBool(hero, false, "CanDreamNail"),
                // Do not call CanNailArt here: that method consumes/reset the
                // accumulated charge timer. CanNailCharge is a side-effect-free
                // readiness query used by the game's own input loop.
                canNailCharge: ReadBool(hero, false, "CanNailCharge"),
                hasSpell: playerData?.hasSpell ?? false,
                actorStateHash: actionTrace.ActorStateHash,
                actionFlags: actionTrace.Flags,
                spellFsmStateHash: actionTrace.SpellFsmStateHash,
                dreamNailFsmStateHash: actionTrace.DreamNailFsmStateHash,
                nailArtsFsmStateHash: actionTrace.NailArtsFsmStateHash,
                nailChargeTimer: actionTrace.NailChargeTimer,
                appliedInputButtons: appliedInputButtons);
        }

        private ActionTrace ReadActionTrace(global::HeroController hero)
        {
            EnsureActionFsmCache(hero);
            global::HeroControllerStates states = hero.cState;
            uint flags = 0;
            flags = SetFlag(flags, 0, states.attacking);
            flags = SetFlag(flags, 1, states.upAttacking);
            flags = SetFlag(flags, 2, states.downAttacking);
            flags = SetFlag(flags, 3, states.nailCharging);
            flags = SetFlag(flags, 4, ReadBool(hero, false, "nailArt_cyclone"));
            flags = SetFlag(flags, 5, states.spellQuake);
            flags = SetFlag(flags, 6, states.doubleJumping);

            return new ActionTrace(
                actorStateHash: HashState(
                    ReadText(hero, string.Empty, "hero_state")),
                flags: flags,
                spellFsmStateHash: HashFsmState(_spellControlFsm),
                dreamNailFsmStateHash: HashFsmState(_dreamNailFsm),
                nailArtsFsmStateHash: HashFsmState(_nailArtsFsm),
                nailChargeTimer: ReadFloat(hero, 0.0f, "nailChargeTimer"));
        }

        private void EnsureActionFsmCache(global::HeroController hero)
        {
            int instanceId = hero.GetInstanceID();
            if (_actionFsmHeroInstanceId == instanceId)
            {
                return;
            }

            _actionFsmHeroInstanceId = instanceId;
            _spellControlFsm = null;
            _dreamNailFsm = null;
            _nailArtsFsm = null;
            foreach (PlayMakerFSM fsm in hero.GetComponentsInChildren<PlayMakerFSM>(true))
            {
                if (fsm == null)
                {
                    continue;
                }

                if (string.Equals(
                        fsm.FsmName,
                        "Spell Control",
                        StringComparison.Ordinal))
                {
                    _spellControlFsm = fsm;
                }
                else if (string.Equals(
                             fsm.FsmName,
                             "Dream Nail",
                             StringComparison.Ordinal))
                {
                    _dreamNailFsm = fsm;
                }
                else if (string.Equals(
                             fsm.FsmName,
                             "Nail Arts",
                             StringComparison.Ordinal))
                {
                    _nailArtsFsm = fsm;
                }
            }
        }

        private static uint SetFlag(uint flags, int bit, bool enabled)
        {
            return enabled ? flags | (1u << bit) : flags;
        }

        private static int HashFsmState(PlayMakerFSM? fsm)
        {
            return fsm == null ? 0 : HashState(fsm.ActiveStateName);
        }

        private static int HashState(string? state)
        {
            return string.IsNullOrEmpty(state)
                ? 0
                : EntityReadHelpers.StableHash(state!);
        }

        private readonly struct ActionTrace
        {
            public ActionTrace(
                int actorStateHash,
                uint flags,
                int spellFsmStateHash,
                int dreamNailFsmStateHash,
                int nailArtsFsmStateHash,
                float nailChargeTimer)
            {
                ActorStateHash = actorStateHash;
                Flags = flags;
                SpellFsmStateHash = spellFsmStateHash;
                DreamNailFsmStateHash = dreamNailFsmStateHash;
                NailArtsFsmStateHash = nailArtsFsmStateHash;
                NailChargeTimer = nailChargeTimer;
            }

            public int ActorStateHash { get; }
            public uint Flags { get; }
            public int SpellFsmStateHash { get; }
            public int DreamNailFsmStateHash { get; }
            public int NailArtsFsmStateHash { get; }
            public float NailChargeTimer { get; }
        }

        private static int ReadInt(
            object? target,
            int fallback,
            string first,
            string second)
        {
            if (target == null)
            {
                return fallback;
            }

            if (TryReadIntName(target, first, out int result)
                || TryReadIntName(target, second, out result))
            {
                return result;
            }

            return fallback;
        }

        private static bool TryReadIntName(object target, string name, out int value)
        {
            if (TryReadMemberPath(target, name, out var rawValue) && TryConvertInt(rawValue, out value))
            {
                return true;
            }
            if (TryInvokeZeroArg(target, name, out var invokedValue)
                && TryConvertInt(invokedValue, out value))
            {
                return true;
            }
            if (TryReadGetInt(target, name, out value))
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

        private static bool ReadBool(object? target, bool fallback, string name)
        {
            return target != null && TryReadBoolName(target, name, out bool result)
                ? result
                : fallback;
        }

        private static bool ReadBool(
            object? target,
            bool fallback,
            string first,
            string second)
        {
            if (target != null
                && (TryReadBoolName(target, first, out bool result)
                    || TryReadBoolName(target, second, out result)))
            {
                return result;
            }

            return fallback;
        }

        private static bool ReadBool(
            object? target,
            bool fallback,
            string first,
            string second,
            string third)
        {
            if (target != null
                && (TryReadBoolName(target, first, out bool result)
                    || TryReadBoolName(target, second, out result)
                    || TryReadBoolName(target, third, out result)))
            {
                return result;
            }

            return fallback;
        }

        private static bool TryReadBoolName(
            object target,
            string name,
            out bool result)
        {
            if ((TryReadMemberPath(target, name, out object? value)
                    || TryInvokeZeroArg(target, name, out value))
                && TryConvertBool(value, out result))
            {
                return true;
            }

            result = false;
            return false;
        }

        private static float ReadFloat(object? target, float fallback, string name)
        {
            return target != null && TryReadFloatName(target, name, out float result)
                ? result
                : fallback;
        }

        private static float ReadFloat(
            object? target,
            float fallback,
            string first,
            string second,
            string third)
        {
            if (target == null)
            {
                return fallback;
            }

            if (TryReadFloatName(target, first, out float result)
                || TryReadFloatName(target, second, out result)
                || TryReadFloatName(target, third, out result))
            {
                return result;
            }

            return fallback;
        }

        private static bool TryReadFloatName(
            object target,
            string name,
            out float result)
        {
            if (TryReadMemberPath(target, name, out object? value)
                && TryConvertFloat(value, out result))
            {
                return true;
            }
            if (TryInvokeZeroArg(target, name, out value)
                && TryConvertFloat(value, out result))
            {
                return true;
            }
            return TryReadGetFloat(target, name, out result);
        }

        private static string ReadText(object? target, string fallback, string name)
        {
            if (target == null)
            {
                return fallback;
            }

            if (TryReadMemberPath(target, name, out object? value) && value != null)
            {
                return value.ToString() ?? fallback;
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

        private static bool TryInvokeZeroArg(object target, string name, out object? value)
        {
            var method = target.GetType().GetMethod(
                name,
                BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance,
                binder: null,
                types: Type.EmptyTypes,
                modifiers: null);
            if (method == null || method.ReturnType == typeof(void))
            {
                value = null;
                return false;
            }

            try
            {
                value = method.Invoke(target, Array.Empty<object>());
                return true;
            }
            catch (Exception)
            {
                value = null;
                return false;
            }
        }

        private static bool TryReadMemberPath(object target, string path, out object? value)
        {
            int separator = path.IndexOf('.');
            if (separator < 0)
            {
                return TryReadRawMember(target, path, out value);
            }

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

        private static bool HasFreePositionConstraints(Rigidbody2D? body)
        {
            if (body == null)
            {
                return false;
            }

            const RigidbodyConstraints2D positionConstraints =
                RigidbodyConstraints2D.FreezePositionX
                | RigidbodyConstraints2D.FreezePositionY;
            return (body.constraints & positionConstraints) == RigidbodyConstraints2D.None;
        }
    }
}
