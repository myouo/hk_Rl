using System;
using System.Collections.Generic;
using System.Reflection;
using Modding;
using UnityEngine;

namespace HKRLEnvMod.Action
{
    public readonly struct PrimitiveInput
    {
        public PrimitiveInput(int movementX, int aimY, uint buttons)
        {
            MovementX = ClampAxis(movementX);
            AimY = ClampAxis(aimY);
            Buttons = buttons & ButtonMask;
        }

        public const uint ButtonMask = (1u << 9) - 1u;

        public int MovementX { get; }
        public int AimY { get; }
        public uint Buttons { get; }

        public static PrimitiveInput Noop => new PrimitiveInput(0, 0, 0);

        private static int ClampAxis(int value)
        {
            if (value < 0)
            {
                return -1;
            }
            if (value > 0)
            {
                return 1;
            }

            return 0;
        }
    }

    /// <summary>
    /// Injects input directly into Hollow Knight's InControl PlayerAction set.
    /// StepController updates <see cref="Current"/> in FixedUpdate; the
    /// HeroUpdateHook commits it immediately before HeroController consumes
    /// input in Update. The button layout MUST match python/hkrl/spaces.py.
    /// </summary>
    public sealed class InputInjector : IDisposable
    {
        // Button bits (mirror python/hkrl/spaces.py BUTTON_BITS):
        //  0 jump_tap  1 jump_hold  2 dash  3 attack  4 cast
        //  5 focus_hold  6 dream_nail  7 nail_art_hold  8 nail_art_release
        private const uint JumpTap = 1u << 0;
        private const uint JumpHold = 1u << 1;
        private const uint Dash = 1u << 2;
        private const uint Attack = 1u << 3;
        private const uint Cast = 1u << 4;
        private const uint FocusHold = 1u << 5;
        private const uint DreamNail = 1u << 6;
        private const uint NailArtHold = 1u << 7;
        private const uint NailArtRelease = 1u << 8;

        private readonly GameInputBridge _bridge = new GameInputBridge();
        private bool _disposed;
        private bool _loggedBridgeError;
        private bool _loggedReady;

        public InputInjector()
        {
            ModHooks.HeroUpdateHook += OnHeroUpdate;
        }

        public PrimitiveInput Current { get; private set; } = PrimitiveInput.Noop;

        /// <summary>Set the movement axis (-1/0/+1) for this tick.</summary>
        public void SetMovementX(int dir)
        {
            Current = new PrimitiveInput(dir, Current.AimY, Current.Buttons);
        }

        /// <summary>Set the aim axis (-1 down / 0 / +1 up).</summary>
        public void SetAimY(int dir)
        {
            Current = new PrimitiveInput(Current.MovementX, dir, Current.Buttons);
        }

        /// <summary>Apply the button bitmask (tap/hold/release semantics by bit).</summary>
        public void SetButtons(uint buttons)
        {
            Current = new PrimitiveInput(Current.MovementX, Current.AimY, buttons);
        }

        public void Apply(PrimitiveInput input)
        {
            Current = input;
        }

        public void Clear()
        {
            Current = PrimitiveInput.Noop;
        }

        public void Dispose()
        {
            if (_disposed)
            {
                return;
            }

            _disposed = true;
            Current = PrimitiveInput.Noop;
            try
            {
                TryCommit(Current);
            }
            finally
            {
                ModHooks.HeroUpdateHook -= OnHeroUpdate;
            }
        }

        private void OnHeroUpdate()
        {
            // Hook bodies must never let an input-version mismatch break the
            // game's Update loop (AGENTS.md §7 / PRD §9.9).
            TryCommit(Current);
        }

        private void TryCommit(PrimitiveInput input)
        {
            try
            {
                if (!_bridge.TryCommit(input))
                {
                    return;
                }

                if (!_loggedReady)
                {
                    _loggedReady = true;
                    global::HKRLEnvMod.Debug.Logger.Info(
                        "In-mod PlayerAction input injection is active.");
                }
            }
            catch (Exception exception)
            {
                if (_loggedBridgeError)
                {
                    return;
                }

                _loggedBridgeError = true;
                global::HKRLEnvMod.Debug.Logger.Error(
                    "Failed to bind Hollow Knight PlayerAction input; "
                    + "check the supported game/Modding API version",
                    exception);
            }
        }

        private sealed class GameInputBridge
        {
            private const BindingFlags InstanceFlags =
                BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic;
            private const BindingFlags StaticFlags =
                BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic;

            private static readonly (string Name, Func<PrimitiveInput, bool> State)[] Bindings =
            {
                ("left", input => input.MovementX < 0),
                ("right", input => input.MovementX > 0),
                ("down", input => input.AimY < 0),
                ("up", input => input.AimY > 0),
                ("jump", input => HasAny(input, JumpTap | JumpHold)),
                ("dash", input => HasAny(input, Dash)),
                (
                    "attack",
                    input => !HasAny(input, NailArtRelease)
                        && HasAny(input, Attack | NailArtHold)
                ),
                // Hollow Knight names the spell action quickCast and the
                // focus/heal action cast.
                ("quickCast", input => HasAny(input, Cast)),
                ("cast", input => HasAny(input, FocusHold)),
                ("dreamNail", input => HasAny(input, DreamNail))
            };

            private Type? _inputHandlerType;
            private MemberInfo? _inputHandlerInstance;
            private MemberInfo? _inputActionsMember;
            private MemberInfo? _currentTickMember;
            private object? _boundActionSet;
            private List<ActionCommitter>? _actions;
            private MethodInfo? _updateMoveVector;
            private object? _moveVector;

            public bool TryCommit(PrimitiveInput input)
            {
                ResolveRootMembers();
                object? handler = ReadMember(_inputHandlerInstance!, null);
                if (handler == null)
                {
                    return false;
                }

                object? actionSet = ReadMember(_inputActionsMember!, handler);
                if (actionSet == null)
                {
                    return false;
                }

                if (!ReferenceEquals(actionSet, _boundActionSet))
                {
                    BindActionSet(actionSet);
                }

                ulong updateTick = Convert.ToUInt64(ReadMember(_currentTickMember!, null));
                float deltaTime = Time.unscaledDeltaTime > 0.0f
                    ? Time.unscaledDeltaTime
                    : Time.fixedDeltaTime;
                List<ActionCommitter> actions = _actions!;
                for (var index = 0; index < actions.Count; index++)
                {
                    var binding = Bindings[index];
                    actions[index].Commit(binding.State(input), updateTick, deltaTime);
                }

                _updateMoveVector!.Invoke(_moveVector, new object[] { updateTick, deltaTime });
                return true;
            }

            private void ResolveRootMembers()
            {
                if (_inputHandlerType != null)
                {
                    return;
                }

                Assembly gameAssembly = typeof(global::HeroController).Assembly;
                _inputHandlerType = gameAssembly.GetType("InputHandler")
                    ?? throw new MissingMemberException(
                        gameAssembly.FullName,
                        "InputHandler");
                _inputHandlerInstance = FindMember(
                    _inputHandlerType,
                    "Instance",
                    StaticFlags);
                _inputActionsMember = FindMember(
                    _inputHandlerType,
                    "inputActions",
                    InstanceFlags);
                Type inputManagerType = gameAssembly.GetType("InControl.InputManager")
                    ?? throw new MissingMemberException(
                        gameAssembly.FullName,
                        "InControl.InputManager");
                _currentTickMember = FindMember(
                    inputManagerType,
                    "CurrentTick",
                    StaticFlags);
            }

            private void BindActionSet(object actionSet)
            {
                Type actionSetType = actionSet.GetType();
                var actions = new List<ActionCommitter>(Bindings.Length);
                for (var index = 0; index < Bindings.Length; index++)
                {
                    string name = Bindings[index].Name;
                    object action = ReadMember(
                            FindMember(actionSetType, name, InstanceFlags),
                            actionSet)
                        ?? throw new MissingMemberException(actionSetType.FullName, name);
                    actions.Add(new ActionCommitter(action));
                }

                _moveVector = ReadMember(
                        FindMember(actionSetType, "moveVector", InstanceFlags),
                        actionSet)
                    ?? throw new MissingMemberException(actionSetType.FullName, "moveVector");
                _updateMoveVector = FindMethod(
                    _moveVector.GetType(),
                    "Update",
                    typeof(ulong),
                    typeof(float));
                _actions = actions;
                _boundActionSet = actionSet;
            }

            private static bool HasAny(PrimitiveInput input, uint mask)
            {
                return (input.Buttons & mask) != 0;
            }

            private static MemberInfo FindMember(Type type, string name, BindingFlags flags)
            {
                FieldInfo? field = type.GetField(name, flags);
                if (field != null)
                {
                    return field;
                }

                PropertyInfo? property = type.GetProperty(name, flags);
                if (property != null)
                {
                    return property;
                }

                throw new MissingMemberException(type.FullName, name);
            }

            private static object? ReadMember(MemberInfo member, object? target)
            {
                if (member is FieldInfo field)
                {
                    return field.GetValue(target);
                }
                if (member is PropertyInfo property)
                {
                    return property.GetValue(target, null);
                }

                throw new InvalidOperationException(
                    $"Unsupported reflected member {member.MemberType}: {member.Name}");
            }

            private static MethodInfo FindMethod(
                Type type,
                string name,
                params Type[] parameterTypes)
            {
                for (Type? current = type; current != null; current = current.BaseType)
                {
                    MethodInfo? method = current.GetMethod(
                        name,
                        InstanceFlags | BindingFlags.DeclaredOnly,
                        binder: null,
                        types: parameterTypes,
                        modifiers: null);
                    if (method != null)
                    {
                        return method;
                    }
                }

                throw new MissingMethodException(type.FullName, name);
            }
        }

        private sealed class ActionCommitter
        {
            private readonly object _action;
            private readonly MethodInfo _commit;
            private readonly MethodInfo? _setValue;
            private readonly MethodInfo? _commitWithState;

            public ActionCommitter(object action)
            {
                _action = action;
                Type type = action.GetType();
                _commit = FindOptionalMethod(type, "Commit")
                    ?? throw new MissingMethodException(type.FullName, "Commit");
                _setValue = FindOptionalMethod(type, "SetValue", typeof(float), typeof(ulong));
                if (_setValue == null)
                {
                    _commitWithState = FindOptionalMethod(
                            type,
                            "CommitWithState",
                            typeof(bool),
                            typeof(ulong),
                            typeof(float))
                        ?? throw new MissingMethodException(type.FullName, "CommitWithState");
                }
            }

            public void Commit(bool state, ulong updateTick, float deltaTime)
            {
                if (_setValue != null)
                {
                    // SetValue replaces the physical binding's pending value
                    // in the same InControl tick; CommitWithState only ORs it.
                    _setValue.Invoke(_action, new object[] { state ? 1.0f : 0.0f, updateTick });
                    _commit.Invoke(_action, Array.Empty<object>());
                    return;
                }

                _commitWithState!.Invoke(
                    _action,
                    new object[] { state, updateTick, deltaTime });
            }

            private static MethodInfo? FindOptionalMethod(
                Type type,
                string name,
                params Type[] parameterTypes)
            {
                const BindingFlags flags =
                    BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic;
                for (Type? current = type; current != null; current = current.BaseType)
                {
                    MethodInfo? method = current.GetMethod(
                        name,
                        flags | BindingFlags.DeclaredOnly,
                        binder: null,
                        types: parameterTypes,
                        modifiers: null);
                    if (method != null)
                    {
                        return method;
                    }
                }

                return null;
            }
        }
    }
}
