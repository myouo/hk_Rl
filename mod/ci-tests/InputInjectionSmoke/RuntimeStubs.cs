using System;

namespace Modding
{
    public static class ModHooks
    {
        public static event Action? HeroUpdateHook;

        public static void RaiseHeroUpdateForTests()
        {
            HeroUpdateHook?.Invoke();
        }
    }
}

namespace UnityEngine
{
    public static class Time
    {
        public static float unscaledDeltaTime { get; set; } = 0.02f;
        public static float fixedDeltaTime { get; set; } = 0.02f;
    }
}

namespace HKRLEnvMod.Debug
{
    public static class Logger
    {
        public static void Info(string message) { }

        public static void Error(string message, Exception exception)
        {
            throw new InvalidOperationException(message, exception);
        }
    }
}

public sealed class HeroController
{
}

namespace InControl
{
    public static class InputManager
    {
        public static ulong CurrentTick { get; set; } = 1;
    }

    public class PlayerAction
    {
        private bool _lastState;
        private bool _nextState;

        public bool IsPressed { get; private set; }
        public bool WasPressed => IsPressed && !_lastState;
        public bool WasReleased => !IsPressed && _lastState;

        internal void SetValue(float value, ulong updateTick)
        {
            _nextState = value != 0.0f;
        }

        public void Commit()
        {
            _lastState = IsPressed;
            IsPressed = _nextState;
        }

        public void CommitWithState(bool state, ulong updateTick, float deltaTime)
        {
            _nextState = state;
            Commit();
        }
    }

    public class PlayerTwoAxisAction
    {
        private readonly PlayerAction _left;
        private readonly PlayerAction _right;
        private readonly PlayerAction _down;
        private readonly PlayerAction _up;

        public PlayerTwoAxisAction(
            PlayerAction left,
            PlayerAction right,
            PlayerAction down,
            PlayerAction up)
        {
            _left = left;
            _right = right;
            _down = down;
            _up = up;
        }

        public float X { get; private set; }
        public float Y { get; private set; }

        internal void Update(ulong updateTick, float deltaTime)
        {
            X = (_right.IsPressed ? 1.0f : 0.0f) - (_left.IsPressed ? 1.0f : 0.0f);
            Y = (_up.IsPressed ? 1.0f : 0.0f) - (_down.IsPressed ? 1.0f : 0.0f);
        }
    }
}

public sealed class HeroActions
{
    public readonly InControl.PlayerAction left = new InControl.PlayerAction();
    public readonly InControl.PlayerAction right = new InControl.PlayerAction();
    public readonly InControl.PlayerAction down = new InControl.PlayerAction();
    public readonly InControl.PlayerAction up = new InControl.PlayerAction();
    public readonly InControl.PlayerAction jump = new InControl.PlayerAction();
    public readonly InControl.PlayerAction dash = new InControl.PlayerAction();
    public readonly InControl.PlayerAction attack = new InControl.PlayerAction();
    public readonly InControl.PlayerAction quickCast = new InControl.PlayerAction();
    public readonly InControl.PlayerAction cast = new InControl.PlayerAction();
    public readonly InControl.PlayerAction dreamNail = new InControl.PlayerAction();
    public readonly InControl.PlayerTwoAxisAction moveVector;

    public HeroActions()
    {
        moveVector = new InControl.PlayerTwoAxisAction(left, right, down, up);
    }
}

public sealed class InputHandler
{
    public static InputHandler Instance { get; } = new InputHandler();

    public HeroActions inputActions { get; } = new HeroActions();
}
