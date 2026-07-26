using System;

namespace Modding
{
    public static class ModHooks
    {
        public static event Action? HeroUpdateHook;
        public static event Func<int, int>? TakeHealthHook;
        public static event Func<int, int>? BeforeAddHealthHook;
        public static event Action? BeforePlayerDeadHook;
        public static event Action<string>? SceneChanged;

        public static void RaiseHeroUpdateForTests()
        {
            HeroUpdateHook?.Invoke();
        }

        public static int RaiseTakeHealthForTests(int amount)
        {
            return TakeHealthHook?.Invoke(amount) ?? amount;
        }

        public static int RaiseBeforeAddHealthForTests(int amount)
        {
            return BeforeAddHealthHook?.Invoke(amount) ?? amount;
        }

        public static void RaiseBeforePlayerDeadForTests()
        {
            BeforePlayerDeadHook?.Invoke();
        }

        public static void RaiseSceneChangedForTests(string sceneName)
        {
            SceneChanged?.Invoke(sceneName);
        }
    }
}

namespace UnityEngine
{
    public static class Time
    {
        public static float unscaledDeltaTime { get; set; } = 0.02f;
        public static float fixedDeltaTime { get; set; } = 0.02f;
        public static float timeScale { get; set; } = 1.0f;
    }
}

namespace UnityEngine.SceneManagement
{
    public readonly struct Scene
    {
        public Scene(string name)
        {
            this.name = name;
        }

        public string name { get; }
    }

    public static class SceneManager
    {
        public static Scene GetActiveScene()
        {
            return new Scene("SmokeStart");
        }
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

public sealed class GameManager
{
    private bool hasFinishedEnteringScene = true;

    public bool HasFinishedEnteringScene => hasFinishedEnteringScene;
}

public static class TimeController
{
    private static float _genericTimeScale = 1.0f;

    public static float GenericTimeScale
    {
        get => _genericTimeScale;
        set
        {
            _genericTimeScale = value;
            UnityEngine.Time.timeScale = value;
        }
    }
}

namespace On
{
    public static class GameManager
    {
        public delegate void orig_SetTimeScale_float(
            global::GameManager self,
            float newTimeScale);

        public delegate void hook_SetTimeScale_float(
            orig_SetTimeScale_float orig,
            global::GameManager self,
            float newTimeScale);

        public static event hook_SetTimeScale_float? SetTimeScale_float;

        public static void RaiseSetTimeScaleForTests(
            global::GameManager self,
            float newTimeScale)
        {
            void Original(global::GameManager _, float scale)
            {
                global::TimeController.GenericTimeScale = scale;
            }

            hook_SetTimeScale_float? hook = SetTimeScale_float;
            if (hook == null)
            {
                Original(self, newTimeScale);
                return;
            }

            hook(Original, self, newTimeScale);
        }
    }
}

namespace HKRLEnvMod.Transport
{
    public readonly struct DecodedAction
    {
        public DecodedAction(
            byte movementX,
            byte aimY,
            uint buttons,
            byte durationIdx,
            short macroId)
        {
            MovementX = movementX;
            AimY = aimY;
            Buttons = buttons;
            DurationIdx = durationIdx;
            MacroId = macroId;
        }

        public byte MovementX { get; }
        public byte AimY { get; }
        public uint Buttons { get; }
        public byte DurationIdx { get; }
        public short MacroId { get; }
    }
}

namespace HKRL
{
    public enum RewardEventKind : byte
    {
        DamageDealt = 0,
        DamageTaken = 1,
        Heal = 2,
        SoulGained = 3,
        BossKilled = 4,
        PlayerDeath = 5,
        SceneChanged = 6,
        InvalidAction = 7,
        Stagger = 8,
    }
}

namespace InControl
{
    public static class InputManager
    {
        public static ulong CurrentTick { get; set; } = 1;
        public static event Action<ulong, float>? OnUpdate;

        public static void RaiseUpdateForTests(float deltaTime = 0.02f)
        {
            CurrentTick++;
            OnUpdate?.Invoke(CurrentTick, deltaTime);
        }
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
