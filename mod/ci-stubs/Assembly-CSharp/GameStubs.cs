using System;
using UnityEngine;

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

    public class Mod
    {
        public Mod(string name)
        {
            Name = name;
        }

        public string Name { get; }

        public virtual string GetVersion()
        {
            return "ci";
        }

        public virtual void Initialize() { }

        public void Log(string text) { }
        public void LogWarn(string text) { }
        public void LogError(string text) { }
    }
}

public sealed class HeroController : MonoBehaviour
{
    public static HeroController? instance { get; set; }
    public static HeroController? SilentInstance => instance;
    public bool enterWithoutInput;
    public HeroControllerStates cState { get; } = new HeroControllerStates();

    public void ClearMPSendEvents() { }
    public void MaxHealth() { }
    public void EnterWithoutInput(bool flag)
    {
        enterWithoutInput = flag;
    }
    public void AcceptInput() { }
    public void RegainControl() { }
    public void StartAnimationControl() { }
}

public sealed class HeroControllerStates
{
    public bool attacking;
    public bool upAttacking;
    public bool downAttacking;
    public bool nailCharging;
    public bool spellQuake;
    public bool doubleJumping;
    public bool focusing;
    public bool onGround;
    public bool wallSliding;
    public bool jumping;
    public bool falling;
    public bool dashing;
    public bool shadowDashing;
    public bool invulnerable;
    public bool bouncing;
    public bool shroomBouncing;
}

public sealed class GameManager : MonoBehaviour
{
    public static GameManager? instance { get; set; }

    private bool hasFinishedEnteringScene = true;

    public bool IsInSceneTransition { get; set; }
    public bool IsLoadingSceneTransition { get; set; }
    public bool HasFinishedEnteringScene => hasFinishedEnteringScene;

    public bool IsMenuScene()
    {
        return false;
    }

    public void LoadGameFromUI(int saveSlot) { }

    public void BeginSceneTransition(SceneLoadInfo info) { }
    public void TimePasses() { }
    public void ResetSemiPersistentItems() { }

    public sealed class SceneLoadInfo
    {
        public string SceneName = string.Empty;
        public string EntryGateName = string.Empty;
        public float EntryDelay;
        public SceneLoadVisualizations Visualization;
        public bool PreventCameraFadeOut;
        public bool WaitForSceneTransitionCameraFade;
        public bool AlwaysUnloadUnusedAssets;
    }

    public enum SceneLoadVisualizations
    {
        Default = 0,
        GodsAndGlory = 5,
    }
}

public sealed class PlayerData
{
    public static PlayerData? instance { get; set; }
    public string dreamReturnScene = string.Empty;
    public bool atBench;
    public bool disablePause;
    public int health = 1;
    public int maxHealth = 1;
    public int maxMP = 99;
    public int MPCharge;
    public bool hasSpell;
}

public static class StaticVariableList
{
    public static void SetValue<T>(string name, T value) { }
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

public sealed class BossSceneController : MonoBehaviour
{
    public delegate void SetupEventDelegate(BossSceneController controller);

    public static BossSceneController? Instance { get; set; }
    public static SetupEventDelegate? SetupEvent { get; set; }
    public event Action? OnBossSceneComplete;

    public bool HasTransitionedIn { get; set; }
    public int BossLevel { get; set; }
    public string DreamReturnEvent { get; set; } = string.Empty;

    public void DoDreamReturn() { }

    public void RaiseBossSceneCompleteForTests()
    {
        OnBossSceneComplete?.Invoke();
    }
}

public sealed class HealthManager : MonoBehaviour
{
    public bool isActiveAndEnabled { get; set; } = true;
    public int hp { get; set; }
    public int maxHp { get; set; }
}

public sealed class DamageHero : MonoBehaviour
{
}

public sealed class DamageEnemies : MonoBehaviour
{
}

namespace InControl
{
    public static class InputManager
    {
        public static ulong CurrentTick { get; set; }
        public static event Action<ulong, float>? OnUpdate;

        public static void RaiseUpdateForTests(float deltaTime = 0.02f)
        {
            CurrentTick++;
            OnUpdate?.Invoke(CurrentTick, deltaTime);
        }
    }

    public class PlayerAction
    {
        internal void SetValue(float value, ulong updateTick) { }
        public void Commit() { }
        public void CommitWithState(bool state, ulong updateTick, float deltaTime) { }
    }

    public class PlayerTwoAxisAction
    {
        internal void Update(ulong updateTick, float deltaTime) { }
    }
}

public sealed class HeroActions
{
    public InControl.PlayerAction left { get; } = new InControl.PlayerAction();
    public InControl.PlayerAction right { get; } = new InControl.PlayerAction();
    public InControl.PlayerAction down { get; } = new InControl.PlayerAction();
    public InControl.PlayerAction up { get; } = new InControl.PlayerAction();
    public InControl.PlayerAction jump { get; } = new InControl.PlayerAction();
    public InControl.PlayerAction dash { get; } = new InControl.PlayerAction();
    public InControl.PlayerAction attack { get; } = new InControl.PlayerAction();
    public InControl.PlayerAction quickCast { get; } = new InControl.PlayerAction();
    public InControl.PlayerAction cast { get; } = new InControl.PlayerAction();
    public InControl.PlayerAction dreamNail { get; } = new InControl.PlayerAction();
    public InControl.PlayerTwoAxisAction moveVector { get; } =
        new InControl.PlayerTwoAxisAction();
}

public sealed class InputHandler
{
    public static InputHandler? Instance { get; set; }
    public HeroActions? inputActions { get; set; }
}
