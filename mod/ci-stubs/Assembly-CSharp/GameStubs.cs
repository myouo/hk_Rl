using UnityEngine;

namespace Modding
{
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
}

public sealed class HealthManager : MonoBehaviour
{
    public bool isActiveAndEnabled { get; set; } = true;
    public int hp { get; set; }
    public int maxHp { get; set; }
}
