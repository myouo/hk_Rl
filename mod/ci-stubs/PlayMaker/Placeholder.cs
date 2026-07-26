namespace PlayMaker
{
    internal static class Placeholder
    {
    }
}

public sealed class PlayMakerFSM : UnityEngine.MonoBehaviour
{
    public string FsmName { get; set; } = string.Empty;
    public string ActiveStateName { get; set; } = string.Empty;

    public static void BroadcastEvent(string eventName) { }
    public void SendEvent(string eventName) { }
}
