using System;
using System.Reflection;

namespace HKRLEnvMod.Env
{
    /// <summary>
    /// Marks the next Hero scene-entry lifecycle as pending before a direct
    /// GameManager.BeginSceneTransition call.
    ///
    /// Hollow Knight's legacy TransitionScene path clears the private flag, but
    /// BeginSceneTransition does not. Without this narrow reset, scene-local
    /// WaitForFinishedEnteringScene actions can consume the previous scene's
    /// completed value and advance before the persistent Hero has entered the
    /// replacement scene.
    /// </summary>
    public static class SceneEntryLifecycle
    {
        private const string FinishedEntryFieldName = "hasFinishedEnteringScene";

        private static readonly FieldInfo? FinishedEntryField =
            typeof(GameManager).GetField(
                FinishedEntryFieldName,
                BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);

        public static void MarkTransitionPending(GameManager gameManager)
        {
            if (gameManager == null)
            {
                throw new ArgumentNullException(nameof(gameManager));
            }

            if (FinishedEntryField == null
                || FinishedEntryField.FieldType != typeof(bool))
            {
                throw new MissingFieldException(
                    typeof(GameManager).FullName,
                    FinishedEntryFieldName);
            }

            // This is the GameManager scene-entry handshake only. It deliberately
            // does not touch a Boss object, FSM, Transform, Rigidbody, health, or
            // animation. HeroController.FinishedEnteringScene will set it true
            // through the game's normal entry pipeline.
            FinishedEntryField.SetValue(gameManager, false);
            if (FinishedEntryField.GetValue(gameManager) is not bool isFinished
                || isFinished)
            {
                throw new InvalidOperationException(
                    "GameManager did not accept the pending scene-entry lifecycle state.");
            }
        }
    }
}
