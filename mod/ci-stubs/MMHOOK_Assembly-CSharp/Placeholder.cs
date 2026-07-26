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
