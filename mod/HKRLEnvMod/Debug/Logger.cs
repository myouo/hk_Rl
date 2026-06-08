namespace HKRLEnvMod.Debug
{
    /// <summary>
    /// Thin logging facade over the Modding API logger with levels.
    /// All hooks log through here (wrap hook bodies in try/catch — PRD §9.9).
    /// </summary>
    public static class Logger
    {
        public static void Info(string message)
        {
            Write(LogLevel.Info, message);
        }

        public static void Warn(string message)
        {
            Write(LogLevel.Warn, message);
        }

        public static void Error(string message)
        {
            Write(LogLevel.Error, message);
        }

        public static void Error(string message, System.Exception exception)
        {
            Write(LogLevel.Error, $"{message}: {exception}");
        }

        private static void Write(LogLevel level, string message)
        {
            string text = string.IsNullOrWhiteSpace(message) ? "<empty log message>" : message;
            global::HKRLEnvMod.HKRLEnvMod? mod = global::HKRLEnvMod.HKRLEnvMod.Instance;

            if (mod != null)
            {
                switch (level)
                {
                    case LogLevel.Info:
                        mod.Log(text);
                        break;
                    case LogLevel.Warn:
                        mod.LogWarn(text);
                        break;
                    case LogLevel.Error:
                        mod.LogError(text);
                        break;
                }

                return;
            }

            string fallbackText = $"[HKRL] {text}";
            switch (level)
            {
                case LogLevel.Info:
                    UnityEngine.Debug.Log(fallbackText);
                    break;
                case LogLevel.Warn:
                    UnityEngine.Debug.LogWarning(fallbackText);
                    break;
                case LogLevel.Error:
                    UnityEngine.Debug.LogError(fallbackText);
                    break;
            }
        }

        private enum LogLevel
        {
            Info,
            Warn,
            Error
        }
    }
}
