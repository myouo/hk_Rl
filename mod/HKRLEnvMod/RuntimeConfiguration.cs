using System;
using System.Collections.Generic;
using System.IO;

namespace HKRLEnvMod
{
    /// <summary>
    /// Resolves the environment-server endpoint and token without depending on
    /// a platform-specific game launcher. Environment variables take priority.
    /// If Steam/Proton cannot inherit them, a permission-restricted
    /// hkrl-runtime.conf next to the mod DLL provides the same values.
    /// </summary>
    internal sealed class RuntimeConfiguration
    {
        internal const string AuthTokenEnv = "HKRL_AUTH_TOKEN";
        internal const string ConfigFileName = "hkrl-runtime.conf";
        internal const string HostEnv = "HKRL_HOST";
        internal const string PortEnv = "HKRL_PORT";
        internal const string SaveSlotEnv = "HKRL_SAVE_SLOT";

        private RuntimeConfiguration(
            string host,
            int port,
            string? authToken,
            int saveSlot,
            bool fileLoaded)
        {
            Host = host;
            Port = port;
            AuthToken = authToken;
            SaveSlot = saveSlot;
            FileLoaded = fileLoaded;
        }

        internal string Host { get; }
        internal int Port { get; }
        internal string? AuthToken { get; }
        internal int SaveSlot { get; }
        internal bool FileLoaded { get; }

        internal static RuntimeConfiguration Load(
            string assemblyLocation,
            string defaultHost = "127.0.0.1",
            int defaultPort = 5555,
            Func<string, string?>? getEnvironment = null,
            Action<string>? warn = null)
        {
            getEnvironment ??= Environment.GetEnvironmentVariable;
            warn ??= _ => { };

            string? assemblyDirectory = Path.GetDirectoryName(assemblyLocation);
            string configPath = Path.Combine(
                string.IsNullOrWhiteSpace(assemblyDirectory) ? "." : assemblyDirectory,
                ConfigFileName);
            Dictionary<string, string> fileValues = LoadFile(configPath, warn);

            string host = ResolveString(
                getEnvironment(HostEnv),
                fileValues,
                HostEnv,
                defaultHost);
            int port = ResolvePort(
                getEnvironment(PortEnv),
                fileValues,
                defaultPort,
                warn);
            string? authToken = ResolveOptionalString(
                getEnvironment(AuthTokenEnv),
                fileValues,
                AuthTokenEnv);
            int saveSlot = ResolveInteger(
                getEnvironment(SaveSlotEnv),
                fileValues,
                SaveSlotEnv,
                defaultValue: 1,
                minimum: 1,
                maximum: 4,
                warn);

            return new RuntimeConfiguration(
                host,
                port,
                authToken,
                saveSlot,
                fileValues.Count > 0);
        }

        private static Dictionary<string, string> LoadFile(
            string path,
            Action<string> warn)
        {
            var values = new Dictionary<string, string>(StringComparer.Ordinal);
            if (!File.Exists(path))
            {
                return values;
            }

            try
            {
                foreach (string rawLine in File.ReadAllLines(path))
                {
                    string line = rawLine.Trim();
                    if (line.Length == 0 || line.StartsWith("#", StringComparison.Ordinal))
                    {
                        continue;
                    }

                    int separator = line.IndexOf('=');
                    if (separator <= 0)
                    {
                        warn($"Ignoring malformed {ConfigFileName} line.");
                        continue;
                    }

                    string key = line.Substring(0, separator).Trim();
                    string value = line.Substring(separator + 1).Trim();
                    if (!IsSupportedKey(key))
                    {
                        warn($"Ignoring unsupported {ConfigFileName} key {key}.");
                        continue;
                    }

                    values[key] = value;
                }
            }
            catch (Exception exception)
            {
                warn($"Unable to read {ConfigFileName}: {exception.GetType().Name}.");
                values.Clear();
            }

            return values;
        }

        private static bool IsSupportedKey(string key)
        {
            return key == HostEnv
                || key == PortEnv
                || key == AuthTokenEnv
                || key == SaveSlotEnv;
        }

        private static string ResolveString(
            string? environmentValue,
            IReadOnlyDictionary<string, string> fileValues,
            string key,
            string defaultValue)
        {
            if (!string.IsNullOrWhiteSpace(environmentValue))
            {
                return environmentValue!.Trim();
            }

            return fileValues.TryGetValue(key, out string? fileValue)
                && !string.IsNullOrWhiteSpace(fileValue)
                    ? fileValue!.Trim()
                    : defaultValue;
        }

        private static string? ResolveOptionalString(
            string? environmentValue,
            IReadOnlyDictionary<string, string> fileValues,
            string key)
        {
            if (!string.IsNullOrEmpty(environmentValue))
            {
                return environmentValue;
            }

            return fileValues.TryGetValue(key, out string? fileValue)
                && !string.IsNullOrEmpty(fileValue)
                    ? fileValue
                    : null;
        }

        private static int ResolvePort(
            string? environmentValue,
            IReadOnlyDictionary<string, string> fileValues,
            int defaultPort,
            Action<string> warn)
        {
            string? value = !string.IsNullOrWhiteSpace(environmentValue)
                ? environmentValue
                : fileValues.TryGetValue(PortEnv, out string? fileValue)
                    ? fileValue
                    : null;
            if (string.IsNullOrWhiteSpace(value))
            {
                return defaultPort;
            }

            if (int.TryParse(value, out int port) && port >= 1 && port <= 65535)
            {
                return port;
            }

            warn($"Ignoring invalid {PortEnv}; using {defaultPort}.");
            return defaultPort;
        }

        private static int ResolveInteger(
            string? environmentValue,
            IReadOnlyDictionary<string, string> fileValues,
            string key,
            int defaultValue,
            int minimum,
            int maximum,
            Action<string> warn)
        {
            string? value = !string.IsNullOrWhiteSpace(environmentValue)
                ? environmentValue
                : fileValues.TryGetValue(key, out string? fileValue)
                    ? fileValue
                    : null;
            if (string.IsNullOrWhiteSpace(value))
            {
                return defaultValue;
            }

            if (int.TryParse(value, out int parsed)
                && parsed >= minimum
                && parsed <= maximum)
            {
                return parsed;
            }

            warn(
                $"Ignoring invalid {key}; using {defaultValue}. "
                + $"Expected an integer in [{minimum}, {maximum}].");
            return defaultValue;
        }
    }
}
