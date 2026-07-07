namespace UnityEngine
{
    public class Object
    {
        public static void DontDestroyOnLoad(Object target) { }

        public static T[] FindObjectsOfType<T>() where T : Object
        {
            return System.Array.Empty<T>();
        }
    }

    public class Component : Object
    {
        public GameObject gameObject { get; set; } = new GameObject();
        public Transform transform => gameObject.transform;

        public T? GetComponent<T>() where T : class
        {
            return gameObject.GetComponent<T>();
        }
    }

    public class MonoBehaviour : Component
    {
    }

    public class GameObject : Object
    {
        private readonly Transform _transform;

        public GameObject(string name = "")
        {
            this.name = name;
            _transform = new Transform { gameObject = this };
        }

        public string name { get; set; }
        public bool activeInHierarchy { get; set; } = true;
        public Transform transform => _transform;

        public static GameObject? Find(string name)
        {
            return null;
        }

        public T AddComponent<T>() where T : Component, new()
        {
            return new T { gameObject = this };
        }

        public T? GetComponent<T>() where T : class
        {
            if (typeof(T) == typeof(Transform))
            {
                return _transform as T;
            }

            return null;
        }

        public Component? GetComponent(string type)
        {
            return null;
        }

        public T[] GetComponents<T>() where T : class
        {
            return System.Array.Empty<T>();
        }

        public int GetInstanceID()
        {
            return GetHashCode();
        }
    }

    public class Transform : Component
    {
        public Vector3 position { get; set; }
        public Vector3 localScale { get; set; } = new Vector3(1.0f, 1.0f, 1.0f);
        public Transform? parent { get; set; }
    }

    public class Rigidbody2D : Component
    {
        public Vector2 velocity { get; set; }
    }

    public class Collider2D : Component
    {
        public bool enabled { get; set; } = true;
        public Bounds bounds { get; set; }
    }

    public readonly struct Vector2
    {
        public Vector2(float x, float y)
        {
            this.x = x;
            this.y = y;
        }

        public float x { get; }
        public float y { get; }
        public float magnitude => Mathf.Sqrt((x * x) + (y * y));
        public static Vector2 zero => new Vector2(0.0f, 0.0f);
    }

    public readonly struct Vector3
    {
        public Vector3(float x, float y, float z = 0.0f)
        {
            this.x = x;
            this.y = y;
            this.z = z;
        }

        public float x { get; }
        public float y { get; }
        public float z { get; }
        public static Vector3 zero => new Vector3(0.0f, 0.0f, 0.0f);
    }

    public readonly struct Bounds
    {
        public Bounds(Vector3 center, Vector3 size)
        {
            this.center = center;
            this.size = size;
        }

        public Vector3 center { get; }
        public Vector3 size { get; }
    }

    public readonly struct Rect
    {
        public Rect(float x, float y, float width, float height)
        {
            this.x = x;
            this.y = y;
            this.width = width;
            this.height = height;
        }

        public float x { get; }
        public float y { get; }
        public float width { get; }
        public float height { get; }
    }

    public static class Mathf
    {
        public static float Max(float a, float b)
        {
            return a > b ? a : b;
        }

        public static float Sqrt(float value)
        {
            return (float)System.Math.Sqrt(value);
        }
    }

    public static class Time
    {
        public static float fixedDeltaTime { get; set; } = 0.02f;
        public static float timeScale { get; set; } = 1.0f;
        public static float unscaledDeltaTime { get; set; } = 0.02f;
        public static float unscaledTime { get; set; }
    }

    public static class Application
    {
        public static string persistentDataPath { get; set; } = ".";
    }

    public static class Debug
    {
        public static void Log(object message) { }
        public static void LogWarning(object message) { }
        public static void LogError(object message) { }
    }

    public static class GUI
    {
        public static void Box(Rect position, string text) { }
        public static void Label(Rect position, string text) { }
    }
}

namespace UnityEngine.SceneManagement
{
    public readonly struct Scene
    {
        public Scene(string name, bool isLoaded = true)
        {
            this.name = name;
            this.isLoaded = isLoaded;
        }

        public string name { get; }
        public bool isLoaded { get; }
    }

    public static class SceneManager
    {
        public static Scene GetActiveScene()
        {
            return new Scene("CI");
        }

        public static void LoadScene(string sceneName) { }
    }
}
