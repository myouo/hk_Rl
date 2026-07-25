using System;
using HKRLEnvMod.Action;
using InControl;
using Modding;

internal static class Program
{
    private static int Main()
    {
        try
        {
            Run();
            Console.WriteLine("InputInjectionSmoke: PASS");
            return 0;
        }
        catch (Exception exception)
        {
            Console.Error.WriteLine($"InputInjectionSmoke: FAIL: {exception}");
            return 1;
        }
    }

    private static void Run()
    {
        HeroActions actions = InputHandler.Instance.inputActions;
        var injector = new InputInjector();
        injector.Apply(
            new PrimitiveInput(
                movementX: -1,
                aimY: 1,
                buttons:
                    (1u << 0)
                    | (1u << 2)
                    | (1u << 3)
                    | (1u << 4)
                    | (1u << 5)
                    | (1u << 6)
                    | (1u << 7)));
        Raise();

        Expect(actions.left.IsPressed, "left");
        Expect(!actions.right.IsPressed, "right");
        Expect(actions.up.IsPressed, "up");
        Expect(!actions.down.IsPressed, "down");
        Expect(actions.jump.IsPressed, "jump");
        Expect(actions.dash.IsPressed, "dash");
        Expect(actions.attack.IsPressed, "attack/nail-art hold");
        Expect(actions.quickCast.IsPressed, "spell quickCast");
        Expect(actions.cast.IsPressed, "focus cast");
        Expect(actions.dreamNail.IsPressed, "dream nail");
        Expect(actions.moveVector.X == -1.0f, "moveVector.X");
        Expect(actions.moveVector.Y == 1.0f, "moveVector.Y");

        injector.Apply(new PrimitiveInput(1, -1, 1u << 8));
        Raise();
        Expect(!actions.attack.IsPressed, "nail-art release clears attack");
        Expect(actions.attack.WasReleased, "nail-art release edge");
        Expect(actions.right.IsPressed && !actions.left.IsPressed, "right movement");
        Expect(actions.down.IsPressed && !actions.up.IsPressed, "down aim");
        Expect(actions.moveVector.X == 1.0f, "updated moveVector.X");
        Expect(actions.moveVector.Y == -1.0f, "updated moveVector.Y");

        injector.Dispose();
        ExpectNeutral(actions, "dispose");

        injector.Apply(new PrimitiveInput(-1, 1, 1u << 0));
        Raise();
        ExpectNeutral(actions, "unhook");
    }

    private static void Raise()
    {
        InputManager.CurrentTick++;
        ModHooks.RaiseHeroUpdateForTests();
    }

    private static void ExpectNeutral(HeroActions actions, string context)
    {
        Expect(!actions.left.IsPressed, $"{context}: left");
        Expect(!actions.right.IsPressed, $"{context}: right");
        Expect(!actions.up.IsPressed, $"{context}: up");
        Expect(!actions.down.IsPressed, $"{context}: down");
        Expect(!actions.jump.IsPressed, $"{context}: jump");
        Expect(!actions.dash.IsPressed, $"{context}: dash");
        Expect(!actions.attack.IsPressed, $"{context}: attack");
        Expect(!actions.quickCast.IsPressed, $"{context}: quickCast");
        Expect(!actions.cast.IsPressed, $"{context}: cast");
        Expect(!actions.dreamNail.IsPressed, $"{context}: dreamNail");
        Expect(actions.moveVector.X == 0.0f, $"{context}: moveVector.X");
        Expect(actions.moveVector.Y == 0.0f, $"{context}: moveVector.Y");
    }

    private static void Expect(bool condition, string label)
    {
        if (!condition)
        {
            throw new InvalidOperationException($"failed assertion: {label}");
        }
    }
}
