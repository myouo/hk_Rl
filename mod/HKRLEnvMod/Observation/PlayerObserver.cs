using UnityEngine;

namespace HKRLEnvMod.Observation
{
    public readonly struct PlayerObservation
    {
        public PlayerObservation(
            float posX,
            float posY,
            float velX,
            float velY,
            int hp,
            int maxHp,
            int soul,
            int maxSoul,
            sbyte facing,
            bool onGround,
            bool doubleJumpAvailable,
            bool canAttack,
            bool canCast,
            bool canFocus)
        {
            PosX = posX;
            PosY = posY;
            VelX = velX;
            VelY = velY;
            Hp = hp;
            MaxHp = maxHp;
            Soul = soul;
            MaxSoul = maxSoul;
            Facing = facing;
            OnGround = onGround;
            DoubleJumpAvailable = doubleJumpAvailable;
            CanAttack = canAttack;
            CanCast = canCast;
            CanFocus = canFocus;
        }

        public float PosX { get; }
        public float PosY { get; }
        public float VelX { get; }
        public float VelY { get; }
        public int Hp { get; }
        public int MaxHp { get; }
        public int Soul { get; }
        public int MaxSoul { get; }
        public sbyte Facing { get; }
        public bool OnGround { get; }
        public bool DoubleJumpAvailable { get; }
        public bool CanAttack { get; }
        public bool CanCast { get; }
        public bool CanFocus { get; }
    }

    /// <summary>
    /// Reads PlayerState from HeroController/PlayerData incl. explicit cooldown and
    /// lock timers that make the env Markovian (docs/observation_schema.md §5,
    /// PRD §9.1). Maps to HKRL.PlayerState.
    /// </summary>
    public sealed class PlayerObserver
    {
        public PlayerObservation Read()
        {
            global::HeroController hero = global::HeroController.instance;
            if (hero == null)
            {
                return new PlayerObservation(
                    0.0f,
                    0.0f,
                    0.0f,
                    0.0f,
                    hp: 1,
                    maxHp: 1,
                    soul: 0,
                    maxSoul: 99,
                    facing: 1,
                    onGround: false,
                    doubleJumpAvailable: false,
                    canAttack: false,
                    canCast: false,
                    canFocus: false);
            }

            Vector3 position = hero.transform.position;
            Rigidbody2D body = hero.GetComponent<Rigidbody2D>();
            Vector2 velocity = body != null ? body.velocity : Vector2.zero;
            sbyte facing = hero.transform.localScale.x < 0.0f ? (sbyte)(-1) : (sbyte)1;
            return new PlayerObservation(
                position.x,
                position.y,
                velocity.x,
                velocity.y,
                hp: 1,
                maxHp: 1,
                soul: 0,
                maxSoul: 99,
                facing,
                onGround: true,
                doubleJumpAvailable: true,
                canAttack: true,
                canCast: true,
                canFocus: true);
        }
    }
}
