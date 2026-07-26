"""Versioned Hall of Gods boss catalog and diagnostic task construction.

The catalog is data, not Python conditionals: ``configs/godhome_bosses.yaml``
owns the mapping from stable boss ids to the exact ``GG_*`` scenes shipped by
the installed Hollow Knight build.  Live compatibility sweeps consume this
module without teaching the training environment about individual bosses.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable
from pathlib import Path

from pydantic import Field, model_validator

from hkrl.utils.config import (
    ActionConfig,
    ObservationConfig,
    StrictConfigModel,
    TaskConfig,
    load_yaml,
)


class GodhomeSweepDefaults(StrictConfigModel):
    """Safe task settings shared by every live compatibility probe."""

    difficulty: str = Field(default="attuned", min_length=1)
    time_limit_seconds: int = Field(default=180, ge=1)
    max_entities: int = Field(default=64, ge=1)
    action_repeat: int = Field(default=1, ge=1, le=255)
    player: dict[str, object] = Field(
        default_factory=lambda: {
            "hp": "max",
            "soul": 0,
            "charms": "default",
        }
    )


class GodhomeBossSpec(StrictConfigModel):
    """One distinct Hall of Gods fight or named boss variant."""

    boss_id: str = Field(min_length=1, pattern=r"^[a-z0-9_]+$")
    wire_id: int = Field(ge=1)
    display_name: str = Field(min_length=1)
    scene: str = Field(min_length=4, pattern=r"^GG_[A-Za-z0-9_]+$")
    variant_of: str | None = Field(default=None, pattern=r"^[a-z0-9_]+$")
    expected_min_boss_entities: int = Field(default=1, ge=1)


class GodhomeBossCatalog(StrictConfigModel):
    """Validated, versioned list used by live sweeps and future curricula."""

    catalog_version: int = Field(ge=1)
    task_defaults: GodhomeSweepDefaults = Field(default_factory=GodhomeSweepDefaults)
    bosses: list[GodhomeBossSpec] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_identity_graph(self) -> GodhomeBossCatalog:
        """Reject ambiguous ids/scenes and dangling variant relationships."""

        _ensure_unique(
            (boss.boss_id for boss in self.bosses),
            label="boss_id",
        )
        _ensure_unique(
            (boss.wire_id for boss in self.bosses),
            label="wire_id",
        )
        _ensure_unique(
            (boss.scene for boss in self.bosses),
            label="scene",
        )

        boss_ids = {boss.boss_id for boss in self.bosses}
        for boss in self.bosses:
            if boss.variant_of is None:
                continue
            if boss.variant_of == boss.boss_id:
                raise ValueError(f"{boss.boss_id}: variant_of cannot reference itself")
            if boss.variant_of not in boss_ids:
                raise ValueError(
                    f"{boss.boss_id}: variant_of references unknown boss {boss.variant_of!r}"
                )
        return self

    def make_task(self, boss: GodhomeBossSpec) -> TaskConfig:
        """Build the primitive-input diagnostic task for ``boss``."""

        if boss not in self.bosses:
            raise ValueError(f"boss {boss.boss_id!r} is not part of this catalog")
        defaults = self.task_defaults
        return TaskConfig(
            task_id=f"godhome_probe_{boss.boss_id}",
            wire_id=boss.wire_id,
            scene=boss.scene,
            difficulty=defaults.difficulty,
            time_limit_seconds=defaults.time_limit_seconds,
            player=dict(defaults.player),
            observation=ObservationConfig(
                max_entities=defaults.max_entities,
                include_fsm_state=True,
                include_hitbox=True,
                tier="privileged",
            ),
            action=ActionConfig(
                action_repeat=defaults.action_repeat,
                enable_macro_actions=False,
                n_macro_actions=0,
                expose_action_combinations=False,
            ),
        )


def load_godhome_catalog(path: str | Path) -> GodhomeBossCatalog:
    """Load and validate a Hall of Gods catalog YAML file."""

    return GodhomeBossCatalog.model_validate(load_yaml(path))


def _ensure_unique(values: Iterable[Hashable], *, label: str) -> None:
    seen: set[Hashable] = set()
    duplicates: set[Hashable] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    if duplicates:
        rendered = ", ".join(repr(value) for value in sorted(duplicates, key=str))
        raise ValueError(f"Godhome catalog contains duplicate {label}(s): {rendered}")
