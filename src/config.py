"""Configuration models and validation for Voxel Slime."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Literal


@dataclass(slots=True)
class SimulationConfig:
    """Configurable parameters for the simulation."""

    size: int = 96
    agents: int = 50_000
    steps: int = 2_000
    seed: int | None = None

    out_dir: str = "outputs/run"
    save_every: int = 10

    boundary: Literal["wrap", "reflect"] = "wrap"
    slice_axis: Literal["x", "y", "z"] = "z"
    slice_index: int | None = None

    species_count: int = 3
    interaction_mode: Literal["symmetric", "cyclic"] = "cyclic"
    self_attract: float = 1.0
    cross_attract: float = -0.35

    predator_enabled: bool = True
    predator_ratio: float = 0.14
    predator_attract: float = 1.8
    predator_repel: float = -1.25
    predator_self_attract: float = 0.35
    predator_consume_amount: float = 4.0
    predator_capture_radius: int = 1
    predator_food_weight: float = 0.45
    predator_toxin_weight: float = 0.2

    food_field_enabled: bool = True
    toxin_field_enabled: bool = True
    food_weight: float = 1.25
    toxin_weight: float = 1.3
    food_sources: int = 3
    toxin_sources: int = 2
    field_source_speed: float = 0.42
    field_source_jitter: float = 0.05
    field_injection_radius: int = 2
    food_injection_amount: float = 22.0
    toxin_injection_amount: float = 18.0
    field_diffuse_rate: float = 0.34
    field_evap_rate: float = 0.06

    deposit_amount: float = 3.0
    diffuse_rate: float = 0.24
    evap_rate: float = 0.008
    max_trail: float = 255.0

    sensor_distance: int = 4
    sensor_spread: float = 0.7
    softmax_temperature: float = 0.1
    turn_jitter: float = 0.06
    random_turn_rate: float = 0.01

    render_colormap: str = "magma"
    render_gamma: float = 0.9
    env_overlay_strength: float = 0.28

    export_obj: bool = False
    obj_threshold: float = 8.0

    def validate(self) -> None:
        """Raise ValueError if settings are invalid."""
        if self.size <= 4:
            raise ValueError("size must be > 4")
        if self.agents <= 0:
            raise ValueError("agents must be > 0")
        if self.steps <= 0:
            raise ValueError("steps must be > 0")
        if self.species_count <= 0:
            raise ValueError("species_count must be > 0")
        if self.predator_enabled and self.species_count < 2:
            raise ValueError("predator_enabled requires species_count >= 2")
        if self.save_every <= 0:
            raise ValueError("save_every must be > 0")
        if self.sensor_distance <= 0:
            raise ValueError("sensor_distance must be > 0")
        if self.boundary not in {"wrap", "reflect"}:
            raise ValueError("boundary must be 'wrap' or 'reflect'")
        if self.slice_axis not in {"x", "y", "z"}:
            raise ValueError("slice_axis must be x, y, or z")
        if self.interaction_mode not in {"symmetric", "cyclic"}:
            raise ValueError("interaction_mode must be 'symmetric' or 'cyclic'")
        if not (0.0 <= self.diffuse_rate <= 1.0):
            raise ValueError("diffuse_rate must be in [0, 1]")
        if not (0.0 <= self.evap_rate < 1.0):
            raise ValueError("evap_rate must be in [0, 1)")
        if not (0.0 <= self.field_diffuse_rate <= 1.0):
            raise ValueError("field_diffuse_rate must be in [0, 1]")
        if not (0.0 <= self.field_evap_rate < 1.0):
            raise ValueError("field_evap_rate must be in [0, 1)")
        if self.max_trail <= 0.0:
            raise ValueError("max_trail must be > 0")
        if self.deposit_amount <= 0.0:
            raise ValueError("deposit_amount must be > 0")
        if self.food_sources < 0 or self.toxin_sources < 0:
            raise ValueError("food_sources and toxin_sources must be >= 0")
        if self.field_source_speed < 0.0:
            raise ValueError("field_source_speed must be >= 0")
        if self.field_source_jitter < 0.0:
            raise ValueError("field_source_jitter must be >= 0")
        if self.field_injection_radius < 0:
            raise ValueError("field_injection_radius must be >= 0")
        if self.food_injection_amount < 0.0 or self.toxin_injection_amount < 0.0:
            raise ValueError("injection amounts must be >= 0")
        if self.predator_consume_amount < 0.0:
            raise ValueError("predator_consume_amount must be >= 0")
        if self.predator_capture_radius < 0:
            raise ValueError("predator_capture_radius must be >= 0")
        if not (0.0 <= self.predator_ratio < 1.0):
            raise ValueError("predator_ratio must be in [0, 1)")
        if self.render_gamma <= 0.0:
            raise ValueError("render_gamma must be > 0")
        if not (0.0 <= self.env_overlay_strength <= 1.5):
            raise ValueError("env_overlay_strength must be in [0, 1.5]")
        if self.slice_index is not None and not (0 <= self.slice_index < self.size):
            raise ValueError("slice_index must be within grid bounds")
        if self.obj_threshold <= 0.0:
            raise ValueError("obj_threshold must be > 0")

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "SimulationConfig":
        """Create config from a plain dictionary, ignoring unknown keys."""
        valid_keys = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in raw.items() if k in valid_keys and v is not None}
        return cls(**kwargs)

    def merged(self, overrides: dict[str, Any]) -> "SimulationConfig":
        """Return a copy with non-None override values applied."""
        data = asdict(self)
        for key, value in overrides.items():
            if value is not None and key in data:
                data[key] = value
        return SimulationConfig(**data)
