"""Dynamic food and toxin fields for chemotaxis-like steering."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import SimulationConfig


@dataclass(slots=True)
class EnvironmentState:
    """State for external scalar fields and moving injection sources."""

    food: np.ndarray
    toxin: np.ndarray
    food_sources_pos: np.ndarray
    food_sources_vel: np.ndarray
    toxin_sources_pos: np.ndarray
    toxin_sources_vel: np.ndarray
    kernel_offsets: np.ndarray
    kernel_weights: np.ndarray


def initialize_environment(cfg: SimulationConfig, rng: np.random.Generator) -> EnvironmentState:
    """Initialize food/toxin fields and moving source seeds."""
    size = cfg.size
    kernel_offsets, kernel_weights = _build_injection_kernel(cfg.field_injection_radius)

    food_sources = cfg.food_sources if cfg.food_field_enabled else 0
    toxin_sources = cfg.toxin_sources if cfg.toxin_field_enabled else 0

    food_pos, food_vel = _init_sources(food_sources, size, cfg.field_source_speed, rng)
    toxin_pos, toxin_vel = _init_sources(toxin_sources, size, cfg.field_source_speed, rng)

    return EnvironmentState(
        food=np.zeros((size, size, size), dtype=np.float32),
        toxin=np.zeros((size, size, size), dtype=np.float32),
        food_sources_pos=food_pos,
        food_sources_vel=food_vel,
        toxin_sources_pos=toxin_pos,
        toxin_sources_vel=toxin_vel,
        kernel_offsets=kernel_offsets,
        kernel_weights=kernel_weights,
    )


def update_environment(
    env: EnvironmentState, cfg: SimulationConfig, rng: np.random.Generator
) -> None:
    """Advance food/toxin source trajectories and update fields in-place."""
    if cfg.food_field_enabled and env.food_sources_pos.size > 0:
        _update_single_field(
            field=env.food,
            source_pos=env.food_sources_pos,
            source_vel=env.food_sources_vel,
            injection_amount=cfg.food_injection_amount,
            cfg=cfg,
            rng=rng,
            kernel_offsets=env.kernel_offsets,
            kernel_weights=env.kernel_weights,
        )
    elif cfg.food_field_enabled:
        _diffuse_and_evaporate_scalar(env.food, cfg)

    if cfg.toxin_field_enabled and env.toxin_sources_pos.size > 0:
        _update_single_field(
            field=env.toxin,
            source_pos=env.toxin_sources_pos,
            source_vel=env.toxin_sources_vel,
            injection_amount=cfg.toxin_injection_amount,
            cfg=cfg,
            rng=rng,
            kernel_offsets=env.kernel_offsets,
            kernel_weights=env.kernel_weights,
        )
    elif cfg.toxin_field_enabled:
        _diffuse_and_evaporate_scalar(env.toxin, cfg)


def _update_single_field(
    field: np.ndarray,
    source_pos: np.ndarray,
    source_vel: np.ndarray,
    injection_amount: float,
    cfg: SimulationConfig,
    rng: np.random.Generator,
    kernel_offsets: np.ndarray,
    kernel_weights: np.ndarray,
) -> None:
    _move_sources(
        source_pos=source_pos,
        source_vel=source_vel,
        size=cfg.size,
        speed=cfg.field_source_speed,
        jitter=cfg.field_source_jitter,
        boundary=cfg.boundary,
        rng=rng,
    )
    _inject_sources(
        field=field,
        source_pos=source_pos,
        amount=injection_amount,
        kernel_offsets=kernel_offsets,
        kernel_weights=kernel_weights,
        boundary=cfg.boundary,
        size=cfg.size,
    )
    _diffuse_and_evaporate_scalar(field, cfg)


def _init_sources(
    count: int, size: int, speed: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    if count <= 0:
        empty = np.zeros((0, 3), dtype=np.float32)
        return empty.copy(), empty.copy()

    pos = rng.uniform(0.0, float(size), size=(count, 3)).astype(np.float32)
    vel = rng.normal(size=(count, 3)).astype(np.float32)
    vel = _normalize_rows(vel) * np.float32(speed)
    return pos, vel


def _move_sources(
    source_pos: np.ndarray,
    source_vel: np.ndarray,
    size: int,
    speed: float,
    jitter: float,
    boundary: str,
    rng: np.random.Generator,
) -> None:
    if source_pos.size == 0:
        return

    if jitter > 0.0:
        source_vel += rng.normal(scale=jitter, size=source_vel.shape).astype(np.float32)

    if speed > 0.0:
        source_vel[:] = _normalize_rows(source_vel) * np.float32(speed)
    else:
        source_vel.fill(0.0)

    source_pos += source_vel

    if boundary == "wrap":
        source_pos %= float(size)
        return

    low_hit = source_pos < 0.0
    high_hit = source_pos >= float(size)
    source_vel[low_hit | high_hit] *= -1.0
    source_pos[:] = np.clip(source_pos, 0.0, float(size - 1))


def _inject_sources(
    field: np.ndarray,
    source_pos: np.ndarray,
    amount: float,
    kernel_offsets: np.ndarray,
    kernel_weights: np.ndarray,
    boundary: str,
    size: int,
) -> None:
    if source_pos.size == 0 or amount <= 0.0:
        return

    for source in source_pos:
        center = np.floor(source).astype(np.int32)
        coords = center[None, :] + kernel_offsets

        if boundary == "wrap":
            coords %= size
        else:
            coords = np.clip(coords, 0, size - 1)

        np.add.at(
            field,
            (coords[:, 0], coords[:, 1], coords[:, 2]),
            amount * kernel_weights,
        )


def _diffuse_and_evaporate_scalar(field: np.ndarray, cfg: SimulationConfig) -> None:
    if cfg.field_diffuse_rate > 0.0:
        mean = _neighbor_mean_scalar(field, cfg.boundary)
        field += cfg.field_diffuse_rate * (mean - field)
    if cfg.field_evap_rate > 0.0:
        field *= 1.0 - cfg.field_evap_rate
    np.clip(field, 0.0, cfg.max_trail, out=field)


def _neighbor_mean_scalar(field: np.ndarray, boundary: str) -> np.ndarray:
    if boundary == "wrap":
        summed = field.copy()
        for axis in range(3):
            summed += np.roll(field, 1, axis=axis)
            summed += np.roll(field, -1, axis=axis)
        return summed / 7.0

    padded = np.pad(field, 1, mode="edge")
    summed = (
        padded[1:-1, 1:-1, 1:-1]
        + padded[:-2, 1:-1, 1:-1]
        + padded[2:, 1:-1, 1:-1]
        + padded[1:-1, :-2, 1:-1]
        + padded[1:-1, 2:, 1:-1]
        + padded[1:-1, 1:-1, :-2]
        + padded[1:-1, 1:-1, 2:]
    )
    return summed / 7.0


def _build_injection_kernel(radius: int) -> tuple[np.ndarray, np.ndarray]:
    if radius <= 0:
        return np.zeros((1, 3), dtype=np.int32), np.ones(1, dtype=np.float32)

    offsets: list[tuple[int, int, int]] = []
    weights: list[float] = []
    sigma = max(radius * 0.75, 0.5)

    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            for dz in range(-radius, radius + 1):
                dist2 = dx * dx + dy * dy + dz * dz
                if dist2 > radius * radius:
                    continue
                offsets.append((dx, dy, dz))
                weights.append(np.exp(-dist2 / (2.0 * sigma * sigma)))

    offset_arr = np.asarray(offsets, dtype=np.int32)
    weight_arr = np.asarray(weights, dtype=np.float32)
    weight_arr /= np.sum(weight_arr)
    return offset_arr, weight_arr


def _normalize_rows(vectors: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return vectors / norms
