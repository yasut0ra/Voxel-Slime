"""Agent initialization and movement for 3D Physarum-like behavior."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import SimulationConfig
from .utils import normalize_vectors


@dataclass(slots=True)
class AgentState:
    """Particle state arrays."""

    positions: np.ndarray  # (N, 3), int32
    directions: np.ndarray  # (N, 3), float32


def initialize_agents(count: int, size: int, rng: np.random.Generator) -> AgentState:
    """Create random agent positions and random unit directions."""
    positions = rng.integers(0, size, size=(count, 3), dtype=np.int32)
    directions = rng.normal(size=(count, 3)).astype(np.float32)
    directions = normalize_vectors(directions).astype(np.float32)
    return AgentState(positions=positions, directions=directions)


def step_agents(
    state: AgentState,
    trail: np.ndarray,
    cfg: SimulationConfig,
    rng: np.random.Generator,
) -> int:
    """Advance all agents by one step and deposit trail. Returns boundary-collision count."""
    size = cfg.size
    positions = state.positions
    directions = state.directions
    count = positions.shape[0]

    right, up = _orthonormal_basis(directions)
    spread = cfg.sensor_spread

    candidates = np.stack(
        [
            directions,
            directions + spread * right,
            directions - spread * right,
            directions + spread * up,
            directions - spread * up,
            directions + spread * (right + up),
            directions + spread * (-right + up),
        ],
        axis=0,
    ).astype(np.float32)
    candidates = normalize_vectors(candidates.reshape(-1, 3)).reshape(candidates.shape)

    sensor_offsets = _direction_to_step(
        candidates.reshape(-1, 3) * float(cfg.sensor_distance)
    ).reshape(candidates.shape[0], count, 3)

    sensor_positions = positions[None, :, :] + sensor_offsets
    if cfg.boundary == "wrap":
        sensor_positions %= size
    else:
        np.clip(sensor_positions, 0, size - 1, out=sensor_positions)

    sample_scores = trail[
        sensor_positions[:, :, 0],
        sensor_positions[:, :, 1],
        sensor_positions[:, :, 2],
    ]

    chosen_dirs = _select_directions(
        scores=sample_scores,
        candidates=candidates,
        temperature=cfg.softmax_temperature,
        rng=rng,
    )

    if cfg.random_turn_rate > 0.0:
        random_turn_mask = rng.random(count) < cfg.random_turn_rate
        num_random = int(np.count_nonzero(random_turn_mask))
        if num_random > 0:
            chosen_dirs[random_turn_mask] += rng.normal(
                scale=0.8, size=(num_random, 3)
            ).astype(np.float32)

    if cfg.turn_jitter > 0.0:
        chosen_dirs += rng.normal(scale=cfg.turn_jitter, size=chosen_dirs.shape).astype(
            np.float32
        )

    chosen_dirs = normalize_vectors(chosen_dirs).astype(np.float32)
    move_step = _direction_to_step(chosen_dirs)
    new_positions = positions + move_step

    collisions = 0
    if cfg.boundary == "wrap":
        new_positions %= size
    else:
        hit = (new_positions < 0) | (new_positions >= size)
        collided_agents = np.any(hit, axis=1)
        collisions = int(np.count_nonzero(collided_agents))
        if collisions > 0:
            chosen_dirs[hit] *= -1.0
            chosen_dirs[collided_agents] += rng.normal(
                scale=1.0, size=(collisions, 3)
            ).astype(np.float32)
        np.clip(new_positions, 0, size - 1, out=new_positions)
        chosen_dirs = normalize_vectors(chosen_dirs).astype(np.float32)

    state.positions = new_positions.astype(np.int32, copy=False)
    state.directions = chosen_dirs.astype(np.float32, copy=False)

    np.add.at(
        trail,
        (state.positions[:, 0], state.positions[:, 1], state.positions[:, 2]),
        cfg.deposit_amount,
    )

    return collisions


def _orthonormal_basis(direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build right/up vectors around current heading for each agent."""
    count = direction.shape[0]
    reference = np.tile(np.array([0.0, 0.0, 1.0], dtype=np.float32), (count, 1))
    parallel = np.abs(np.sum(direction * reference, axis=1)) > 0.9
    reference[parallel] = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    right = np.cross(direction, reference)
    right = normalize_vectors(right).astype(np.float32)

    up = np.cross(right, direction)
    up = normalize_vectors(up).astype(np.float32)
    return right, up


def _direction_to_step(direction: np.ndarray) -> np.ndarray:
    """Map continuous heading vectors to 26-neighborhood integer voxel steps."""
    step = np.rint(direction).astype(np.int32)
    zero_mask = np.all(step == 0, axis=1)

    if np.any(zero_mask):
        zero_rows = np.where(zero_mask)[0]
        dominant_axis = np.argmax(np.abs(direction[zero_mask]), axis=1)
        signs = np.sign(direction[zero_mask, dominant_axis]).astype(np.int32)
        signs[signs == 0] = 1
        step[zero_rows, dominant_axis] = signs

    return step


def _select_directions(
    scores: np.ndarray,
    candidates: np.ndarray,
    temperature: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Select next direction from sensed trail scores using argmax or softmax."""
    num_dirs, num_agents = scores.shape

    if temperature <= 1e-6:
        best_idx = np.argmax(scores, axis=0)
    else:
        stabilized = scores - np.max(scores, axis=0, keepdims=True)
        logits = stabilized / max(temperature, 1e-6)
        weights = np.exp(logits)
        weights /= np.sum(weights, axis=0, keepdims=True)
        cumulative = np.cumsum(weights, axis=0)
        rand = rng.random(num_agents)
        best_idx = np.sum(cumulative < rand[None, :], axis=0)
        best_idx = np.clip(best_idx, 0, num_dirs - 1)

    return candidates[best_idx, np.arange(num_agents)]
