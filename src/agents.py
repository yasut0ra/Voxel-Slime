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
    species_ids: np.ndarray  # (N,), int32
    predator_species_id: int  # -1 when disabled


def initialize_agents(cfg: SimulationConfig, rng: np.random.Generator) -> AgentState:
    """Create random agent positions, unit directions, and species labels."""
    count = cfg.agents
    size = cfg.size
    positions = rng.integers(0, size, size=(count, 3), dtype=np.int32)
    directions = rng.normal(size=(count, 3)).astype(np.float32)
    directions = normalize_vectors(directions).astype(np.float32)

    predator_species_id = -1
    if cfg.predator_enabled and cfg.species_count >= 2:
        predator_species_id = cfg.species_count - 1
        prey_species_count = cfg.species_count - 1
        species_ids = rng.integers(0, prey_species_count, size=count, dtype=np.int32)

        predator_count = int(round(count * cfg.predator_ratio))
        predator_count = min(max(predator_count, 1), count - 1)
        predator_indices = rng.choice(count, size=predator_count, replace=False)
        species_ids[predator_indices] = predator_species_id
    else:
        species_ids = rng.integers(0, cfg.species_count, size=count, dtype=np.int32)

    return AgentState(
        positions=positions,
        directions=directions,
        species_ids=species_ids,
        predator_species_id=predator_species_id,
    )


def step_agents(
    state: AgentState,
    trail: np.ndarray,
    interaction_matrix: np.ndarray,
    cfg: SimulationConfig,
    rng: np.random.Generator,
    food_field: np.ndarray | None = None,
    toxin_field: np.ndarray | None = None,
) -> tuple[int, int]:
    """Advance all agents and return (boundary_collisions, predation_count)."""
    size = cfg.size
    positions = state.positions
    directions = state.directions
    species_ids = state.species_ids
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

    sample_scores = _compute_sense_scores(
        trail=trail,
        sensor_positions=sensor_positions,
        species_ids=species_ids,
        interaction_matrix=interaction_matrix,
    )

    if food_field is not None:
        food_scores = food_field[
            sensor_positions[:, :, 0],
            sensor_positions[:, :, 1],
            sensor_positions[:, :, 2],
        ]
        sample_scores += food_scores * _agent_food_weights(state, cfg)[None, :]

    if toxin_field is not None:
        toxin_scores = toxin_field[
            sensor_positions[:, :, 0],
            sensor_positions[:, :, 1],
            sensor_positions[:, :, 2],
        ]
        sample_scores -= toxin_scores * _agent_toxin_weights(state, cfg)[None, :]

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
        (
            species_ids,
            state.positions[:, 0],
            state.positions[:, 1],
            state.positions[:, 2],
        ),
        cfg.deposit_amount,
    )

    predations = _apply_predation(state, trail, cfg, rng)
    return collisions, predations


def build_interaction_matrix(cfg: SimulationConfig) -> np.ndarray:
    """Construct species interaction weights used by sensor scoring."""
    if not cfg.predator_enabled or cfg.species_count < 2:
        return _build_base_interaction(cfg.species_count, cfg)

    predator_id = cfg.species_count - 1
    prey_count = cfg.species_count - 1
    matrix = np.zeros((cfg.species_count, cfg.species_count), dtype=np.float32)

    matrix[:prey_count, :prey_count] = _build_base_interaction(prey_count, cfg)
    matrix[:prey_count, predator_id] = cfg.predator_repel

    matrix[predator_id, :prey_count] = cfg.predator_attract
    matrix[predator_id, predator_id] = cfg.predator_self_attract
    return matrix


def _build_base_interaction(species_count: int, cfg: SimulationConfig) -> np.ndarray:
    matrix = np.full((species_count, species_count), cfg.cross_attract, dtype=np.float32)
    np.fill_diagonal(matrix, cfg.self_attract)

    if cfg.interaction_mode == "cyclic" and species_count >= 2:
        matrix.fill(0.0)
        for species in range(species_count):
            matrix[species, species] = cfg.self_attract
            matrix[species, (species - 1) % species_count] = abs(cfg.cross_attract)
            matrix[species, (species + 1) % species_count] = cfg.cross_attract
    return matrix


def _compute_sense_scores(
    trail: np.ndarray,
    sensor_positions: np.ndarray,
    species_ids: np.ndarray,
    interaction_matrix: np.ndarray,
) -> np.ndarray:
    """Sample all trail channels and reduce to directional score per agent."""
    samples = trail[
        :,
        sensor_positions[:, :, 0],
        sensor_positions[:, :, 1],
        sensor_positions[:, :, 2],
    ]
    samples = np.transpose(samples, (1, 2, 0))
    weights = interaction_matrix[species_ids]
    return np.einsum("dnk,nk->dn", samples, weights, optimize=True)


def _agent_food_weights(state: AgentState, cfg: SimulationConfig) -> np.ndarray:
    weights = np.full(state.species_ids.shape[0], cfg.food_weight, dtype=np.float32)
    if state.predator_species_id >= 0:
        predator_mask = state.species_ids == state.predator_species_id
        weights[predator_mask] = cfg.predator_food_weight
    return weights


def _agent_toxin_weights(state: AgentState, cfg: SimulationConfig) -> np.ndarray:
    weights = np.full(state.species_ids.shape[0], cfg.toxin_weight, dtype=np.float32)
    if state.predator_species_id >= 0:
        predator_mask = state.species_ids == state.predator_species_id
        weights[predator_mask] = cfg.predator_toxin_weight
    return weights


def _apply_predation(
    state: AgentState,
    trail: np.ndarray,
    cfg: SimulationConfig,
    rng: np.random.Generator,
) -> int:
    if not cfg.predator_enabled or state.predator_species_id < 0:
        return 0

    predator_mask = state.species_ids == state.predator_species_id
    prey_mask = ~predator_mask
    predator_count = int(np.count_nonzero(predator_mask))
    prey_count = int(np.count_nonzero(prey_mask))
    if predator_count == 0 or prey_count == 0:
        return 0

    predator_positions = state.positions[predator_mask]
    capture_mask = np.zeros((cfg.size, cfg.size, cfg.size), dtype=bool)
    capture_mask[
        predator_positions[:, 0],
        predator_positions[:, 1],
        predator_positions[:, 2],
    ] = True
    capture_mask = _expand_capture_mask(capture_mask, cfg.predator_capture_radius, cfg.boundary)

    prey_positions = state.positions[prey_mask]
    hunted_local = capture_mask[
        prey_positions[:, 0],
        prey_positions[:, 1],
        prey_positions[:, 2],
    ]
    predations = int(np.count_nonzero(hunted_local))
    if predations == 0:
        return 0

    prey_indices = np.flatnonzero(prey_mask)
    hunted_indices = prey_indices[np.where(hunted_local)[0]]

    state.positions[hunted_indices] = rng.integers(
        0, cfg.size, size=(predations, 3), dtype=np.int32
    )
    new_dirs = rng.normal(size=(predations, 3)).astype(np.float32)
    state.directions[hunted_indices] = normalize_vectors(new_dirs).astype(np.float32)

    if cfg.predator_consume_amount > 0.0 and state.predator_species_id > 0:
        for prey_species in range(state.predator_species_id):
            np.add.at(
                trail[prey_species],
                (
                    predator_positions[:, 0],
                    predator_positions[:, 1],
                    predator_positions[:, 2],
                ),
                -cfg.predator_consume_amount,
            )
        np.clip(trail, 0.0, cfg.max_trail, out=trail)

    return predations


def _expand_capture_mask(mask: np.ndarray, radius: int, boundary: str) -> np.ndarray:
    if radius <= 0:
        return mask

    expanded = mask.copy()
    for _ in range(radius):
        if boundary == "wrap":
            grown = expanded.copy()
            for axis in range(3):
                grown |= np.roll(expanded, 1, axis=axis)
                grown |= np.roll(expanded, -1, axis=axis)
        else:
            padded = np.pad(expanded, 1, mode="edge")
            grown = (
                padded[1:-1, 1:-1, 1:-1]
                | padded[:-2, 1:-1, 1:-1]
                | padded[2:, 1:-1, 1:-1]
                | padded[1:-1, :-2, 1:-1]
                | padded[1:-1, 2:, 1:-1]
                | padded[1:-1, 1:-1, :-2]
                | padded[1:-1, 1:-1, 2:]
            )
        expanded = grown
    return expanded


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
