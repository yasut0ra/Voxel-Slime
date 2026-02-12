"""Trail field diffusion and evaporation."""

from __future__ import annotations

import numpy as np

from .config import SimulationConfig


def diffuse_and_evaporate(trail: np.ndarray, cfg: SimulationConfig) -> None:
    """Apply one diffusion + evaporation update in-place."""
    if cfg.diffuse_rate > 0.0:
        neighbor_mean = _neighbor_mean(trail, boundary=cfg.boundary)
        trail += cfg.diffuse_rate * (neighbor_mean - trail)

    if cfg.evap_rate > 0.0:
        trail *= 1.0 - cfg.evap_rate

    np.clip(trail, 0.0, cfg.max_trail, out=trail)


def _neighbor_mean(trail: np.ndarray, boundary: str) -> np.ndarray:
    """Compute mean over center + 6 axis neighbors."""
    if boundary == "wrap":
        summed = trail.copy()
        for axis in range(3):
            summed += np.roll(trail, 1, axis=axis)
            summed += np.roll(trail, -1, axis=axis)
        return summed / 7.0

    padded = np.pad(trail, 1, mode="edge")
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
