"""Utility helpers for Voxel Slime."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def ensure_dir(path: Path | str) -> Path:
    """Create a directory path if it does not exist and return it."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def normalize_vectors(vectors: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Normalize a batch of vectors in-place-safe form."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return vectors / norms


def collapse_trail_channels(trail: np.ndarray) -> np.ndarray:
    """Collapse species channels to a single scalar field."""
    if trail.ndim == 3:
        return trail
    if trail.ndim == 4:
        return np.sum(trail, axis=0)
    raise ValueError(f"Unsupported trail shape: {trail.shape}")


def load_simple_yaml(path: Path | str) -> dict[str, Any]:
    """
    Load a flat key:value YAML file without external dependencies.

    Supported value types: int, float, bool, quoted string, bare string.
    """
    data: dict[str, Any] = {}
    text = Path(path).read_text(encoding="utf-8")

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue

        key, value = line.split(":", 1)
        key = key.strip()
        value = value.split("#", 1)[0].strip()
        data[key] = _parse_scalar(value)

    return data


def _parse_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None

    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]

    try:
        if any(ch in value for ch in (".", "e", "E")):
            return float(value)
        return int(value)
    except ValueError:
        return value
