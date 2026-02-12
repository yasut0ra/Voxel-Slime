"""Render trail field slices and projections to color PNG frames."""

from __future__ import annotations

from pathlib import Path

import imageio.v3 as iio
import numpy as np
from matplotlib import cm
from matplotlib import colors as mcolors

from .utils import ensure_dir

_AXIS_MAP = {"x": 0, "y": 1, "z": 2}
_BASE_PALETTE_HEX = [
    "#2de2e6",
    "#ff3864",
    "#ffd300",
    "#9d4edd",
    "#06d6a0",
    "#ff8fab",
]


def save_frame_pair(
    trail: np.ndarray,
    step: int,
    out_dir: Path,
    slice_axis: str = "z",
    slice_index: int | None = None,
    colormap: str = "magma",
    gamma: float = 0.9,
    food_field: np.ndarray | None = None,
    toxin_field: np.ndarray | None = None,
    env_overlay_strength: float = 0.28,
) -> tuple[Path, Path]:
    """Save a color mid-slice and color max-intensity projection for the current step."""
    frames_root = ensure_dir(Path(out_dir) / "frames")
    slice_dir = ensure_dir(frames_root / "slice")
    mip_dir = ensure_dir(frames_root / "mip")

    slice_img = render_slice(
        trail,
        axis=slice_axis,
        index=slice_index,
        mode="slice",
        colormap=colormap,
        gamma=gamma,
        food_field=food_field,
        toxin_field=toxin_field,
        env_overlay_strength=env_overlay_strength,
    )
    mip_img = render_slice(
        trail,
        axis=slice_axis,
        mode="mip",
        colormap=colormap,
        gamma=gamma,
        food_field=food_field,
        toxin_field=toxin_field,
        env_overlay_strength=env_overlay_strength,
    )

    slice_path = slice_dir / f"slice_{step:06d}.png"
    mip_path = mip_dir / f"mip_{step:06d}.png"
    iio.imwrite(slice_path, slice_img)
    iio.imwrite(mip_path, mip_img)
    return slice_path, mip_path


def render_slice(
    trail: np.ndarray,
    axis: str = "z",
    index: int | None = None,
    mode: str = "slice",
    colormap: str = "magma",
    gamma: float = 0.9,
    food_field: np.ndarray | None = None,
    toxin_field: np.ndarray | None = None,
    env_overlay_strength: float = 0.28,
) -> np.ndarray:
    """Render either a single 2D slice or max-intensity projection along an axis."""
    axis_idx = _AXIS_MAP[axis]
    species_count = 1 if trail.ndim == 3 else trail.shape[0]
    spatial_axis = axis_idx if trail.ndim == 3 else axis_idx + 1

    if mode == "slice":
        idx = trail.shape[spatial_axis] // 2 if index is None else int(index)
        idx = max(0, min(trail.shape[spatial_axis] - 1, idx))
        field = _extract_slice(trail, axis_idx, idx)
    elif mode == "mip":
        field = np.max(trail, axis=spatial_axis)
    else:
        raise ValueError("mode must be 'slice' or 'mip'")

    if species_count == 1:
        if field.ndim == 3:
            field = field[0]
        rgb = _apply_colormap(field, colormap=colormap, gamma=gamma)
    else:
        rgb = _compose_multispecies_rgb(field, gamma=gamma)

    if food_field is None and toxin_field is None:
        return rgb

    food_proj = _extract_scalar(food_field, axis_idx, mode, index) if food_field is not None else None
    toxin_proj = (
        _extract_scalar(toxin_field, axis_idx, mode, index) if toxin_field is not None else None
    )
    return _overlay_environment(
        rgb=rgb,
        food=food_proj,
        toxin=toxin_proj,
        strength=env_overlay_strength,
    )


def _extract_slice(trail: np.ndarray, axis_idx: int, index: int) -> np.ndarray:
    """Extract a single slice from scalar or channelized fields."""
    if trail.ndim == 3:
        if axis_idx == 0:
            return trail[index, :, :]
        if axis_idx == 1:
            return trail[:, index, :]
        return trail[:, :, index]

    if axis_idx == 0:
        return trail[:, index, :, :]
    if axis_idx == 1:
        return trail[:, :, index, :]
    return trail[:, :, :, index]


def _extract_scalar(
    field: np.ndarray,
    axis_idx: int,
    mode: str,
    index: int | None,
) -> np.ndarray:
    """Extract scalar slice or projection from a 3D field."""
    if mode == "slice":
        idx = field.shape[axis_idx] // 2 if index is None else int(index)
        idx = max(0, min(field.shape[axis_idx] - 1, idx))
        if axis_idx == 0:
            return field[idx, :, :]
        if axis_idx == 1:
            return field[:, idx, :]
        return field[:, :, idx]
    return np.max(field, axis=axis_idx)


def _apply_colormap(image: np.ndarray, colormap: str, gamma: float) -> np.ndarray:
    """Map a scalar image to RGB with robust normalization and gamma shaping."""
    norm = _robust_normalize(image)
    mapped = cm.get_cmap(colormap)(norm)[..., :3]
    mapped = np.clip(mapped, 0.0, 1.0) ** (1.0 / max(gamma, 1e-6))
    return (mapped * 255.0).astype(np.uint8)


def _compose_multispecies_rgb(field: np.ndarray, gamma: float) -> np.ndarray:
    """Blend species channels into an RGB image with vivid additive mixing."""
    if field.ndim != 3:
        raise ValueError(f"Expected species field shape (S, H, W), got {field.shape}")

    species_count = field.shape[0]
    weights = np.stack([_robust_normalize(field[idx]) for idx in range(species_count)], axis=0)
    palette = _build_palette(species_count)

    rgb = np.einsum("shw,sc->hwc", weights, palette, optimize=True)
    rgb = 1.0 - np.exp(-1.75 * rgb)
    rgb = np.clip(rgb, 0.0, 1.0) ** (1.0 / max(gamma, 1e-6))
    return (rgb * 255.0).astype(np.uint8)


def _overlay_environment(
    rgb: np.ndarray,
    food: np.ndarray | None,
    toxin: np.ndarray | None,
    strength: float,
) -> np.ndarray:
    """Overlay food/toxin scalar fields with cyan/orange accents."""
    if strength <= 0.0:
        return rgb

    rgbf = rgb.astype(np.float32) / 255.0
    if food is not None:
        food_n = _robust_normalize(food)[..., None]
        rgbf += strength * food_n * np.array([0.08, 0.95, 0.65], dtype=np.float32)
    if toxin is not None:
        toxin_n = _robust_normalize(toxin)[..., None]
        rgbf += strength * toxin_n * np.array([1.0, 0.35, 0.08], dtype=np.float32)

    rgbf = np.clip(rgbf, 0.0, 1.0)
    return (rgbf * 255.0).astype(np.uint8)


def _robust_normalize(image: np.ndarray) -> np.ndarray:
    """Normalize using robust percentiles to stabilize contrast over time."""
    low, high = np.percentile(image, [1.0, 99.5])
    if high <= low:
        return np.zeros_like(image, dtype=np.float32)
    norm = (image - low) / (high - low)
    return np.clip(norm, 0.0, 1.0).astype(np.float32)


def _build_palette(species_count: int) -> np.ndarray:
    """Return RGB colors in [0,1] for each species."""
    if species_count <= len(_BASE_PALETTE_HEX):
        return np.array(
            [mcolors.to_rgb(color) for color in _BASE_PALETTE_HEX[:species_count]],
            dtype=np.float32,
        )

    hsv = np.zeros((species_count, 3), dtype=np.float32)
    hsv[:, 0] = np.linspace(0.0, 1.0, species_count, endpoint=False)
    hsv[:, 1] = 0.78
    hsv[:, 2] = 1.0
    return mcolors.hsv_to_rgb(hsv).astype(np.float32)
