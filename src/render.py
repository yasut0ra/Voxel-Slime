"""Render trail field slices and projections to grayscale PNG frames."""

from __future__ import annotations

from pathlib import Path

import imageio.v3 as iio
import numpy as np

from .utils import ensure_dir

_AXIS_MAP = {"x": 0, "y": 1, "z": 2}


def save_frame_pair(
    trail: np.ndarray,
    step: int,
    out_dir: Path,
    slice_axis: str = "z",
    slice_index: int | None = None,
) -> tuple[Path, Path]:
    """Save a mid-slice and max-intensity projection frame for the current step."""
    frames_root = ensure_dir(Path(out_dir) / "frames")
    slice_dir = ensure_dir(frames_root / "slice")
    mip_dir = ensure_dir(frames_root / "mip")

    slice_img = render_slice(trail, axis=slice_axis, index=slice_index, mode="slice")
    mip_img = render_slice(trail, axis=slice_axis, mode="mip")

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
) -> np.ndarray:
    """Render either a single 2D slice or max-intensity projection along an axis."""
    axis_idx = _AXIS_MAP[axis]

    if mode == "slice":
        idx = trail.shape[axis_idx] // 2 if index is None else int(index)
        idx = max(0, min(trail.shape[axis_idx] - 1, idx))
        if axis_idx == 0:
            img = trail[idx, :, :]
        elif axis_idx == 1:
            img = trail[:, idx, :]
        else:
            img = trail[:, :, idx]
    elif mode == "mip":
        img = np.max(trail, axis=axis_idx)
    else:
        raise ValueError("mode must be 'slice' or 'mip'")

    return _normalize_to_uint8(img)


def _normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    """Contrast-stretch image to 8-bit grayscale using robust percentiles."""
    low, high = np.percentile(image, [1.0, 99.5])
    if high <= low:
        return np.zeros_like(image, dtype=np.uint8)
    norm = (image - low) / (high - low)
    norm = np.clip(norm, 0.0, 1.0)
    return (norm * 255.0).astype(np.uint8)
