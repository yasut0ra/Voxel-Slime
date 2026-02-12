"""Mesh export utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from skimage.measure import marching_cubes

from .utils import ensure_dir


def export_obj(trail: np.ndarray, threshold: float, out_path: Path | str) -> Path:
    """Extract an isosurface from the trail field and save it as OBJ."""
    out_path = Path(out_path)
    ensure_dir(out_path.parent)

    min_val = float(np.min(trail))
    max_val = float(np.max(trail))
    if not (min_val < threshold < max_val):
        raise ValueError(
            f"obj_threshold must be between trail min/max ({min_val:.4f}, {max_val:.4f}); got {threshold:.4f}"
        )

    verts, faces, normals, _ = marching_cubes(trail, level=threshold)
    _write_obj(out_path, verts, faces, normals)
    return out_path


def _write_obj(
    out_path: Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    normals: np.ndarray,
) -> None:
    """Write mesh arrays to OBJ with per-vertex normals."""
    with out_path.open("w", encoding="utf-8") as fh:
        fh.write("# Voxel Slime mesh\n")
        fh.write(f"# vertices: {len(vertices)} faces: {len(faces)}\n")

        for vx, vy, vz in vertices:
            fh.write(f"v {vx:.6f} {vy:.6f} {vz:.6f}\n")
        for nx, ny, nz in normals:
            fh.write(f"vn {nx:.6f} {ny:.6f} {nz:.6f}\n")

        faces_1based = faces + 1
        for a, b, c in faces_1based:
            fh.write(f"f {a}//{a} {b}//{b} {c}//{c}\n")
