#!/usr/bin/env python3
"""Create a rotating GIF preview from an OBJ mesh."""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render turntable GIF from OBJ mesh")
    parser.add_argument("--obj", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--frames", default=72, type=int)
    parser.add_argument("--fps", default=24, type=int)
    parser.add_argument("--elev", default=24.0, type=float)
    parser.add_argument("--dpi", default=120, type=int)
    parser.add_argument("--size", default=6.0, type=float, help="Figure size in inches")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    vertices, faces = load_obj(args.obj)
    if len(vertices) == 0 or len(faces) == 0:
        raise ValueError(f"OBJ appears empty: {args.obj}")

    tris = vertices[faces]
    face_colors = build_face_colors(tris)
    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = max(float(np.max(maxs - mins)) / 2.0, 1e-6)

    images: list[np.ndarray] = []
    fig = plt.figure(figsize=(args.size, args.size), dpi=args.dpi)
    ax = fig.add_subplot(111, projection="3d")
    fig.patch.set_facecolor("black")

    for azim in np.linspace(0.0, 360.0, args.frames, endpoint=False):
        ax.cla()
        ax.set_facecolor("black")
        ax.grid(False)
        ax.set_axis_off()
        ax.view_init(elev=args.elev, azim=float(azim))

        poly = Poly3DCollection(
            tris,
            facecolors=face_colors,
            linewidths=0.04,
            edgecolors=(0.02, 0.02, 0.02, 0.18),
            alpha=1.0,
        )
        ax.add_collection3d(poly)

        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)
        ax.set_box_aspect((1, 1, 1))

        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)
        img = rgba[..., :3]
        images.append(img.copy())

    plt.close(fig)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    duration = 1.0 / max(args.fps, 1)
    iio.imwrite(args.out, images, format_hint=".gif", duration=duration, loop=0)
    print(f"Turntable GIF saved: {args.out} ({len(images)} frames @ {args.fps} fps)")


def load_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[list[float]] = []
    faces: list[list[int]] = []

    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("v "):
            _, x, y, z = line.split(maxsplit=3)
            vertices.append([float(x), float(y), float(z)])
        elif line.startswith("f "):
            parts = line.split()[1:]
            indices = []
            for part in parts:
                token = part.split("/")[0]
                indices.append(int(token) - 1)
            if len(indices) == 3:
                faces.append(indices)
            elif len(indices) > 3:
                for i in range(1, len(indices) - 1):
                    faces.append([indices[0], indices[i], indices[i + 1]])

    return np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.int32)


def build_face_colors(tris: np.ndarray) -> np.ndarray:
    centroids = tris.mean(axis=1)
    z = centroids[:, 2]
    z_min = float(np.min(z))
    z_max = float(np.max(z))
    if z_max <= z_min:
        z_norm = np.zeros_like(z)
    else:
        z_norm = (z - z_min) / (z_max - z_min)

    colors = plt.get_cmap("magma")(z_norm)
    colors[:, 3] = 1.0
    return colors


if __name__ == "__main__":
    main()
