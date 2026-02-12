#!/usr/bin/env python3
"""Create a GIF from a directory of PNG frames."""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v3 as iio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build GIF from PNG frames")
    parser.add_argument("--frames-dir", required=True, type=Path)
    parser.add_argument("--pattern", default="*.png", type=str)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--fps", default=24, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    frames = sorted(args.frames_dir.glob(args.pattern))
    if not frames:
        raise FileNotFoundError(
            f"No frames matching '{args.pattern}' in {args.frames_dir}"
        )

    images = [iio.imread(frame) for frame in frames]
    args.out.parent.mkdir(parents=True, exist_ok=True)

    duration = 1.0 / max(args.fps, 1)
    iio.imwrite(args.out, images, format_hint=".gif", duration=duration, loop=0)
    print(f"GIF saved: {args.out} ({len(images)} frames @ {args.fps} fps)")


if __name__ == "__main__":
    main()
