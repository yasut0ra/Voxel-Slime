#!/usr/bin/env python3
"""CLI entrypoint for Voxel Slime."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.config import SimulationConfig
from src.simulate import run_simulation
from src.utils import load_simple_yaml


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="3D Physarum-inspired voxel slime simulation")
    parser.add_argument("--preset", default="fungi", choices=["fungi", "nebula", "coral"])

    parser.add_argument("--size", type=int)
    parser.add_argument("--agents", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--seed", type=int)

    parser.add_argument("--out", dest="out_dir", type=str)
    parser.add_argument("--save-every", type=int)
    parser.add_argument("--boundary", choices=["wrap", "reflect"])

    parser.add_argument("--slice-axis", choices=["x", "y", "z"])
    parser.add_argument("--slice-index", type=int)

    parser.add_argument("--export-obj", action="store_true")
    parser.add_argument("--obj-threshold", type=float)

    return parser.parse_args()


def load_preset(preset_name: str) -> SimulationConfig:
    """Load base configuration from configs/<preset>.yaml."""
    preset_path = Path("configs") / f"{preset_name}.yaml"
    if not preset_path.exists():
        raise FileNotFoundError(f"Preset file not found: {preset_path}")

    preset_data = load_simple_yaml(preset_path)
    return SimulationConfig.from_dict(preset_data)


def main() -> None:
    """Program entrypoint."""
    args = parse_args()
    base_cfg = load_preset(args.preset)

    overrides = {
        "size": args.size,
        "agents": args.agents,
        "steps": args.steps,
        "seed": args.seed,
        "out_dir": args.out_dir,
        "save_every": args.save_every,
        "boundary": args.boundary,
        "slice_axis": args.slice_axis,
        "slice_index": args.slice_index,
        "obj_threshold": args.obj_threshold,
        "export_obj": args.export_obj,
    }

    cfg = base_cfg.merged(overrides)
    cfg.validate()

    print(f"Preset: {args.preset}")
    print(f"Output directory: {cfg.out_dir}")
    run_simulation(cfg)


if __name__ == "__main__":
    main()
