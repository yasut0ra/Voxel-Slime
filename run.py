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
    parser.add_argument("--species", dest="species_count", type=int)
    parser.add_argument("--interaction-mode", choices=["symmetric", "cyclic"])
    parser.add_argument("--self-attract", type=float)
    parser.add_argument("--cross-attract", type=float)
    parser.add_argument("--no-predator", action="store_true")
    parser.add_argument("--predator-ratio", type=float)
    parser.add_argument("--predator-attract", type=float)
    parser.add_argument("--predator-repel", type=float)
    parser.add_argument("--predator-self-attract", type=float)
    parser.add_argument("--predator-consume-amount", type=float)
    parser.add_argument("--predator-capture-radius", type=int)
    parser.add_argument("--predator-food-weight", type=float)
    parser.add_argument("--predator-toxin-weight", type=float)

    parser.add_argument("--no-food-field", action="store_true")
    parser.add_argument("--no-toxin-field", action="store_true")
    parser.add_argument("--food-weight", type=float)
    parser.add_argument("--toxin-weight", type=float)
    parser.add_argument("--food-sources", type=int)
    parser.add_argument("--toxin-sources", type=int)
    parser.add_argument("--field-source-speed", type=float)
    parser.add_argument("--field-source-jitter", type=float)
    parser.add_argument("--field-injection-radius", type=int)
    parser.add_argument("--food-injection-amount", type=float)
    parser.add_argument("--toxin-injection-amount", type=float)
    parser.add_argument("--field-diffuse-rate", type=float)
    parser.add_argument("--field-evap-rate", type=float)

    parser.add_argument("--out", dest="out_dir", type=str)
    parser.add_argument("--save-every", type=int)
    parser.add_argument("--boundary", choices=["wrap", "reflect"])

    parser.add_argument("--slice-axis", choices=["x", "y", "z"])
    parser.add_argument("--slice-index", type=int)
    parser.add_argument("--render-colormap", type=str)
    parser.add_argument("--render-gamma", type=float)
    parser.add_argument("--env-overlay-strength", type=float)

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
        "species_count": args.species_count,
        "interaction_mode": args.interaction_mode,
        "self_attract": args.self_attract,
        "cross_attract": args.cross_attract,
        "predator_enabled": False if args.no_predator else None,
        "predator_ratio": args.predator_ratio,
        "predator_attract": args.predator_attract,
        "predator_repel": args.predator_repel,
        "predator_self_attract": args.predator_self_attract,
        "predator_consume_amount": args.predator_consume_amount,
        "predator_capture_radius": args.predator_capture_radius,
        "predator_food_weight": args.predator_food_weight,
        "predator_toxin_weight": args.predator_toxin_weight,
        "food_field_enabled": False if args.no_food_field else None,
        "toxin_field_enabled": False if args.no_toxin_field else None,
        "food_weight": args.food_weight,
        "toxin_weight": args.toxin_weight,
        "food_sources": args.food_sources,
        "toxin_sources": args.toxin_sources,
        "field_source_speed": args.field_source_speed,
        "field_source_jitter": args.field_source_jitter,
        "field_injection_radius": args.field_injection_radius,
        "food_injection_amount": args.food_injection_amount,
        "toxin_injection_amount": args.toxin_injection_amount,
        "field_diffuse_rate": args.field_diffuse_rate,
        "field_evap_rate": args.field_evap_rate,
        "out_dir": args.out_dir,
        "save_every": args.save_every,
        "boundary": args.boundary,
        "slice_axis": args.slice_axis,
        "slice_index": args.slice_index,
        "render_colormap": args.render_colormap,
        "render_gamma": args.render_gamma,
        "env_overlay_strength": args.env_overlay_strength,
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
