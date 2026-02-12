"""Simulation orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np

from .agents import initialize_agents, step_agents
from .config import SimulationConfig
from .export import export_obj
from .render import save_frame_pair
from .trail import diffuse_and_evaporate
from .utils import ensure_dir


@dataclass(slots=True)
class SimulationResult:
    """Summary of simulation artifacts."""

    output_dir: Path
    frames_saved: int
    elapsed_seconds: float
    obj_path: Path | None


def run_simulation(cfg: SimulationConfig) -> SimulationResult:
    """Run full particle + trail simulation and write frames/artifacts to disk."""
    cfg.validate()

    out_dir = ensure_dir(cfg.out_dir)
    rng = np.random.default_rng(cfg.seed)

    trail = np.zeros((cfg.size, cfg.size, cfg.size), dtype=np.float32)
    agents = initialize_agents(count=cfg.agents, size=cfg.size, rng=rng)

    frames_saved = 0
    obj_path: Path | None = None

    start = perf_counter()
    print(
        f"Starting simulation | size={cfg.size}^3 agents={cfg.agents} steps={cfg.steps} "
        f"boundary={cfg.boundary} seed={cfg.seed}"
    )

    progress_stride = max(1, cfg.steps // 20)

    for step in range(1, cfg.steps + 1):
        collisions = step_agents(agents, trail, cfg, rng)
        diffuse_and_evaporate(trail, cfg)

        should_save = step == 1 or step == cfg.steps or step % cfg.save_every == 0
        should_report = should_save or (step % progress_stride == 0)

        if should_save:
            save_frame_pair(
                trail,
                step=step,
                out_dir=out_dir,
                slice_axis=cfg.slice_axis,
                slice_index=cfg.slice_index,
            )
            frames_saved += 1

        if should_report:
            elapsed = perf_counter() - start
            sps = step / max(elapsed, 1e-9)
            eta = (cfg.steps - step) / max(sps, 1e-9)
            save_mark = "save" if should_save else "...."
            print(
                f"[{step:5d}/{cfg.steps}] {save_mark} collisions={collisions:5d} "
                f"speed={sps:7.1f} step/s eta={eta:7.1f}s"
            )

    if cfg.export_obj:
        target_path = Path(out_dir) / "mesh.obj"
        try:
            obj_path = export_obj(trail, threshold=cfg.obj_threshold, out_path=target_path)
            print(f"OBJ exported: {obj_path}")
        except ValueError as err:
            print(f"OBJ export skipped: {err}")

    elapsed_total = perf_counter() - start
    print(f"Finished in {elapsed_total:.2f}s | saved frames={frames_saved}")

    return SimulationResult(
        output_dir=Path(out_dir),
        frames_saved=frames_saved,
        elapsed_seconds=elapsed_total,
        obj_path=obj_path,
    )
