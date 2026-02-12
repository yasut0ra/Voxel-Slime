# Voxel Slime

Physarum-inspired 3D generative art: thousands of agents deposit and follow trails inside a voxel grid to grow organic network structures.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run a striking default simulation (fungi preset, 2,000 steps):

```bash
python run.py --preset fungi --seed 7 --out outputs/fungi
```

Build a GIF from max-intensity frames:

```bash
python scripts/make_gif.py \
  --frames-dir outputs/fungi/frames/mip \
  --pattern "mip_*.png" \
  --out outputs/fungi/fungi_mip.gif \
  --fps 24
```

Export OBJ mesh from trail isosurface:

```bash
python run.py --preset coral --steps 1200 --seed 42 --export-obj --obj-threshold 12 --out outputs/coral_obj
```

## CLI

```bash
python run.py \
  --preset {fungi,nebula,coral} \
  --size 96 \
  --agents 50000 \
  --steps 2000 \
  --seed 42 \
  --out outputs/run_name \
  --save-every 10 \
  --export-obj \
  --obj-threshold 9.5 \
  --boundary {wrap,reflect} \
  --slice-axis {x,y,z} \
  --slice-index 48
```

## Algorithm (5 steps)

1. Initialize many agents with random positions and headings in a 3D voxel grid.
2. Sense trail concentration in multiple directions around each heading (forward, left/right, up/down, diagonals).
3. Turn toward stronger trail, move one voxel-neighborhood step, and deposit new trail.
4. Diffuse trail (neighbor averaging blur) and evaporate trail each iteration.
5. Render grayscale frames (mid-slice + max-intensity projection) and optionally extract a 3D mesh via marching cubes.

## Outputs

- `outputs/<run>/frames/slice/slice_XXXXXX.png`: axis-aligned slice frames with auto-contrast.
- `outputs/<run>/frames/mip/mip_XXXXXX.png`: max-intensity projection frames with auto-contrast.
- `outputs/<run>/mesh.obj` (optional): marching cubes mesh with vertex normals.

## Presets

- `fungi`: moderate diffusion + low evaporation + strong sensing -> filament networks.
- `nebula`: high diffusion + higher evaporation + soft sensing -> cloudy nebula-like forms.
- `coral`: lower diffusion + low evaporation + higher deposit -> chunky branching coral forms.

## Notes

- Deterministic runs are supported with `--seed`.
- Inspired by Physarum particle/trail models and chemotaxis-like behavior.
- For faster exploratory tests, reduce `--steps`, `--agents`, or `--size`.

## Repository Layout

```text
voxel-slime/
  run.py
  src/
    __init__.py
    config.py
    agents.py
    trail.py
    simulate.py
    render.py
    export.py
    utils.py
  configs/
    fungi.yaml
    nebula.yaml
    coral.yaml
  scripts/
    make_gif.py
  requirements.txt
  README.md
  LICENSE
```
