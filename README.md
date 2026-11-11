# MNDF

Minimal simulation and CMA-ES optimization code for heterogeneous multilayer swarm dynamics.
Crossing and Search-and-Rescue share one implementation in `mndf/dynamics.py`.

## Model and supplied parameters

- Crossing (`scene_type_id=1`): relay TopK30, ordinary-source TopK10,
  fixed self-drive direction `(1, 0)`, `dt=0.125`, duration 100.
- Search-and-Rescue (`scene_type_id=2`): relay TopK30, ordinary-source TopK10,
  self-drive along current velocity, collision radius 3, collision shaping
  coefficient `0.001`, `dt=0.125`, duration 400.

Each task includes 27 optimized parameter files with 57 variables per file.
The configuration grid is `rc={30,60,90}`, `ra={5,10,15}`, `rs={15,30,45}`.
Search-and-Rescue parameters are from the **OriginalCollision / FixInitial** run.
They are not the parameters of the earlier `0.00003` collision-coefficient variant.
The supplied parameter files are unchanged; `PROVENANCE.json` records their
source paths and SHA256 values, along with constraint and search-bound files.
Those paths document provenance and are not runtime dependencies.

## Install

Use Python 3.10 or newer; the verified environment uses Python 3.10,
PyTorch 2.4.0, NumPy 1.26.4, cma 4.0.0, and SciPy 1.15.2.
Install from the release directory:

```bash
python -m pip install -r requirements.txt
```

CUDA is recommended for simulation and optimization. `--device cpu` is supported
for small runs. No plotting libraries or files from the parent project are needed.
Run commands below from this directory; this is a source release, not an installed wheel.

## Simulate

Two representative input scenes are included, one per task:

```bash
python -m mndf simulate --config crossing_heter_1_rc60_ra10_rs30 --device cuda:0
python -m mndf simulate --config searchandrescue_heter_1_rc60_ra10_rs30 --device cuda:0
```

For a short run, add `--t-max 2`. Results are printed as JSON. Use
`--output /path/outside/this/release/fitness.json` to save them.

To use another configuration or initial scene:

```bash
python -m mndf simulate --config crossing_heter_1_rc30_ra5_rs15 --data-dir /path/to/crossing-data --scene 1 --device cuda:0
```

The data directory must contain:

```text
NodeInitialLocationData/<config_name>_<scene_index>.json
TaskInfo/<config_name>_<scene_index>.json
```

Node JSON is a list of records containing `id`, `type`, `pos`, `velocity`,
`velocity_max`, `acceleration_max`, `r_s`, `r_c`, `r_a`, and `task`;
`group_index` is optional. Task JSON depends on the task; the supplied examples
show the original formats. A sibling `Constraint/` directory is used when
present; otherwise the bundled constraints are loaded.

## Generate scenes

Generate the two input configurations supplied in `examples/`, with the original
default base seed 42, into a separate directory:

```bash
python -m mndf generate --examples --samples 1 --seed 42 --output ../generated-examples
python -m mndf simulate --config crossing_heter_1_rc60_ra10_rs30 --data-dir ../generated-examples --device cuda:0
python -m mndf simulate --config searchandrescue_heter_1_rc60_ra10_rs30 --data-dir ../generated-examples --device cuda:0
```

Generate all ten training scenes for all 54 configurations, covering both tasks:

```bash
python -m mndf generate --samples 10 --seed 42 --output ../generated-training
```

Select a single task with `--task crossing` or `--task searchandrescue`, or a single
configuration with `--config crossing_heter_1_rc60_ra10_rs30`. The output contains
`NodeInitialLocationData/`, `TaskInfo/`, and a `generation.json` manifest. Existing
matching output files are protected; explicit `--force` permits replacement.
To regenerate the bundled example inputs in place:

```bash
python -m mndf generate --examples --samples 1 --seed 42 --output examples --force
```

The generator retains the original numerical scene algorithms and removes their
plotting and experiment orchestration dependencies:

- **Crossing:** randomized channel control-point heights, cubic splines,
  circular entrance and quintic Hermite connection; channel and circular boundary
  nodes; 40 random obstacles; 63 robots placed with original spacing rejection
  rules and initial velocity `(5, 0)`.
- **Search-and-Rescue:** circular boundary of radius 300 with 360 boundary nodes,
  60 tasks with initial workload 10 in the original radial interval `[150, 250]`,
  and 63 robots in three structured swarms centered collectively at `(0.1, 0.1)`.
  Robot velocity points away from that center with magnitude 5. The original
  random robot placement is evaluated before replacement to preserve the random
  sequence used for task generation, matching the original FixInitial pipeline.
- Each scene uses Python and NumPy seed `base_seed + scene_index`, where scene
  indices start at 1. All configurations share a scene's geometry; only the
  selected communication, action, and sensing ranges change. Ordinary nodes keep
  communication radius 30; `rc` changes relay nodes only. Search-and-Rescue robot
  positions and velocities remain fixed across scenes, while tasks change.

Changing the seed produces new scenes; the supplied optimized parameters were
obtained with the original training inputs, not with an arbitrary new seed.
The generator uses the three-swarm scale associated with those parameters.
With the default seed, bundled node JSON is reproduced exactly as parsed JSON.
Raw spline task coordinates can differ at approximately `3e-14` across numerical
builds; regression checks use absolute tolerance `1e-10` for task coordinates.

Run the scene generation regression tests:

```bash
python -m unittest discover -s tests -v
```

## Optimize

```bash
python -m mndf optimize --config crossing_heter_1_rc60_ra10_rs30 --data-dir /path/to/crossing-data --device cuda:0 --output /path/to/new-run
python -m mndf optimize --config searchandrescue_heter_1_rc60_ra10_rs30 --data-dir /path/to/searchandrescue-fixinitial-data --device cuda:0 --output /path/to/new-sr-run
```

Full optimization requires scenes 1 through 10 for the chosen configuration.
Only one example scene per task is shipped; use the generator above to produce
the complete training inputs without the parent project. Bulk training data,
historical results, figures, ablation algorithms, and experiment schedulers are
excluded. For example:

```bash
python -m mndf optimize --config searchandrescue_heter_1_rc60_ra10_rs30 --data-dir ../generated-training --device cuda:0 --output ../new-sr-run
```

The optimizer starts from seeded random parameters with seed 42, sigma 0.5,
128 candidates per generation and a fixed 100-generation budget. It searches
on scene 1, ranks all sampled candidates, selects up to 256 distinct candidates,
and chooses the best mean objective over all ten scenes. The final JSON uses
the same parameter structure as the supplied optimized parameters.
Search bounds are in `Parameter/`; network relations are in `Constraint/`.

The compact optimizer uses one chosen device and evaluates the complete
population in batches of 32. Population size remains 128 regardless of device
count. It does not include the original multi-GPU experiment queue.
Use `--generations 1 --t-max 2` for a short pipeline check with ten input scenes;
this does not reproduce the full optimized result.

Outputs go to a new or empty directory outside this release. Supplied optimized
parameters are never used as an optimization output directory. On this server,
start long runs in a detached tmux session and use the `wxd` environment:

```bash
tmux new-session -d -s mndf-opt 'cd /absolute/path/to/MNDF && conda run --no-capture-output -n wxd python -m mndf optimize --config crossing_heter_1_rc60_ra10_rs30 --data-dir /absolute/path/to/crossing-data --device cuda:0 --output /absolute/path/to/new-run > /absolute/path/to/optimization.log 2>&1'
```

## Files

- `mndf/dynamics.py`: shared topology, control forces, state integration,
  Search-and-Rescue task updates, and fitness evaluation.
- `mndf/cli.py`: simulation and CMA-ES entry point.
- `mndf/scenes.py` and `mndf/_scene_*.py`: scene generation and original numerical
  geometry, initialization, and fixed-formation helpers.
- `optimized_params/`: 54 original optimized parameter files.
- `Constraint/`: 54 network relation configurations.
- `Parameter/`: 54 parameter search-bound configurations.
- `configs/`: task-specific numerical and optimization settings.
- `examples/`: one node/task input pair per task.
- `requirements.txt`: verified numerical dependencies.

The core preserves task-specific fitness definitions, including Crossing's
collision coefficient `0.00003` and Search-and-Rescue's coefficient `0.001`.
Python API `evaluation_main_batched` optionally returns metric time series
(`collect_series=True`) or saves raw state tensors (`state_save_count`,
`state_save_dir`); it does not generate plots.
