# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A neural Eikonal-equation path planner. A network learns a travel-time field `tau(x0, x1)`
between pairs of configurations; its spatial gradient gives the local speed (clearance), and
following `-grad(tau)` yields collision-free paths. The same architecture is reused across
several configuration spaces, distinguished only by `dim`:

- **2-D shape** (`dim=3`): rigid body in SE(2) — `(x, y, theta)`.
- **3-D shape** (`dim=6`): rigid body in SE(3) — `(x, y, z, rx, ry, rz)`, where `(rx,ry,rz)` is a rotation vector (axis-angle).
- **arm** (`dim=6`): UR5 joint space, handled by the parallel `models/metric_arm/` package.
- **maze / gibson**: 2-D point-robot variants.

All rotation coordinates are stored **normalized by `2*pi`** so every coordinate lives on a
comparable scale (the network applies identical Fourier features to all of them).

## Pipeline

Every task follows **preprocess → train → evaluate**. The full set of concrete commands per
environment lives in `README.md`; the shapes of those commands are:

```
# 2-D (DXF inputs)
python dataprocessing/preprocess_dxf.py --env <env.dxf> --shape <shape.dxf> --out <dir> --num_samples 400000 [--visualize]
python train/train_2dshape.py --dataPath datasets/2dshape/<name>
python evaluate_training.py --dataPath testing_data/2dshape/<name> --out ./results/output_2d/<name>

# 3-D (OBJ inputs)
python dataprocessing/preprocess_obj.py --env <env.obj> --shape <shape.obj> --out <dir> --num_samples 400000 [--visualize] [--yrot] [--device cuda:2]
python train/train_3dshape.py --dataPath datasets/3dshape/<name>
python evaluate_training_3d.py --dataPath testing_data/3dshape/<name> --out ./results/output_3d/<name>
```

- Add `--testing_data --num_samples 1000` to a preprocess call to generate the held-out start/goal pairs used for evaluation.
- There is **no test suite, linter, or build step.** Validation is done by training and inspecting the rendered travel-time / trajectory plots.
- `evaluate_training_3d.py` launches a `viser` web viewer on port **8080** (browse episodes via a GUI dropdown) and also writes plotly HTML summaries to `--out`.

## Environment

Runs inside Docker (the host has a Conda env too, but Docker is the supported path). Build once,
then run with X11 + GPU forwarding — see `README.md` for the exact `docker run` invocation
(it bind-mounts the repo to `/workspace` and forwards port 8080).

```
docker build -f Dockerfile.server -t pytorchserver .
```

Key deps: PyTorch 2.2 / CUDA 12.1, `libigl` (mesh SDF / tetrahedralization), `ezdxf`/`shapely`
(2-D geometry), `viser` + `plotly` (3-D eval rendering), `wandb`.

## Architecture

### The `Model` / `Function` / `NN` triad

Each task package is a near-copy of the same three files:

- `models/metric_2dshape/` — the 2-D shape pipeline (`train/train_2dshape.py`, `evaluate_training.py`,
  `evaluate_training_batched.py`).
- `models/shape_metric/` — 3-D shape (default for every trainer/eval script), maze, gibson.
  Blocked Eikonal loss: translation and rotation blocks each satisfy `S_block |grad_block tau| = 1`.
- `models/lie_metric/` — same network/trainer as `shape_metric`; loss is the block-anisotropic
  Eikonal `sqrt(S_dist^2 |grad_v tau|^2 + S_angle^2 |grad_w tau|^2) = 1`, with the rotation block
  taken as the body-frame Lie gradient (`lie_rot_grad`, see `anisotropic_speed_field_implementation.pdf`).
- `models/metric_baseline/` — the original metric NTFields (single-route network, isotropic Eikonal
  on the combined `speed.npy`), copied from `ntrlshape/.../baselines/ntrl-demo/models/metric` with the
  TD and normal losses switched off. Select a package with `--models` on `train/train_3dshape.py`
  and `evaluate_training_3d_batched.py`.
- `models/lie_metric_6s/` — `metric_baseline`'s network and trainer with six independent speeds:
  `sqrt(g^T diag(S_x^2, S_y^2, S_z^2, S_r1^2, S_r2^2, S_r3^2) g) = 1` per endpoint, where
  `g = (dtau/dp [world], lie_rot_grad [body])`, mapped as `1e-2*(sqrt(N)-1)^2`. Needs `speeds_6.npy`
  from `dataprocessing/preprocess_obj_6s.py`: exact free travel along each world axis and free rotation
  about each body axis against the env point cloud (min over +/-), normalized like
  `speed_dists` / `speed_angles`. That script takes every `preprocess_obj.py` flag, or
  `--from_dataset <dir>` to reuse an existing dataset's pairs.
- `models/metric_lie_rhombus/` — `lie_metric` with an additive block time cost
  `dt = da/S_dist + db/S_angle` (da = translation step length, db = rotation step angle; a rhombus
  instead of the ellipse `sqrt(da^2/S_dist^2 + db^2/S_angle^2)`). Its Eikonal equation is the dual
  norm `max(S_dist |grad_v tau|, S_angle |grad_w tau|) = 1`, mapped `1e-2*(sqrt(N)-1)^2`.
  `--dual-p` replaces the max with a p-norm (`p = 2` is `lie_metric`).
- `models/metric_arm/` — the UR5.

**Changes to the core algorithm usually need to be mirrored across all three packages.**

- **`model_network_metric.py` — `NN`**: the field network. Inputs are Fourier-mapped
  (`input_mapping`), passed through Lipschitz-normalized layers (`lip_norm`), and the travel time
  between the two endpoints is a learned **metric**: each endpoint is embedded, and `tau` is a
  smooth (`logsumexp`) aggregation of per-group distances between the two embeddings. `out(coords)`
  returns `(tau, w, coords)` and is the single entry point used everywhere. Both endpoints are
  stacked and run through the net in one batch.
- **`model_function_metric.py` — `Function`**: the loss and all field queries. `Loss` is the
  Eikonal residual: `|grad tau| * speed == 1`, with separate distance/angle speed terms, plus a
  surface-normal alignment term and an optional time-difference (`tau_loss`) term. `TravelTimes`,
  `Speed`, and `Gradient` are the inference-time queries used by evaluation. `plot` renders the
  travel-time field.
- **`model_train_metric.py` — `Model`**: the training driver. Owns hyperparameters in the
  `self.Params` dict (epochs, batch size, LR, etc.), the training loop, and checkpointing.

### Training loop specifics (non-obvious)

- The loop in `Model.train()` does **only 5 batches per epoch** (`if ii>4: break`) and divides
  losses by 4.
- It uses an **accept/reject scheme**: each epoch is retried (network + optimizer rolled back to a
  random one of the last 5 saved states) until the loss ratio `current_diff/prev_diff < 1.2`. Watch
  for this when reasoning about "why didn't my change take effect" — bad epochs are silently reverted.
- `beta` (loss scale) is reset to `1/total_diff` each epoch; `speed`/`speed_dist`/`speed_angle` are
  passed through fixed polynomial warps before entering the loss (see lines around the dataloader
  unpacking) — these reshape the clearance signal, not raw clearance.
- LR is **hard-overwritten to `5e-4` inside the loop**, so `--lr` / `self.Params` LR is largely
  cosmetic; change the in-loop assignment to actually alter LR.
- On the first `train()` call the contents of the task's model package (`source_folder` in
  `model_train_metric.py`) are **copied into the run folder**
  (`Experiments/<task>/<dataset>_<timestamp>/models/`) to snapshot the exact code used.

### Checkpoints

- Per-cadence checkpoints `Model_Epoch_XXXXX_ValLoss_*.pt` plus a rolling `latest.pt`, written to
  both the run folder and the top-level `ModelPath` (`./Experiments/<task>/`). Eval scripts load
  `latest.pt` by default.
- A checkpoint stores `model_state_dict`, `optimizer_state_dict`, and crucially `B_state_dict` (the
  random Fourier frequency matrix). The network **cannot** be reconstructed without `B`, so always
  load it via `Model.load` / `load_pretrained_state_dict`.

### Data format

`models/data_mlp.py::Database` loads five `.npy` files from a dataset dir and concatenates them into
one tensor per row, unpacked by offset in the training loop:

```
sampled_points.npy (N, 2*dim)  speed.npy (N, 2)  normal.npy (N, 2*dim)
speed_dists.npy (N, 2)  speed_angles.npy (N, 2)
```

The column layout in the concatenated tensor (see `model_train_metric.py`) is:
`points[:2*dim] | speed[2] | normal[2*dim] | speed_dist[2] | speed_angle[2]`. Each row is a
**correlated pair** `(x0, x1)`: `x0` sampled in the narrow clearance band, `x1` a random
collision-free SE(k) displacement.

### Experiment tracking

W&B is **on by default**. Shared helpers in `train/wandb_utils.py` (`add_wandb_args`,
`apply_overrides`, `start_run`, `finish_run`) are wired into the trainers; per-epoch `wandb.log`
calls in `model_train_metric.py` are guarded so they no-op without a run. Disable with `--no-wandb`.
Common overrides: `--epochs`, `--batch-size`, `--lr`, `--data`, `--set KEY=VALUE`.

## Experiment tree (`research_tree/`): keep it current

`research_tree/tree.yaml` records every model variant tried, including failures. It is a flat
list of nodes linked by `parent`, with one root per package in `models/` (`metric_baseline`,
`shape_metric`, `lie_metric`, `lie_metric_6s`, `metric_lie_rhombus`). `python research_tree/render.py` validates it
and rebuilds `research_tree/tree.html`, a self-contained page with a diagram, an outline, and a
details panel for each node. The field list and status meanings are in the header comment of
`tree.yaml`.

- **Session start:** before proposing or running a model change, read the tree with
  `python research_tree/render.py --text` (add `--root <id>` for one subtree). This avoids
  repeating dead ends; RNL on `lie_metric`, for example, failed at every setting.
- **Update it in the same session, without being asked,** whenever:
  - you create or propose a variant (loss, residual, weighting, architecture, speeds,
    preprocessing that feeds a model, or a hyperparameter sweep). Add a node with status
    `running` if training started, or `idea` if it is only proposed.
  - a training or evaluation run finishes. Fill in `fwd` / `coll` / `result` and set the status
    to `win`, `mixed` or `fail`.
  - an analysis changes how an existing result should be read. Edit that node's `notes`.
  - a run is abandoned. Set it to `dropped` and say why in `result`.

  Then run `python research_tree/render.py`, fix any validation errors, and mention the tree
  change in your reply.
- **Rules:**
  - Never delete a node; mark it `fail` or `dropped` instead.
  - `parent` is the variant this run actually changed, ideally one change per node. A new
    package under `models/` gets a new root.
  - `fwd` is only the headline metric defined in `meta.metric`: Tshape3d_env1, standard 1k set,
    forward-MPPI success %. Read it from the `regular/forward` line of
    `results/output_3d/<run>/success_rate.txt`. Put other seeds, sets, envs and eval scripts in
    `result`, and say there when a number comes from a different evaluation.
  - Fill `runs` (the `Experiments/` run folder and the `results/` output) and `code` (the files
    that implement the change), so every node points back to its artifacts.

## Conventions

- Trainers and eval scripts do `sys.path.append('.')` and must be **run from this nested `ntrl-demo`
  root** (the one containing `models/`, `train/`, `datasets/`).
- Source/goal configurations are hardcoded as the `source` argument to `md.Model(...)` in each
  trainer (e.g. SE(3) goal in `train_3dshape.py`); change them there.
- `dim` is passed explicitly to `Model` and threaded everywhere; it is the single switch between the
  2-D / 3-D / arm variants.
