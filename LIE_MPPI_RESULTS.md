# Lie MPPI implementation and evaluation

## Controller and step calibration

[`controllers/lie_mppi.py`](controllers/lie_mppi.py) samples unit body tangents
`xi = (v, w)`, with angular components measured in turns. The selected metric is

```text
d(g,h)^2 = ||p_g-p_h||^2 + (angle(R_g^T R_h)/(2*pi))^2
||xi||^2 = ||v||^2 + ||w||^2 = 1
Exp_g(t*xi) = (p + t*R*v, R*exp(2*pi*t*[w]x))
```

For this Riemannian exponential, `d(g, Exp_g(t*xi)) = t` below the rotational
injectivity radius. Thus the common parameter is **t = step = 0.015**. This
is a closed-form calibration; no direction-dependent root search is required.
See [equation (4) of the geodesic derivation](https://doi.org/10.1007/s00022-025-00754-9).
Each horizon step composes at its own current pose. The weighted control mean
is formed in the current tangent space and executed through the same map.

## Full comparison

Dataset: `testing_data/3dshape/Tshape3d_env1`, all 1,000 cases, all valid endpoints.
Checkpoint:
`Experiments/3dshape_lie_metric/Tshape3d_env1_lie_metric_rsqrtN_s1/Model_Epoch_04000_ValLoss_1.616883e-03.pt`.

Both fresh runs used CPU, planning/cloud seed 0, batch 125, 50 candidates,
horizon 5, 200 iterations, temperature 50, zero momentum and the same
product-geodesic convergence threshold of 0.01. Both used the existing learned
cost and waypoint collision protocol.

| Run | Success | Collision | No convergence |
| --- | ---: | ---: | ---: |
| Historical original | 670/1000 (67.0%) | 276/1000 (27.6%) | 69/1000 (6.9%) |
| Fresh original | 680/1000 (68.0%) | 269/1000 (26.9%) | 76/1000 (7.6%) |
| Lie MPPI | 676/1000 (67.6%) | 272/1000 (27.2%) | 71/1000 (7.1%) |

Lie MPPI rescued 54 cases and lost 58: overall success decreased by
**0.4 percentage points** against the fresh baseline. It reduced nonconvergence
by 0.5 points and increased collisions by 0.3 points. Collision and
nonconvergence can overlap. This run establishes the requested sampling
geometry, but does not demonstrate an overall success improvement.

CPU planning took 786.79 seconds for the original and 780.04 seconds for Lie
MPPI; this single timing pair does not establish a speed advantage. Historical
GPU outcomes provide context rather than an identical random stream:
[PyTorch does not guarantee matching CPU/GPU seeded executions](https://docs.pytorch.org/docs/2.14/notes/randomness.html).
This is one checkpoint and one planning seed. Collision checks cover recorded
waypoints, following the original protocol.

## Validation and reproduction

All 12 geometry/controller unit tests passed. An independent SciPy rotation
calculation checked 16,000 production float32 sample steps from actual test
poses: distances ranged from 0.01499994235 to 0.01500007305, with maximum error
**7.31e-8**. The dedicated evaluator also completed a two-case smoke run across
all five controller variants.

```bash
python -m unittest discover -s tests -v
python analysis/compare_lie_mppi.py \
  --historical results/compare/plan/lie_rsqrtN_s1_g/Tshape3d_env1.json \
  --out results/lie_mppi_comparison/<new-run> --device cuda:0
```

Use `--device cpu` for CPU execution. The dedicated evaluator is
`evaluate_training_3d_lie_mppi.py`; the original evaluator exposes `--sampler lie`.
Complete per-case results, commands, hashes, versions, angle breakdown and
sampling validation are in
[`results/lie_mppi_comparison/Tshape3d_env1_rsqrtN_s1_seed0/`](results/lie_mppi_comparison/Tshape3d_env1_rsqrtN_s1_seed0/).
