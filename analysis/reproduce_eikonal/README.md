# Five-seed Lie Eikonal reproduction

This reproduces every configuration described in `Isotropic vs Anisotropic Lie Eikonal.html`: the metric baseline, anisotropic Lie model, isotropic Lie model, original-network control, chart-gradient control, and the L1 block sum mentioned in the notes.

The full sweep uses **physical GPU 3 only**, fresh training seeds 1–5, 4,000 epochs, all 19 existing test sets, and evaluation seed 0. It generates 90 final checkpoints and 570 evaluations. Existing checkpoints and scores are preserved. The full report is marked complete only after every evaluation passes consistency checks.

Run from the repository root using the installed `pytorchserver:latest` image:

```bash
docker run --rm --gpus device=3 --network none --user 1006:1007 \
  --name lie-eikonal-five-seed-device3 --detach \
  -e CUDA_VISIBLE_DEVICES=0 -e MPLCONFIGDIR=/tmp/eikonal-matplotlib \
  -e OMP_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -v /media/jeffrey/Lie-Shape-NTFields:/workspace -w /workspace \
  pytorchserver:latest python -u analysis/reproduce_eikonal/run.py --parallel 4
```

The runner checks the visible GPU UUID before initializing training. Docker maps physical GPU 3 to the container's `cuda:0`. Four concurrent jobs share that one GPU. No network access or W&B upload is used.

Outputs are in `results/eikonal_five_seed_device3/`: `REPORT.html`, `REPORT.md`, raw per-case `plan/` JSON, final `checkpoints/`, `logs/`, `per_seed.csv`, `suite_per_seed.csv`, `summary.json`, frozen `source/`, input/source hashes in `manifest.json`, package versions in `environment.txt`, and live `status.json`. Reports refresh during execution and are finalized automatically. Failed jobs are shown explicitly.

Run the same command again after an interruption to resume validated checkpoints and completed evaluations. The runner refuses source/input changes against the frozen manifest and refuses concurrent sweeps in the same output directory. Incomplete training restarts from its explicit seed; training state is not silently resumed at an arbitrary epoch.

For integration validation, replace `--parallel 4` with `--smoke --parallel 2 --out results/eikonal_five_seed_device3_smoke`. This exercises all six models with two epochs and ten planning cases each. Smoke scores are never treated as benchmark results.

The report includes sample standard deviations, all per-seed values in CSV, per-set success/collision/nonconvergence, network/gradient controls, original-versus-fresh differences, Welch tests matching the original report, and Holm-adjusted p-values for the 38 primary per-set comparisons. It documents the nested square-root residual and Fourier initialization differences without changing their implementation.

## Authorized expansion to all four GPUs

The user subsequently authorized GPUs 0–2 as well as GPU 3. The original scheduler was paused with SIGSTOP while its active training processes continued. The expansion coordinator preserves completed results and waits for those active checkpoints instead of restarting them. It adds GPU 3 to the new queue once the original four training processes finish.

```bash
docker run --rm --gpus all --network none --user 1006:1007 \
  --name lie-eikonal-five-seed-all-gpus --detach \
  -e CUDA_VISIBLE_DEVICES= -e MPLCONFIGDIR=/tmp/eikonal-matplotlib \
  -e OMP_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -v /media/jeffrey/Lie-Shape-NTFields:/workspace -w /workspace \
  pytorchserver:latest python -u analysis/eikonal_multigpu.py
```

Each worker uses one assigned GPU UUID through CUDA_VISIBLE_DEVICES. The coordinator uses up to two concurrent jobs per GPU and two collision workers per evaluation. A validation run on GPU 0 reproduced the GPU 3 trained weights bitwise and all ten per-case planning records exactly with two versus eight collision workers. The optimization code, five training seeds, frozen datasets, planning seed, batch size, and MPPI parameters are preserved.

The original manifest and scheduler status are retained under `expansion/`. The manifest records the authorization, GPU UUIDs, runtime adapter hash and report-source update in `execution_updates`. Results remain in the original output directory. The report describes the expanded GPU usage and automatically finalizes when all 570 evaluations have passed.

## Portable interactive artifact

After the reproduction finishes, run `python3 analysis/reproduce_eikonal/artifact.py`
to build `results/eikonal_five_seed_device3/CANVAS_REPORT.html`. This offline HTML
artifact uses the HTML Canvas API to explore every environment, test set and
success/failure metric. It includes five seed values, mean and sample SD, the full
report, static scientific figures, and embedded CSV/JSON/Markdown downloads. It
does not require a native ChatGPT Canvas session. `artifact_manifest.json` records
the artifact and input hashes; checkpoints and detailed per-case records remain
in the reproduction directory.
