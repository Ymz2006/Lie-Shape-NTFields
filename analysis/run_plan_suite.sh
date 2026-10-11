#!/bin/bash
# Run analysis/plan_eval.py for one checkpoint over the Tshape3d_env1 test suite.
# usage (inside the container, cwd /workspace, GPU pinned via CUDA_VISIBLE_DEVICES):
#   bash analysis/run_plan_suite.sh <kind> <checkpoint> <tag> [set ...]
# Extra plan_eval flags go in $PLAN_EXTRA, e.g. PLAN_EXTRA=--group-conv.
kind=$1; ckpt=$2; tag=$3; shift 3
sets=${@:-"Tshape3d_env1 Tshape3d_env1_tight_3d Tshape3d_env1_tight_3d_exp1 Tshape3d_env1_tight_3d_exp2 Tshape3d_env1_tight_3d_exp3 Tshape3d_env1_tight001_3d Tshape3d_env1_rotheavy Tshape3d_env1_bigangle Tshape3d_env1_transonly"}
out=results/compare/plan/$tag
mkdir -p $out
for s in $sets; do
  [ -f testing_data/3dshape/$s/sampled_points.npy ] || { echo "skip $s (missing)"; continue; }
  [ -f $out/$s.json ] && { echo "skip $s (done)"; continue; }
  OMP_NUM_THREADS=1 python -u analysis/plan_eval.py --kind $kind --checkpoint $ckpt \
    --dataPath testing_data/3dshape/$s --out $out/$s.json --device cuda:0 \
    --batch 125 --workers 8 --seed 0 $PLAN_EXTRA 2>&1 | grep -v "UserWarning\|warnings.warn" | tail -1 > $out/$s.log
  echo "$tag $s: $(cut -c1-200 $out/$s.log)"
done
