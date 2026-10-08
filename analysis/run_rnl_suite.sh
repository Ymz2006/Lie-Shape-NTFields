#!/bin/bash
# Planning suite for the relative-normal-loss models (two-route network, Shape-NTFields RNL
# settings, final epoch) and the no-RNL two-route baselines.  Inside the container.
until [ $(ls Experiments/3dshape_*_rnl/*_sntf/Model_Epoch_05000_*.pt 2>/dev/null | wc -l) -ge 4 ]; do sleep 20; done
SETS="Tshape3d_env1 Tshape3d_env1_tight_3d Tshape3d_env1_tight_3d_exp1 Tshape3d_env1_tight_3d_exp2 Tshape3d_env1_rotheavy Tshape3d_env1_bigangle"
ck() { ls $1/Model_Epoch_0${2}_*.pt | head -1; }
declare -A JOBS
JOBS[rnl_shape_none]="shape $(ck Experiments/3dshape_shape_metric/Tshape3d_env1_shape_metric 4000)"
JOBS[rnl_lie_none]="lie $(ck Experiments/3dshape_lie_metric/Tshape3d_env1_lie_metric 4000)"
for e in shape lie; do for m in orig lie; do
  JOBS[rnl_${e}_${m}]="$e $(ck Experiments/3dshape_${e}_metric_rnl/Tshape3d_env1_${e}_metric_rnl_${m}_sntf 5000)"
done; done
for t in rnl_shape_orig rnl_shape_lie rnl_lie_orig rnl_lie_lie rnl_shape_none rnl_lie_none; do
  set -- ${JOBS[$t]}
  bash analysis/run_plan_suite.sh $1 $2 $t $SETS > results/compare/plan/$t.suite.log 2>&1
done
