#!/bin/bash
# Evaluate the quaternion-encoding runs (quat = canonical sheet, quat_sheets = double-cover min)
# on Tshape3d_env1 with chart and SE(3) convergence, then probe them next to the chart / rot6 runs.
# usage (host): bash analysis/eval_quat_runs.sh
set -e
REPO=/home/yuchenliu/jeffrey/Lie-Shape-NTFields
run() {  # name gpu cmd
  docker run --rm --gpus "\"device=$2\"" --shm-size 8g -u 1006:1007 -v $REPO:/workspace -w /workspace --name "$1" pytorchserver bash -c "$3"
}
i=0
for tag in quatsheets quat; do
  for s in 1 2 3; do
    name=Tshape3d_env1_lie_metric_rsqrtN_${tag}_s${s}
    ck=$(ls $REPO/Experiments/3dshape_lie_metric/$name/Model_Epoch_04000_*.pt | head -1 | sed "s#$REPO/##")
    gpu=$(( i % 2 == 0 ? 1 : 2 )); i=$((i+1))
    run eval_${tag}_s${s} $gpu "PLAN_EXTRA='--workers 6' bash analysis/run_plan_suite.sh lie $ck lie_rsqrtN_${tag}_s${s} Tshape3d_env1 && PLAN_EXTRA='--group-conv --workers 6' bash analysis/run_plan_suite.sh lie $ck lie_rsqrtN_${tag}_s${s}_g Tshape3d_env1" > $REPO/results/compare/plan/lie_rsqrtN_${tag}_s${s}.suite.log 2>&1 &
    [ $((i % 3)) -eq 0 ] && wait
  done
done
wait
echo "plan evals done"
