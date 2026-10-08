#!/bin/bash
# Wait for GPU memory, build the rotation / cut-locus / planning sets, label and
# plan the new sets for the seed-1 checkpoints, then resume the training queue.
# Runs inside the container (cwd /workspace).
free_gpu() {  # prints the index of the GPU (0 or 2) with >= $1 MB free, else nothing
  nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits |
    awk -F', ' -v need=$1 '($1==0||$1==2) && ($3-$2)>=need {print $1; exit}'
}
until g=$(free_gpu 1000); [ -n "$g" ]; do sleep 20; done
echo "$(date +%T) building sets on GPU $g"
CUDA_VISIBLE_DEVICES=$g PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -u analysis/make_sets.py \
  --out results/compare/sets --device cuda:0 --chunk 16 --only rot,cut,plan --seed 7 2>&1 |
  grep --line-buffered 'rot pairs\|cut set\|try\|wrote\|done in\|Error\|error' > results/compare/sets/make_sets5.log
echo "$(date +%T) sets done"
OMP_NUM_THREADS=1 python -u analysis/lin_valid.py testing_data/3dshape/Tshape3d_env1_rotheavy \
  testing_data/3dshape/Tshape3d_env1_bigangle testing_data/3dshape/Tshape3d_env1_transonly 2>&1 |
  grep -v 'Body\|load_obj\|wall\|\[' >> results/compare/linvalid.log
M=Experiments/3dshape_metric_baseline/Tshape3d_env1_metric_baseline/Model_Epoch_04000_ValLoss_1.808295e-03.pt
SH=Experiments/3dshape_shape_metric/Tshape3d_env1_shape_metric_origarch/Model_Epoch_04000_ValLoss_9.996648e-03.pt
LI=Experiments/3dshape_lie_metric/Tshape3d_env1_lie_metric_origarch/Model_Epoch_04000_ValLoss_4.130011e-03.pt
NEW="Tshape3d_env1_rotheavy Tshape3d_env1_bigangle Tshape3d_env1_transonly"
for pair in "metric $M metric_s1" "shape $SH shape_s1" "lie $LI lie_s1"; do
  set -- $pair
  until g=$(free_gpu 700); [ -n "$g" ]; do sleep 20; done
  CUDA_VISIBLE_DEVICES=$g PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    bash analysis/run_plan_suite.sh $1 $2 $3 $NEW >> results/compare/plan/$3.suite.log 2>&1 &
  sleep 60
done
echo "$(date +%T) resuming training queue"
NTF_NO_PLOT=1 python -u analysis/train_queue.py --settle 75 --max_jobs 7 --need 1200 --need_metric 1200 \
  --jobs Tshape3d_env1_lie_s3,Tshape3d_env1_shape_s3,Tshape3d_env1_metric_s3,Ashape3d_env4_lie_s1,Ashape3d_env4_shape_s1,Ashape3d_env4_metric_s1 \
  > results/compare/train_queue5.log 2>&1
wait
echo "$(date +%T) after_sets finished"
