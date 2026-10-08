#!/bin/bash
# Second seed for the shape-Eikonal RNL comparison (two-route network) + its no-RNL baseline.
# usage: rnl_seed2.sh <orig|lie|none>   (inside the container; GPU pinned by the caller)
m=$1
SETS="Tshape3d_env1 Tshape3d_env1_tight_3d Tshape3d_env1_tight_3d_exp1 Tshape3d_env1_tight_3d_exp2 Tshape3d_env1_rotheavy Tshape3d_env1_bigangle"
if [ "$m" = none ]; then
  name=Tshape3d_env1_shape_metric_s2; mp=./Experiments/3dshape_shape_metric; ep=4000
else
  name=Tshape3d_env1_shape_metric_rnl_${m}_sntf_s2; mp=./Experiments/3dshape_shape_metric_rnl; ep=5000
fi
OMP_NUM_THREADS=4 python train/train_3dshape.py --dataPath datasets/3dshape/Tshape3d_env1/ --models shape_metric \
  --rnl $m --epochs $ep --device cuda:0 --modelPath $mp --name $name --no-wandb > Experiments/${name}_train.log 2>&1
ck=$(ls $mp/$name/Model_Epoch_0${ep}_*.pt)
bash analysis/run_plan_suite.sh shape $ck rnl_shape_${m}_s2 $SETS > results/compare/plan/rnl_shape_${m}_s2.suite.log 2>&1
