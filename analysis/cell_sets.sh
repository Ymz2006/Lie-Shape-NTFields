#!/bin/bash
# Probe / rotation / cut-locus sets for the extra environments, when GPU memory allows.
free_gpu() { nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits |
  awk -F', ' -v need=$1 '($1==0||$1==2) && ($3-$2)>=need {print $1; exit}'; }
for cell in Lshape3d_env3 Ashape3d_env4; do
  until g=$(free_gpu 1000); [ -n "$g" ]; do sleep 20; done
  echo "$(date +%T) $cell on GPU $g"
  CUDA_VISIBLE_DEVICES=$g PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -u analysis/make_sets.py \
    --cell $cell --out results/compare/sets_$cell --device cuda:0 --chunk 16 --only probe,rot,cut --seed 21 2>&1 |
    grep --line-buffered 'probe:\|goal pool\|rot pairs\|cut set\|done in\|Error\|error'
done
echo "$(date +%T) cell sets done"
