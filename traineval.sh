#!/bin/bash

export case=$1
export conf=$2
export gpu_id=$3

export CUDA_VISIBLE_DEVICES=$gpu_id

echo "RECONSTRUCT CASE: $case" 
echo "CONF: $conf" 
echo "GPU_ID = $gpu_id"

echo "python exp_runner.py --mode train --conf $conf --case $case"
python reconstruct_mesh.py --mode train --conf $conf --case $case

echo "python extract_mesh.py --conf $conf --case $case --eval_metric"
python extract_mesh.py --conf $conf --case $case --eval_metric

echo "NeuDA train & evaluation done!"
