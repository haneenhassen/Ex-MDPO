#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
batch=64
steps=10000
envs="l2rpn_wcci_2022"
for num in 41 42 43
do
MIN=0
MAX=$num

# Generate a random integer
seed=$(( MIN + RANDOM % (MAX - MIN + 1) ))
echo "seed = $seed"

python rlexplore_with_sb3_Grid2Op.py --seed=$seed --batchsize=$batch --timesteps=$steps --env=$envs &
python rlexplore_with_sb3_Grid2Op_E3B.py --seed=$seed --batchsize=$batch  --timesteps=$steps --env=$envs  &
python rlexplore_with_sb3_Grid2Op_NGU.py --seed=$seed  --batchsize=$batch  --timesteps=$steps --env=$envs  &
python rlexplore_with_sb3_Grid2Op_RE3.py --seed=$seed --batchsize=$batch  --timesteps=$steps --env=$envs  &
done

