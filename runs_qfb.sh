#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
batch=128
timesteps=10000
for num in 41 42 43 
do
MIN=0
MAX=$num

# Generate a random integer
seed=$(( MIN + RANDOM % (MAX - MIN + 1) ))
echo "seed = $seed"

python rlexplore_with_sb3_Atari.py --timesteps=$timesteps --seed=$seed  --batchsize=$batch  &
python rlexplore_with_sb3_Atari_E3B.py --timesteps=$timesteps --seed=$seed --batchsize=$batch  &
python rlexplore_with_sb3_Atari_NGU.py --timesteps=$timesteps --seed=$seed --batchsize=$batch  &
python rlexplore_with_sb3_Atari_RE3.py --timesteps=$timesteps --seed=$seed --batchsize=$batch &
done

