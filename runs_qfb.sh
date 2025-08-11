#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
batch=128
model='TRPO'
timesteps=10000
for num in 41 #42 43 
do
MIN=0
MAX=$num
env="Gravitar-v4"

# Generate a random integer
seed=$(( MIN + RANDOM % (MAX - MIN + 1) ))
echo "seed = $seed"

python rlexplore_with_sb3_Atari.py --timesteps=$timesteps --seed=$seed  --batchsize=$batch --env=$env --model=$model &
python rlexplore_with_sb3_Atari_E3B.py --timesteps=$timesteps --seed=$seed --batchsize=$batch --env=$env --model=$model &
python rlexplore_with_sb3_Atari_NGU.py --timesteps=$timesteps --seed=$seed --batchsize=$batch --env=$env --model=$model &
python rlexplore_with_sb3_Atari_RE3.py --timesteps=$timesteps --seed=$seed --batchsize=$batch --env=$env --model=$model &
done

