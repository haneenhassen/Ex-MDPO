#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
batch=32
timesteps=10000000
for env in "Hopper-v4" "Humanoid-v4" "Walker2d-v4" #"Ant-v2" "HalfCheetah-v2" "HumanoidStandup-v2" "Walker2d-v2"
do
for num in 41 42 43 44 45
do
MIN=0
MAX=$num

# Generate a random integer
seed=$(( MIN + RANDOM % (MAX - MIN + 1) ))
echo "seed = $seed"

python rlexplore_with_sb3_Mujoco.py --seed=$seed  --batchsize=$batch --env=$env --timesteps=$timesteps&
python rlexplore_with_sb3_Mujoco_Ex.py --seed=$seed --batchsize=$batch --env=$env --method="RE3" --timesteps=$timesteps&
python rlexplore_with_sb3_Mujoco_Ex.py --seed=$seed --batchsize=$batch --env=$env --method="E3B" --timesteps=$timesteps&
python rlexplore_with_sb3_Mujoco_Ex.py --seed=$seed --batchsize=$batch --env=$env --method="NGU" --timesteps=$timesteps&
python rlexplore_with_sb3_Mujoco_Ex.py --seed=$seed --batchsize=$batch --env=$env --method="allmax" --timesteps=$timesteps&
python rlexplore_with_sb3_Mujoco_Ex.py --seed=$seed --batchsize=$batch --env=$env --method="allsum" --timesteps=$timesteps&
done
done

