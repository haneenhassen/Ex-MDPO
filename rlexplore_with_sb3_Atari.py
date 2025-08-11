import argparse

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env, make_atari_env #, make_atari_env_sb
#from rllte.env import make_atari_env, make_atari_env_sb
#from stable_baselines3 import MDPO
from stable_baselines3 import MDPO1 as MDPO
from stable_baselines3.common.evaluation import evaluate_policy
import torch as th
from torch.cuda.amp import autocast

import numpy as np
import gymnasium as gym
#from gym import spaces
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor, VecNormalize, VecFrameStack, VecTransposeImage

#from utils import make_atari_env
import time
#from vizdoom import gymnasium_wrapper
from rllte.env.utils import FrameStack


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", help='RNG seed', type=int, default=1)
    parser.add_argument("--timesteps", type=int, default=int(1e7), help="",)
    parser.add_argument("--batchsize", type=int, default=int(32), help="",)
    parser.add_argument("--env", type=str, default="Asterix-v4", help="",)

    args = parser.parse_args()
    device = 'cuda'
    n_envs = 1
    env_id = args.env #"MiniGrid-DoorKey-6x6-v0"
    random_seed = args.seed #42
    batch = args.batchsize
    episode = 1
    total_timesteps = args.timesteps

    th.manual_seed(random_seed)
    np.random.seed(random_seed)

    envs =  make_atari_env(env_id, n_envs = 1, wrapper_kwargs={'action_repeat_probability':4.0,'frame_skip':1})
    envs = VecTransposeImage(envs)
    #envs = VecFrameStack(envs, 1)

    envs.seed(random_seed)
    envs.reset()


    start = time.localtime()
    model = MDPO("MlpPolicy", envs, batch_size=batch, verbose=1, seed=random_seed, device=device, tensorboard_log=f'./logs_f/{env_id}/MDPO/{random_seed}')
    model.learn(total_timesteps=total_timesteps)
    
    mean_reward, std_reward = evaluate_policy(model, envs, n_eval_episodes=10)
    
    model.save(f'./logs_f/models/MDPO_{random_seed}_{env_id}')
    end = time.localtime()

    f = open("outputs.txt", "a")
    f.write(f"model MDP with seed = {random_seed} on {env_id} finished in start time_{start} and end time_{end} mean_reward {mean_reward} std_reward {std_reward} \n")
    f.close()
    
    envs.close()
    del envs


if __name__ == "__main__":
    main()
