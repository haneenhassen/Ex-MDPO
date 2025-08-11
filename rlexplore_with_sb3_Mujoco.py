from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env, make_atari_env #, make_atari_env_sb
#from rllte.env import make_atari_env, make_atari_env_sb
from stable_baselines3 import TRPO, PPO, SAC, MDPO
from stable_baselines3.common.evaluation import evaluate_policy
import torch as th
from torch.cuda.amp import autocast

import numpy as np
import gymnasium as gym
#from gym import spaces
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize, VecFrameStack, VecTransposeImage

import time
#from vizdoom import gymnasium_wrapper

from rllte.env import make_rllte_env #, make_dmc_env, make_mario_env, make_atari_env, make_box_env
from rllte.xplore.reward import NGU

#import gymnasium_robotics
#gym.register_envs(gymnasium_robotics)


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

    #env_kwargs = {'continuous':True}

    envs =  make_vec_env(env_id) #, env_kwargs = env_kwargs)

    th.manual_seed(random_seed)
    envs.seed(random_seed)
    np.random.seed(random_seed)
    envs.action_space.seed(random_seed)
    #envs = gym.wrappers.ResizeObservation(envs, (84, 84))

    #envs = VecTransposeImage(envs)
    #envs = VecFrameStack(envs, 4)
    #envs = SubprocVecEnv([envs])
    #envs = DummyVecEnv([envs])
    envs = VecNormalize(envs)
    #envs = VecTransposeImage(envs)
    envs.reset()

    #print(device, envs.observation_space, envs.action_space)

    # ===================== build the reward ===================== #

    #model = PPO("MlpPolicy", envs, verbose=1, seed=1, device=device, tensorboard_log='./logs/MontezumaRevenge-v4/TRPO_E/')
    start = time.localtime()
    model = MDPO("MlpPolicy", envs, batch_size=batch, verbose=1, seed=random_seed, device=device, tensorboard_log=f'./logs_f/{env_id}/MDPO/')
    model.learn(total_timesteps=10e7)

    mean_reward, std_reward = evaluate_policy(model, envs, n_eval_episodes=10)

    model.save(f'MDPO_{env_id}')
    end = time.localtime()

    f = open("outputs.txt", "a")
    f.write(f"model MDP on {env_id} finished in start time_{start} and end time_{end} mean_reward {mean_reward} std_reward {std_reward}")
    f.close()

    envs.close()
    del envs

if __name__ == "__main__":
    main()
