import os
import argparse
import re
import numpy as np
import time


import copy
from typing import Dict, Literal, Any
import json

from stable_baselines3.common.base_class import BaseAlgorithm 
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback 
from stable_baselines3.common.env_util import make_vec_env, make_atari_env #, make_atari_env_sb
from stable_baselines3 import TRPO, PPO, SAC, MDPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor, VecNormalize, VecFrameStack, VecTransposeImage
from stable_baselines3.common.noise import NormalActionNoise, ActionNoise
import torch as th
from torch.utils.tensorboard import SummaryWriter

import gymnasium as gym
from gymnasium import Env
from gymnasium.spaces import Discrete, MultiDiscrete, Box

from rllte.env.utils import FrameStack
from rllte.xplore.reward import NGU, RE3, E3B

from datetime import datetime as dt
from replay_buffer.generalizing_replay_wrapper import GeneralizingReplayWrapper

import grid2op
from lightsim2grid import LightSimBackend
from grid2op.Reward import L2RPNSandBoxScore
from custom_reward import *
from grid2op.gym_compat import GymEnv, BoxGymObsSpace, DiscreteActSpace, BoxGymActSpace, MultiDiscreteActSpace
from lightsim2grid import LightSimBackend


class RLeXploreWithOnPolicyRL(BaseCallback):
    """
    A custom callback for combining RLeXplore and on-policy algorithms from SB3.
    """
    def __init__(self, irs, verbose=2):
        super(RLeXploreWithOnPolicyRL, self).__init__(verbose)
        self.irs = irs
        self.buffer = None
        self.device = 'cuda'

    def init_callback(self, model: BaseAlgorithm) -> None:
        super().init_callback(model)
        self.buffer = self.model.rollout_buffer

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        #print('h_on_step')
        observations = self.locals["obs_tensor"]
        device = self.device #observations.device
        actions = th.as_tensor(self.locals["actions"], device=device)
        rewards = th.as_tensor(self.locals["rewards"], device=device)
        dones = th.as_tensor(self.locals["dones"], device=device)
        next_observations = th.as_tensor(self.locals["new_obs"], device=device)

        # ===================== watch the interaction ===================== #
        #print("Calling self.irs.watch...")
        self.irs.watch(observations, actions, rewards, dones, dones, next_observations)
        # ===================== watch the interaction ===================== #
        return True

    def _on_rollout_end(self) -> None:
        # ===================== compute the intrinsic rewards ===================== #
        # prepare the data samples
        obs = th.as_tensor(self.buffer.observations)
        # get the new observations
        new_obs = obs.clone()
        new_obs[:-1] = obs[1:]
        new_obs[-1] = th.as_tensor(self.locals["new_obs"])
        actions = th.as_tensor(self.buffer.actions)
        rewards = th.as_tensor(self.buffer.rewards)
        dones = th.as_tensor(self.buffer.episode_starts)
        print(obs.shape, actions.shape, rewards.shape, dones.shape, obs.shape)
        # compute the intrinsic rewards
        intrinsic_rewards = self.irs.compute(
            samples=dict(observations=obs, actions=actions, 
                         rewards=rewards, terminateds=dones, 
                         truncateds=dones, next_observations=new_obs),
            sync=True)
        # add the intrinsic rewards to the buffer
        self.buffer.advantages += intrinsic_rewards.cpu().numpy()
        self.buffer.returns += intrinsic_rewards.cpu().numpy()
        # ===================== compute the intrinsic rewards ===================== #

class Grid2opEnvWrapper(Env):
    def __init__(self,
                 env_config: Dict[Literal["backend_cls",
                                          "backend_options",
                                          "env_name",
                                          "env_is_test",
                                          "obs_attr_to_keep",
                                          "act_type",
                                          "reward_cls",
                                          "act_attr_to_keep"],
                                  Any] = None):
        super().__init__()
        if env_config is None:
            env_config = {}

        # handle the backend
        backend_cls = LightSimBackend
        if "backend_cls" in env_config:
            backend_cls = env_config["backend_cls"]
        backend_options = {}
        if "backend_options" in env_config:
            backend_options = env_config["backend_options"]
        backend = backend_cls(**backend_options)
        # handle the reward
        reward_cls = L2RPNSandBoxScore
        if "reward_cls" in env_config:
            reward_cls = env_config["reward_cls"]
        reward = reward_cls
        # create the grid2op environment
        env_name = "l2rpn_case14_sandbox"
        if "env_name" in env_config:
            env_name = env_config["env_name"]
        if "env_is_test" in env_config:
            is_test = bool(env_config["env_is_test"])
        else:
            is_test = False
        self._g2op_env = grid2op.make(env_name, reward_class=reward, backend=backend, test=is_test, other_rewards={'loss': LossReward})
        # NB by default this might be really slow (when the environment is reset)
        # see https://grid2op.readthedocs.io/en/latest/data_pipeline.html for maybe 10x speed ups !
        # TODO customize reward or action_class for example !

        self._gym_env = GymEnv(self._g2op_env)

        # customize observation space
        obs_attr_to_keep = ["rho", "p_or", "gen_p", "load_p"]
        if "obs_attr_to_keep" in env_config:
            obs_attr_to_keep = copy.deepcopy(env_config["obs_attr_to_keep"])
        self._gym_env.observation_space.close()
        self._gym_env.observation_space = BoxGymObsSpace(self._g2op_env.observation_space,
                                                         attr_to_keep=obs_attr_to_keep
                                                         )
        # export observation space for the Grid2opEnv
        self.observation_space = Box(shape=self._gym_env.observation_space.shape,
                                     low=self._gym_env.observation_space.low,
                                     high=self._gym_env.observation_space.high)

        # customize the action space
        act_type = "discrete"
        if "act_type" in env_config:
            act_type = env_config["act_type"]

        self._gym_env.action_space.close()
        if act_type == "discrete":
            # user wants a discrete action space
            act_attr_to_keep =  ["set_line_status_simple", "set_bus"]
            if "act_attr_to_keep" in env_config:
                act_attr_to_keep = copy.deepcopy(env_config["act_attr_to_keep"])
            self._gym_env.action_space = DiscreteActSpace(self._g2op_env.action_space,
                                                          attr_to_keep=act_attr_to_keep)
            self.action_space = Discrete(self._gym_env.action_space.n)
        elif act_type == "box":
            # user wants continuous action space
            act_attr_to_keep =  ["redispatch", "set_storage", "curtail"]
            if "act_attr_to_keep" in env_config:
                act_attr_to_keep = copy.deepcopy(env_config["act_attr_to_keep"])
            self._gym_env.action_space = BoxGymActSpace(self._g2op_env.action_space,
                                                        attr_to_keep=act_attr_to_keep)
            self.action_space = Box(shape=self._gym_env.action_space.shape,
                                    low=self._gym_env.action_space.low,
                                    high=self._gym_env.action_space.high)
        elif act_type == "multi_discrete":
            # user wants a multi-discrete action space
            act_attr_to_keep = ["one_line_set", "one_sub_set"]
            if "act_attr_to_keep" in env_config:
                act_attr_to_keep = copy.deepcopy(env_config["act_attr_to_keep"])
            self._gym_env.action_space = MultiDiscreteActSpace(self._g2op_env.action_space,
                                                               attr_to_keep=act_attr_to_keep)
            self.action_space = MultiDiscrete(self._gym_env.action_space.nvec)
        else:
            raise NotImplementedError(f"action type '{act_type}' is not currently supported.")
            
            
    def reset(self, seed=None, options=None):
        # use default _gym_env (from grid2op.gym_compat module)
        return self._gym_env.reset(seed=seed, options=options)
        
    def step(self, action):
        # use default _gym_env (from grid2op.gym_compat module)
        return self._gym_env.step(action)
        


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", help='RNG seed', type=int, default=0)
    parser.add_argument("--timesteps", type=int, default=int(1e7), help="",)
    parser.add_argument("--env", type=str, default="l2rpn_icaps_2021", help="",)
    parser.add_argument("--batchsize", type=int, default=int(32), help="",)
    parser.add_argument("--info", type=str, default="", help="",)

    args = parser.parse_args()
    device = 'cuda'
    n_envs = 1
    env_id = args.env 
    irs_name = "NGU"
    random_seed = args.seed #42
    batch = args.batchsize
    episode = 1
    total_timesteps = args.timesteps
    info = args.info
    save_dir = f'./logs_f/{env_id}/{total_timesteps}/{info}/E_MDPO/{irs_name}/{random_seed}/'
    th.manual_seed(random_seed)
    np.random.seed(random_seed)

   #grid2op_env = grid2op.make(env_path, test=True, reward_class=L2RPNSandBoxScore, backend=LightSimBackend())
                #other_rewards={'loss': LossReward})
    env_config3 = {"env_name": env_id,
               "env_is_test": True,
               "act_type": "box",
              "backend_cls" : LightSimBackend(),
              }
    #gym_env = Grid2OpEnvWrapper()  # grid2op_env is your original Grid2Op env
    envs = make_vec_env(lambda : Grid2opEnvWrapper(), n_envs=n_envs)
    
    # ===================== build the reward ===================== #
    irs = NGU(envs=envs, device = device, batch_size=32, latent_dim=30, lr=5e-4, k=10, beta=0.9, update_proportion=0.5, encoder_model="espeholt")
    # ===================== build the reward ===================== #

    start = time.localtime()
    model = MDPO("MlpPolicy", envs, learning_rate=1e-4, batch_size=batch, n_steps=100, ent_coef=0.00001, sgd_steps=10, verbose=1, seed=random_seed, device=device, tensorboard_log=save_dir)
    model.learn(total_timesteps=total_timesteps, callback=[RLeXploreWithOnPolicyRL(irs)])
    
    mean_reward, std_reward = evaluate_policy(model, envs, n_eval_episodes=10)
    
    model.save(f'{save_dir}/models/MDPO_E_{irs_name}_{random_seed}_{env_id}')
    end = time.localtime()

    f = open(f"outputs_{env_id.split('_')[0]}.txt", "a")
    f.write(f"model MDP_E_{irs_name} with seed = {random_seed} on {env_id} finished in start time_{start} and end time_{end} mean_reward {mean_reward} std_reward {std_reward} \n")
    f.close()
    
    envs.close()
    del envs

if __name__ == "__main__":
    main()
