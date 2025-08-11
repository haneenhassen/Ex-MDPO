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
from minigrid.wrappers import ReseedWrapper, FlatObsWrapper, ImgObsWrapper, FullyObsWrapper
#from gym import spaces
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor, VecNormalize, VecFrameStack, VecTransposeImage


import time
#from vizdoom import gymnasium_wrapper
from rllte.env.utils import FrameStack
from rllte.env import make_rllte_env, make_minigrid_env #, make_dmc_env, make_mario_env, make_atari_env, make_box_env
from rllte.xplore.reward import Fabric, NGU, E3B, RE3

class ThreeMax(Fabric):
    def __init__(self, m1, m2, m3):
        super().__init__(m1, m2, m3)
    
    def compute(self, samples, sync):
        rwd1, rwd2, rw3 = super().compute(samples, sync)

        return max(rwd1, rwd2, rw3)

class ThreeMin(Fabric):
    def __init__(self, m1, m2, m3):
        super().__init__(m1, m2, m3)
    
    def compute(self, samples, sync):
        rwd1, rwd2, rw3 = super().compute(samples, sync)

        return ((rwd1 + rwd2) * rw3)
    
class RLeXploreWithOnPolicyRL(BaseCallback):
    """
    A custom callback for combining RLeXplore and on-policy algorithms from SB3.
    """
    def __init__(self, irs, verbose=1):
        super(RLeXploreWithOnPolicyRL, self).__init__(verbose)
        self.irs = irs
        self.buffer = None
        #self.device = 'cuda'

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
        device = observations.device
        actions = th.as_tensor(self.locals["actions"], device=device)
        rewards = th.as_tensor(self.locals["rewards"], device=device)
        dones = th.as_tensor(self.locals["dones"], device=device)
        next_observations = th.as_tensor(self.locals["new_obs"], device=device)

        # ===================== watch the interaction ===================== #
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", help='RNG seed', type=int, default=0)
    parser.add_argument("--timesteps", type=int, default=int(1e7), help="",)
    parser.add_argument("--env", type=str, default="Asterix-v4", help="",)
    parser.add_argument("--batchsize", type=int, default=int(32), help="",)
    parser.add_argument("--method", type=str, default="NGU", help="",)

    args = parser.parse_args()
    device = 'cuda'
    n_envs = 1
    env_id = args.env #"MiniGrid-DoorKey-6x6-v0"
    random_seed = args.seed #42
    batch = args.batchsize
    episode = 1
    total_timesteps = args.timesteps
    method = args.method

    th.manual_seed(random_seed)
    np.random.seed(random_seed)

    envs =  make_atari_env(env_id, n_envs = 1, wrapper_kwargs={'action_repeat_probability':4.0, 'frame_skip':1})
    envs = VecTransposeImage(envs)
    #envs = VecFrameStack(envs, 1)
    envs.seed(random_seed)
    envs.reset()

    # ===================== build the reward ===================== #
    if method == "E3B" :
        irs = E3B(envs=envs, device=device, batch_size=64, latent_dim=30, lr=0.0005, update_proportion=0.005)
    if method == "NGU" :
        irs = NGU(envs=envs, device = device, batch_size=64, lr=0.0005, k=10, beta=0.3, latent_dim=30) 
    if method == "RE3" :
        irs = RE3(envs=envs, device=device, k=3, latent_dim=30, beta=0.005, kappa=0.00001, storage_size=1000)
    if method == "allmax" :
        irs = ThreeMax(E3B(envs=envs, device=device, batch_size=64, latent_dim=30, lr=0.0005, update_proportion=0.005),
                  NGU(envs=envs, device = device, batch_size=64, lr=0.0005, k=10, beta=0.3, latent_dim=30),
                  RE3(envs=envs, device=device, k=3, latent_dim=30, beta=0.005, kappa=0.00001, storage_size=1000))
    if method == "allsum" :
        irs = ThreeMin(E3B(envs=envs, device=device, batch_size=64, latent_dim=30, lr=0.0005, update_proportion=0.005),
                  NGU(envs=envs, device = device, batch_size=64, lr=0.0005, k=10, beta=0.3, latent_dim=30),
                  RE3(envs=envs, device=device, k=3, latent_dim=30, beta=0.005, kappa=0.00001, storage_size=1000))

    # ===================== build the reward ===================== #

    start = time.localtime()
    model = MDPO("MlpPolicy", envs, batch_size=batch, verbose=1, seed=random_seed, device=device, tensorboard_log=f'./logs_f/{env_id}/MDPO_E/{method}/{random_seed}') 
    model.learn(total_timesteps=total_timesteps, callback=RLeXploreWithOnPolicyRL(irs))
    
    mean_reward, std_reward = evaluate_policy(model, envs, n_eval_episodes=10)
    
    model.save(f'./logs_f/models/MDPO_E_{method}_{random_seed}_{env_id}')
    end = time.localtime()

    f = open("outputs.txt", "a")
    f.write(f"model MDP_E_{method} with seed = {random_seed} on {env_id} finished in start time_{start} and end time_{end} mean_reward {mean_reward} std_reward {std_reward} \n")
    f.close()
    
    envs.close()
    del envs

if __name__ == "__main__":
    main()

