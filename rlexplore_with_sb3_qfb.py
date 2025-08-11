import os
import argparse

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env, make_atari_env #, make_atari_env_sb
#from rllte.env import make_atari_env, make_atari_env_sb
from stable_baselines3 import TRPO, PPO, SAC, MDPO
from stable_baselines3.common.evaluation import evaluate_policy
import torch as th
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import gymnasium as gym
#from qfb_env import QFBEnv
from qfb_env.envs.qfb_nonlinear_env import QFBNLEnv
from qfb_env.envs.qfb_env import QFBEnv
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor, VecNormalize, VecFrameStack, VecTransposeImage
from stable_baselines3.common.noise import NormalActionNoise, ActionNoise

#from utils import make_atari_env
import time
#from vizdoom import gymnasium_wrapper
from rllte.env.utils import FrameStack

import tensorflow as tf
from NAF2.naf2 import NAF2
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from datetime import datetime as dt

from qfb_env.envs.qfb_nonlinear_env import QFBNLEnv
from qfb_env.envs.qfb_env import QFBEnv

from replay_buffer.generalizing_replay_wrapper import GeneralizingReplayWrapper

# import warnings
# warnings.filterwarnings("ignore", category=FutureWarning)

class EvaluationAndCheckpointCallback(BaseCallback):
    MAX_EPS = 20

    def __init__(self, env, save_dir, EVAL_FREQ=100, CHKPT_FREQ=1000, verbose=1):
        super(EvaluationAndCheckpointCallback, self).__init__(verbose)
        self.env = env
        self.save_dir = save_dir
        self.model_name = os.path.split(save_dir)[-1]
        self.EVAL_FREQ = EVAL_FREQ
        self.CHKPT_FREQ = CHKPT_FREQ

        self.gamma = 0.99
        self.discounts = [np.power(self.gamma, i) for i in range(self.env.EPISODE_LENGTH_LIMIT)]

        self.writer = SummaryWriter(log_dir=os.path.join(save_dir, "tensorboard_logs"))  # TensorBoard logger

    def quick_save(self, suffix=None):
        save_path = os.path.join(self.save_dir, f'{self.model_name}_{suffix if suffix else self.num_timesteps}_steps')
        self.model.save(save_path)
        if self.verbose > 0:
            print(f'Model saved to: {save_path}')

    def _on_step(self):
        if self.num_timesteps % self.EVAL_FREQ == 0:
            returns, ep_lens, success = [], [], []

            # Episode loop
            for _ in range(self.MAX_EPS):
                o = self.env.reset()
                ep_return, step, d = 0.0, 0, False

                while not d:
                    a = self.model.predict(o, deterministic=True)[0]
                    o, r, d, _ = self.env.step(a)
                    ep_return += r
                    step += 1

                ep_lens.append(step)
                returns.append(ep_return / self.env.REWARD_SCALE)
                success.append(1.0 if step < self.env.max_steps else 0.0)

            # Compute averages
            avg_return, avg_length, success_rate = np.mean(returns), np.mean(ep_lens), np.mean(success) * 100

            # Log results to TensorBoard
            self.writer.add_scalar("eval/episode_return", avg_return, self.num_timesteps)
            self.writer.add_scalar("eval/episode_length", avg_length, self.num_timesteps)
            self.writer.add_scalar("eval/success_rate", success_rate, self.num_timesteps)
            self.writer.flush()

            # Save model if successful
            if success_rate > 0:
                self.quick_save()
                if self.verbose > 1:
                    print(f"Saving model checkpoint at {self.num_timesteps} steps.")

        # Save periodic checkpoints
        if self.num_timesteps % self.CHKPT_FREQ == 0:
            self.quick_save()

        return True

class DecayingNormalActionNoise(ActionNoise):
        def __init__(self, n_act, eps_thresh):
                self.eps_thresh = eps_thresh
                self.cur_ep = 0
                # self.n_act = n_act

        def reset(self) -> None:
                self.cur_ep += 1

        def __call__(self):
                # return np.random.randn(self.n_act) * max(1 - self.cur_ep / self.eps_thresh, 0)
                return np.random.randn() * max(1 - self.cur_ep / self.eps_thresh, 0)

        def __repr__(self):
                return f'{type(self).__name__}__eps-thresh={self.eps_thresh}'

class DecayingLearningRate():
        def __init__(self, init_lr, final_lr, frac_decay):
                self.init_lr = init_lr
                self.final_lr = final_lr
                self.frac_decay = frac_decay

        def __call__(self, frac):
                return self.init_lr - (self.init_lr - self.final_lr) * min(((1-frac)/self.frac_decay), 1)

        def __repr__(self):
                return f'{type(self).__name__}__init-lr={self.init_lr}__final-lr={self.final_lr}__frac-decay={self.frac_decay}'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", help='RNG seed', type=int, default=1)
    parser.add_argument("--timesteps", type=int, default=int(1e7), help="",)
    parser.add_argument("--batchsize", type=int, default=int(32), help="",)
    parser.add_argument("--env", type=str, default="QEB", help="",)
    
    args = parser.parse_args()
    device = 'cuda'
    n_envs = 1
    env_id = args.env #"MiniGrid-DoorKey-6x6-v0"
    random_seed = args.seed #42
    batch = args.batchsize
    episode = 1
    total_timesteps = args.timesteps
    save_dir = f'./logs_f/QEB_envs/MDPO/{random_seed}/'

    th.manual_seed(random_seed)
    np.random.seed(random_seed)


    env_kwargs = dict(rm_loc=os.path.join('metadata', 'LHC_TRM_B1.response'),
                                                  calibration_loc=os.path.join('metadata', 'LHC_circuit.calibration'),
                                                  perturb_state=False,
                                                  noise_std=0.0)
 
    envs = QFBNLEnv(**env_kwargs)
    eval_env = QFBNLEnv(**env_kwargs) 
    #envs = ObservationWrapper(envs)
    #envs = VecTransposeImage(envs)
    envs = DummyVecEnv([lambda:envs for _ in range(n_envs)])
    envs = VecMonitor(envs)

    callback_chkpt = CheckpointCallback(save_freq=1000, save_path=save_dir, name_prefix='MDPO')
    eval_callback = EvaluationAndCheckpointCallback(eval_env, save_dir=save_dir)

    start = time.localtime()
    model = MDPO("MlpPolicy", envs, learning_rate=2.5e-4, batch_size=batch, n_steps=128,  ent_coef=0.00001, sgd_steps=10, verbose=1, seed=random_seed, device=device, tensorboard_log=f'./logs_f/{env_id}/MDPO/{random_seed}')
    model.learn(total_timesteps=total_timesteps, callback=[eval_callback, callback_chkpt])

    mean_reward, std_reward = evaluate_policy(model, envs, n_eval_episodes=10)
    
    model.save(f'./logs_f/models/{model}_{random_seed}_{env_id}_{total_timesteps}')
    end = time.localtime()

    f = open("outputs.txt", "a")
    f.write(f"model {model} with seed = {random_seed} on {env_id} timesteps = {total_timesteps} finished in start time_{start} and end time_{end} mean_reward {mean_reward} std_reward {std_reward} \n")
    f.close()
    
    envs.close()
    del envs


if __name__ == "__main__":
    main()
