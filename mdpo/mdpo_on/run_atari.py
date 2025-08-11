#!/usr/bin/env python3
import os

from mpi4py import MPI

from stable_baselines.common import set_global_seeds
from stable_baselines import bench, logger
from stable_baselines.mdpo_on import MDPO
import stable_baselines.common.tf_util as tf_util
from stable_baselines.common.atari_wrappers import make_atari, wrap_deepmind
from stable_baselines.common.cmd_util import atari_arg_parser
from stable_baselines.common.policies import CnnPolicy


def train(env_id, num_timesteps, seed, lam, sgd_steps, klcoeff, log, tsallis_coeff):
    """
    Train TRPO model for the atari environment, for testing purposes

    :param env_id: (str) Environment ID
    :param num_timesteps: (int) The total number of samples
    :param seed: (int) The initial seed for training
    """
    rank = MPI.COMM_WORLD.Get_rank()

    with tf_util.single_threaded_session():
        rank = MPI.COMM_WORLD.Get_rank()
        seed = MPI.COMM_WORLD.Get_rank()
        logdir =None
        log_path = './Data_14-05/experiments/'+str(env_id)+'_OURS-LOADED/noent_klcoeffanneal_samesgdsteps'+str(sgd_steps)+'_'+str(tsallis_coeff)+'_'+str(seed)
        if not log:
            if rank == 0:
                logger.configure(log_path)
            else:
                logger.configure(log_path, format_strs=[])
                logger.set_level(logger.DISABLED)
        else:
            logdir = './Data_14-05/logs/'+ str(env_id)
            if rank == 0:
                logger.configure()
            else:
                logger.configure(format_strs=[])
                logger.set_level(logger.DISABLED)
        
    workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank()
    set_global_seeds(workerseed)
    env = make_atari(env_id)

    env = bench.Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)))
    env.seed(workerseed)

    env = wrap_deepmind(env)
    env.seed(workerseed)

    model = MDPO('MlpPolicy', env, tensorboard_log = logdir, timesteps_per_batch=2048, max_kl=0.01, cg_iters=10, cg_damping=0.1, entcoeff=0.0, 
                     gamma=0.99, lam=0.95, vf_iters=5, vf_stepsize=1e-3, verbose=1, seed=seed, sgd_steps=sgd_steps, klcoeff=klcoeff, method="multistep-SGD", tsallis_q=tsallis_coeff)
    model.learn(total_timesteps=100000000, seed=seed, reset_num_timesteps=False)
    env.close()
    # Free memory
    del env


def main():
    """
    Runs the test
    """
    args = atari_arg_parser().parse_args()
    train(args.env, num_timesteps=args.num_timesteps,  lam=0.98,  klcoeff=0.1, seed=0, sgd_steps=5, log=args.log, tsallis_coeff=1.0)#seed=args.seed,
    #train(args.env, num_timesteps=args.num_timesteps, seed=args.run)


if __name__ == "__main__":
    main()
