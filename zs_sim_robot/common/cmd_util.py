import os
try:
    from mpi4py import MPI
except ImportError:
    MPI = None

import gym
from gym.wrappers import FlattenObservation, FilterObservation
import logger
from bench import Monitor
from common.misc_util import set_global_seeds
from common.atari_wrappers import make_atari, wrap_deepmind
from common.vec_env.subproc_vec_env import SubprocVecEnv
from common.vec_env.dummy_vec_env import DummyVecEnv
from common import retro_wrappers
from common.wrappers import ClipActionsWrapper

def make_vec_env(env_id, env_type, horizon, with_obs, with_obs_end, obs_idx, obs_pen, sparse, ah_with_goal_loc, ah_goal_loc_idx, ah_with_reach_goal, ctrl_rwd, final_rwd, ctrl_rwd_coef, goal_height, num_env, seed,
                 wrapper_kwargs=None,
                 env_kwargs=None,
                 start_index=0,
                 reward_scale=1.0,
                 flatten_dict_observations=True,
                 gamestate=None,
                 initializer=None,
                 force_dummy=False):
    """
    Create a wrapped, monitored SubprocVecEnv for Atari and MuJoCo.
    """
    wrapper_kwargs = wrapper_kwargs or {}
    env_kwargs = env_kwargs or {}
    mpi_rank = MPI.COMM_WORLD.Get_rank() if MPI else 0
    seed = seed + 10000 * mpi_rank if seed is not None else None
    logger_dir = logger.get_dir()
    set_global_seeds(seed)
    if env_type!='corl':
        env=gym.make(env_id)
        env.seed(seed)
    else:
        if env_id=='ah':
            from common.ah_env import ah_env_withobs, ah_env_noobs
            if with_obs:
                env = ah_env_withobs(obs_idx=obs_idx,env_seed=seed,ah_goal_loc_idx=ah_goal_loc_idx,ctrl_rwd=ctrl_rwd,ctrl_rwd_coef=ctrl_rwd_coef,with_reach_goal_terminal=ah_with_reach_goal,state_with_goal_loc=ah_with_goal_loc,with_obs_end=with_obs_end,sparse=sparse,obs_pen=obs_pen,final_rwd=final_rwd,horizon=horizon)
            else:
                env = ah_env_noobs(env_seed=seed,ah_goal_loc_idx=ah_goal_loc_idx,ctrl_rwd=ctrl_rwd,ctrl_rwd_coef=ctrl_rwd_coef,with_reach_goal_terminal=ah_with_reach_goal,state_with_goal_loc=ah_with_goal_loc,sparse=sparse,final_rwd=final_rwd,horizon=horizon)
        elif env_id=='corl_Reacher-v2':
            from common.corl_reacher_env import corl_reacher
            env = corl_reacher(env_seed=seed)
        elif env_id=='corl_Acrobot-v1':
            from common.corl_acrobot_env import corl_acrobot
            env = corl_acrobot(env_seed=seed,goal_height=goal_height)
    env = Monitor(env,env_id,logger_dir and os.path.join(logger_dir, str(mpi_rank) + '.' + str(start_index)),allow_early_resets=True)
    if isinstance(env.action_space, gym.spaces.Box):
        env = ClipActionsWrapper(env)
    return env
        

def arg_parser():
    """
    Create an empty argparse.ArgumentParser.
    """
    import argparse
    return argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)


def common_arg_parser():
    """
    Create an argparse.ArgumentParser for run_mujoco.py.
    """
    parser = arg_parser()
    parser.add_argument('--env', help='environment ID', type=str, default='Reacher-v2')
    parser.add_argument('--env_type', help='type of environment, used when the environment type cannot be automatically determined', type=str)
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--with_obs', help='adaptive hand with obstacles or not', type=int, default=0)
    parser.add_argument('--with_obs_end', help='adaptive hand with obstacles ending or not', type=int, default=1)
    parser.add_argument('--obs_idx', help='obstacle index for adaptive hand', type=int, default=20)
    parser.add_argument('--ah_with_goal_loc', help='state with goal loc or not for adaptive hand', type=int, default=0)
    parser.add_argument('--ah_goal_loc_idx', help='goal loc index for adaptive hand', type=int, default=8)
    parser.add_argument('--ah_with_reach_goal', help='with reach goal as terminal or not for adaptive hand', type=int, default=1)
    parser.add_argument('--ctrl_rwd', help='adaptive hand with control reward or not', type=int, default=0)
    parser.add_argument('--ctrl_rwd_coef', help='adaptive hand control reward coefficient', type=int, default=1)
    parser.add_argument('--horizon', help='adaptive hand horizon', type=int, default=2000)
    parser.add_argument('--goal_height', help='goal height for Acrobot-v1', type=float, default=1.)
    parser.add_argument('--final_rwd', help='final reward for adaptive hand', type=float, default=0.)
    parser.add_argument('--sparse', help='sparse reward or not', type=int, default=0)
    parser.add_argument('--obs_pen', help='obstacle penalty', type=float, default=1e6)
    parser.add_argument('--alg', help='Algorithm', type=str, default='ppo2')
    parser.add_argument('--num_timesteps', type=float, default=1e6),
    parser.add_argument('--network', help='network type (mlp, cnn, lstm, cnn_lstm, conv_only)', default=None)
    parser.add_argument('--gamestate', help='game state to load (so far only used in retro games)', default=None)
    parser.add_argument('--num_env', help='Number of environment copies being run in parallel. When not specified, set to number of cpus for Atari, and to 1 for Mujoco', default=None, type=int)
    parser.add_argument('--reward_scale', help='Reward scale factor. Default: 1.0', default=1.0, type=float)
    parser.add_argument('--save_path', help='Path to save trained model to', default=None, type=str)
    parser.add_argument('--save_video_interval', help='Save video every x steps (0 = disabled)', default=0, type=int)
    parser.add_argument('--save_video_length', help='Length of recorded video. Default: 200', default=200, type=int)
    parser.add_argument('--log_path', help='Directory to save learning curve data.', default=None, type=str)
    parser.add_argument('--play', default=False, action='store_true')
    return parser


def parse_unknown_args(args):
    """
    Parse arguments not consumed by arg parser into a dictionary
    """
    retval = {}
    preceded_by_key = False
    for arg in args:
        if arg.startswith('--'):
            if '=' in arg:
                key = arg.split('=')[0][2:]
                value = arg.split('=')[1]
                retval[key] = value
            else:
                key = arg[2:]
                preceded_by_key = True
        elif preceded_by_key:
            retval[key] = arg
            preceded_by_key = False

    return retval
