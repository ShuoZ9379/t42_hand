import gym
import sys, os, pickle
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import tensorflow as tf
from common.tf_util import get_session
from common.cmd_util import make_vec_env
from common.policies import build_policy
from common.policies_aip import build_policy as build_policy_aip
from r_diff.model_config import get_make_mlp_model
from r_diff.r_diff import R_diff_model
from common.input import observation_placeholder
from ppo2.model import Model
def build_env(env_id,seed=0,goal_height=1.0,ho=0,ah_goal_loc_idx=8,ctrl_rwd=1,env_type='corl',with_obs=0,obs_idx=20,ah_with_goal_loc=0,ah_with_reach_goal=1,ctrl_rwd_coef=1,dm_epochs=500):
    config = tf.ConfigProto(allow_soft_placement=True,intra_op_parallelism_threads=1,inter_op_parallelism_threads=1)
    config.gpu_options.allow_growth = True
    get_session(config=config)
    env = make_vec_env(env_id, env_type, 2000, with_obs, 1, obs_idx, 1e6, 0, ah_with_goal_loc, ah_goal_loc_idx, ah_with_reach_goal, ctrl_rwd, 0, ctrl_rwd_coef, ho, goal_height, 1, seed, reward_scale=1.0, flatten_dict_observations=True,dm_epochs=dm_epochs)
    return env


def cl_one_eps_reacher(env,cur_seed):
    corl_env=build_env('corl_Reacher-v2',seed=0,goal_height=1.0,ho=0.999,dm_epochs=100)
    ob_space = corl_env.observation_space
    ac_space = corl_env.action_space
    policy = build_policy(corl_env, 'mlp', value_network='copy')
    with tf.Session(graph=tf.Graph()):
        tf.set_random_seed(cur_seed)
        model = Model(policy=policy, env_type='corl', ob_space=ob_space, ac_space=ac_space, nbatch_act=1, nbatch_train=64,ent_coef=0.0, vf_coef=0.5,max_grad_norm=0.5, comm=None, mpi_rank_weight=1)
        model.load('./ppo2_results/models_bkup/corl_Reacher-v2/seed_ho0.999_7')

        env_aip=build_env('Reacher-v2',seed=0,goal_height=1.0,ho=0.999,ah_goal_loc_idx=8,ctrl_rwd=1,env_type='mujoco',dm_epochs=100)
        ob_space = env_aip.observation_space
        ac_space = env_aip.action_space
        policy_aip = build_policy_aip(env_aip, 'mlp', value_network='copy')
        make_model = get_make_mlp_model(num_fc=2, num_fwd_hidden=500, layer_norm=False)
        r_diff_model = R_diff_model(env=env_aip, env_id='Reacher-v2', make_model=make_model, update_epochs=5,  batch_size=512, r_diff_classify=False)
        ob = observation_placeholder(ob_space,name='Ob')
        ref_ac = observation_placeholder(ac_space,name='Ref_ac')
        alpha_holder=tf.placeholder(shape=(None,1), dtype=np.float32, name='Alpha')
        model_suf='_models_ho'+str(0.999)+'_dmep'+str(100)+'_prsd'+str(7)
        with tf.variable_scope("pi_aip"):
            aip_pi=policy_aip(ablation='auto',env_type='mujoco',env_id='Reacher-v2',ref_type='ppo',policy_ref=model,r_diff_model=r_diff_model,alpha_func='squared',observ_placeholder=ob,ac_placeholder=ref_ac,alpha_placeholder=alpha_holder)
        aip_pi.load('./icra_results/reacher'+model_suf+'/AIP_alpha_auto_seed_0')

        all_obs=env.reset().reshape(1,-1)
        rwds=np.empty(0)
        dones=np.empty(0)
        acts=np.empty((0,2))
        trans = 0
        done=False
        goal=all_obs[0,4:6]
        pt=1
        while not done:
            act, v, state, neglogp, action_ref, alpha = aip_pi.step(all_obs[-1,:],stochastic=True,ref_stochastic=False)
            if pt:
                print(action_ref)
                pt=0
            obs, rwd, done, _ = env.step(act)
            all_obs=np.concatenate((all_obs,obs.reshape(1,-1)),axis=0)
            rwds=np.concatenate((rwds,np.array([rwd])),axis=0)
            dones=np.concatenate((dones,np.array([done])),axis=0)
            acts=np.concatenate((acts,act.reshape(1,-1)),axis=0)
            trans += 1    
    return all_obs,acts,rwds,dones,goal,trans

def process_raw_eps(env_name,raw_eps,acts,use_partial_state):
    if env_name=='Reacher-v2':
        if use_partial_state:
            raw_eps=np.concatenate((raw_eps[:,:4],raw_eps[:,6:8],raw_eps[:,8:10]+raw_eps[:,4:6]),axis=1)
        processed_eps=np.concatenate((raw_eps[:-1,:],acts,np.roll(raw_eps,-1,axis=0)[:-1,:]),axis=1)
        return processed_eps
    else:
        raise Exception('Not implemented!')
        
def main(env_name='Reacher-v2',init_seed=1):
    eps_data_folder='./mjo_eps_data'
    if not os.path.exists(eps_data_folder):
        os.makedirs(eps_data_folder)
    data_file_name=eps_data_folder+'/'+env_name+'_bias_ho0.99995_train_episode_data.pkl'
    # Init environment
    env = gym.make(env_name)
    cur_seed = 1000000+int(init_seed)
    np.random.seed(cur_seed)
    env.seed(cur_seed)
    processed_eps_ls,rwd_ls,done_ls,goal_ls=[],[],[],[]
    if env_name=='Reacher-v2':
        raw_eps,acts,rwds,dones,goal,trans=cl_one_eps_reacher(env,cur_seed)
    else:
        raise NotImplementedError
    processed_eps=process_raw_eps(env_name,raw_eps,acts,use_partial_state=True)
    processed_eps_ls.append(processed_eps)
    rwd_ls.append(rwds)
    done_ls.append(dones)
    goal_ls.append(goal)

    with open(data_file_name,'wb') as f:
        pickle.dump([processed_eps_ls,rwd_ls,done_ls,goal_ls],f)
    print(len(processed_eps_ls),len(rwd_ls),len(done_ls))
    print(processed_eps_ls[0].shape,rwd_ls[0].shape,done_ls[0].shape)
    print(done_ls[0])
    print(goal_ls[0])
    print(raw_eps[-1,8:10])
    
    if env_name=='Reacher-v2':
        fig, ax = plt.subplots(figsize=(8,8))
        goal_plan = plt.Circle((goal[0], goal[1]), 0.02, color='m')
        ax.add_artist(goal_plan)
        plt.text(goal[0]-0.01, goal[1]-0.01,str(init_seed),fontsize=20)
        plt.scatter(processed_eps[0,6], processed_eps[0,7], s=15, marker='*', color ='k',label='start')
        plt.plot(raw_eps[:,8]+raw_eps[:,4],raw_eps[:,9]+raw_eps[:,5],'-r',label='rollout path')
        plt.xlim([-0.22, 0.22])
        plt.ylim([-0.22, 0.22])
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.savefig('./mjo_exp/reacher_bias_ho0.99995_train_data.png',dpi=200)
        plt.show()

if __name__ == '__main__':
    if len(sys.argv) > 1:
        main(*sys.argv[1:])
    else:
        main()