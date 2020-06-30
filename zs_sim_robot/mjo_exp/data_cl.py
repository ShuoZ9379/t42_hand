import gym
import sys, os, pickle
import numpy as np
from .hyperparameters import *
from tqdm import tqdm

def cl_one_eps(env_name,env,cl_max_num_steps):
    if env_name=='Reacher-v2':
        all_obs=env.reset().reshape(1,-1)
        rwds=np.empty(0)
        dones=np.empty(0)
        acts=np.empty((0,2))
        trans = 0
        done=False
        goal=all_obs[0,4:6]
        while not done:
            act = env.action_space.sample()
            obs, rwd, done, _ = env.step(act)
            all_obs=np.concatenate((all_obs,obs.reshape(1,-1)),axis=0)
            rwds=np.concatenate((rwds,np.array([rwd])),axis=0)
            dones=np.concatenate((dones,np.array([done])),axis=0)
            acts=np.concatenate((acts,act.reshape(1,-1)),axis=0)
            trans += 1
            if trans>=cl_max_num_steps:
                return all_obs,acts,rwds,dones,trans
        return all_obs,acts,rwds,dones,goal,trans
    elif env_name=='Acrobot-v1':
        all_obs=env.reset().reshape(1,-1)
        rwds=np.empty(0)
        dones=np.empty(0)
        acts=np.empty((0,1))
        trans = 0
        done=False
        goal=1.
        while not done:
            act = env.action_space.sample()
            obs, rwd, done, _ = env.step(act)
            all_obs=np.concatenate((all_obs,obs.reshape(1,-1)),axis=0)
            rwds=np.concatenate((rwds,np.array([rwd])),axis=0)
            dones=np.concatenate((dones,np.array([done])),axis=0)
            acts=np.concatenate((acts,np.array(act).reshape(1,-1)),axis=0)
            trans += 1
            if trans>=cl_max_num_steps:
                return all_obs,acts,rwds,dones,trans
        return all_obs,acts,rwds,dones,goal,trans
    else:
        raise Exception('Not implemented!')

def process_raw_eps(env_name,raw_eps,acts,use_partial_state):
    if env_name=='Reacher-v2':
        if use_partial_state:
            #raw_eps=np.concatenate((raw_eps[:,6:8],raw_eps[:,8:10]+raw_eps[:,4:6]),axis=1)
            raw_eps=np.concatenate((raw_eps[:,:4],raw_eps[:,6:8],raw_eps[:,8:10]+raw_eps[:,4:6]),axis=1)
        processed_eps=np.concatenate((raw_eps[:-1,:],acts,np.roll(raw_eps,-1,axis=0)[:-1,:]),axis=1)
        return processed_eps
    elif env_name=='Acrobot-v1':
        processed_eps=np.concatenate((raw_eps[:-1,:],acts,np.roll(raw_eps,-1,axis=0)[:-1,:]),axis=1)
        return processed_eps
    else:
        raise Exception('Not implemented!')

def main(env_name, data_file_suffix, init_seed=0):
    eps_data_folder='./mjo_eps_data'
    if not os.path.exists(eps_data_folder):
        os.makedirs(eps_data_folder)
    data_file_name=eps_data_folder+'/'+env_name+'_'+data_file_suffix+'_episode_data.pkl'

    # Init environment
    env = gym.make(env_name)
    processed_eps_ls,rwd_ls,done_ls,goal_ls=[],[],[],[]
    init_seed = int(init_seed)

    trans_sum=0
    i=0
    while trans_sum<max_num_trans:
    #if env_name=='Reacher-v2':
        #for j in tqdm(range(20000)):
        cur_seed=init_seed+i
        np.random.seed(cur_seed)
        env.seed(cur_seed)
        raw_eps,acts,rwds,dones,goal,trans=cl_one_eps(env_name,env,cl_max_num_steps)
        i+=1
        if env_name=='Acrobot-v1' and raw_eps.shape[0]==501:
            continue
        processed_eps=process_raw_eps(env_name,raw_eps,acts,use_partial_state)
        processed_eps_ls.append(processed_eps)
        rwd_ls.append(rwds)
        done_ls.append(dones)
        goal_ls.append(goal)
        trans_sum+=trans
        print(len(processed_eps_ls),trans_sum,i)
        with open(data_file_name,'wb') as f:
            pickle.dump([processed_eps_ls,rwd_ls,done_ls,goal_ls],f)

    print(len(processed_eps_ls),len(rwd_ls),len(done_ls))
    print(processed_eps_ls[0].shape,rwd_ls[0].shape,done_ls[0].shape)
    print(done_ls[0])
    print(goal_ls[0])
    print(trans_sum)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        main(*sys.argv[1:])
    else:
        raise Exception('Missing environment!')


