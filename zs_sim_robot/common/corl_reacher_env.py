import numpy as np
import torch
import pickle
import gym

def normalize(data,x_std_arr,x_mean_arr):
	return (data - x_mean_arr[:data.shape[-1]]) / x_std_arr[:data.shape[-1]]

def denormalize(data,y_std_arr,y_mean_arr):
	return data * y_std_arr[:data.shape[-1]] + y_mean_arr[:data.shape[-1]]

def gen_init_state(env):
	init=env.reset()
	init=np.concatenate((init[:4],init[6:8],init[8:10]+init[4:6],init[4:6]))
	return init

class observation_space(object):
	def __init__(self,state_dim):
		self.low=-np.array([np.inf for i in range(state_dim)])
		self.high=np.array([np.inf for i in range(state_dim)])
		self.dtype=np.ones(state_dim,dtype=np.float64).dtype
		self.shape=np.ones(state_dim).shape
		self.n=np.inf

class action_space(object):
	def __init__(self):
		self.low=np.array([-1.,-1.])
		self.high=np.array([1.,1.])
		self.dtype=np.ones(2,dtype=np.float64).dtype
		self.shape=np.ones(2).shape
		self.n=np.inf
	def sample(self):
		a=np.zeros(2)
		a[0]=np.random.uniform(-1,1)
		a[1]=np.random.uniform(-1,1)
		return a

class corl_reacher(object):
	def __init__(self,env_seed=0,ho=0,ctrl_rew=True,with_reach_goal_terminal=False):
		self.env_name='corl_Reacher-v2'
		self.suc=False
		torch.manual_seed(env_seed)
		self.gym_env=gym.make('Reacher-v2')
		self.gym_env.seed(env_seed)
		self.ctrl_rew=ctrl_rew
		self.with_reach_goal_terminal=with_reach_goal_terminal
		self.state_dim=10
		self.observation_space=observation_space(self.state_dim)
		self.action_space=action_space()
		self.reward_range=(-np.inf,np.inf)
		self.metadata=None
		self.horizon=50

		if ho==0.999:
			self.reacher_model_path='./trans_model_data/Reacher-v2_model/Reacher-v2_model_lr0.0001_nodes512_seed0_ho0.999_epochs_100'
		elif ho==0:
			self.reacher_model_path='./trans_model_data/Reacher-v2_model/Reacher-v2_model_lr0.0001_nodes512_seed0_epochs_50'
		else:
			self.reacher_model_path='./trans_model_data/Reacher-v2_model/Reacher-v2_model_lr0.0001_nodes512_seed0_ho'+str(ho)+'_epochs_50'
		if ho==0:
			self.norm_path='./trans_model_data/Reacher-v2_normalization/normalization_arr'
		else:
			self.norm_path='./trans_model_data/Reacher-v2_normalization/normalization_arr_ho'+str(ho)
		
		with open(self.reacher_model_path, 'rb') as pickle_file:
			self.reacher_model = torch.load(pickle_file, map_location='cpu')
		with open(self.norm_path, 'rb') as pickle_file:
			x_norm_arr, y_norm_arr = pickle.load(pickle_file)
			self.x_mean_arr, self.x_std_arr = x_norm_arr[0], x_norm_arr[1]
			self.y_mean_arr, self.y_std_arr = y_norm_arr[0], y_norm_arr[1]
		

	def reset(self,big_goal_radius=0.02):
		self.init_state=gen_init_state(self.gym_env)
		self.goal_loc=self.init_state[-2:]
		self.goal_radius=0.6875*big_goal_radius
		self.num_steps=0
		self.cur_state=self.init_state
		self.suc=False
		return self.cur_state

	def compare_reset(self,goal_loc,cur_state):
		self.cur_state=cur_state
		self.goal_loc=goal_loc

	def step(self,ac):
		info={}
		self.num_steps+=1
		ac = np.nan_to_num(ac)
		ac = np.clip(ac, self.action_space.low, self.action_space.high)
		
		sa=np.concatenate([self.cur_state[:8],ac.reshape(-1,)])
		inpt = normalize(sa,self.x_std_arr,self.x_mean_arr)
		inpt = torch.tensor(inpt, dtype=torch.float)
		state_delta = self.reacher_model(inpt)    
		state_delta = state_delta.detach().numpy()
		state_delta = denormalize(state_delta,self.y_std_arr,self.y_mean_arr)
		next_state = sa[:8] + state_delta
		self.cur_state = np.concatenate((next_state,self.cur_state[-2:]))

		if not self.ctrl_rew:
			reward=-np.linalg.norm(self.goal_loc-self.cur_state[6:8])
		else:
			reward=-np.linalg.norm(self.goal_loc-self.cur_state[6:8])-np.square(ac).sum()


		#if (self.cur_state[:4]>1.1).any() or (self.cur_state[:4]<-1.1).any():
		#	reward+=-10000
		#	done=True
		#elif np.linalg.norm(self.cur_state[6:8])>0.22:
		#	reward+=-10000
		#	done=True
		if self.num_steps>=self.horizon:
			done=True
		elif self.with_reach_goal_terminal and np.linalg.norm(self.goal_loc-self.cur_state[6:8])<=self.goal_radius:
			self.suc=True
			done=True
		else:
			done=False

		return self.cur_state,reward,done,info


	def get_info(self):
		return self.num_steps, self.cur_state
