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
	return init

class observation_space(object):
	def __init__(self):
		self.low=-np.array([-1.,-1.,-1.,-1.,-12.566371,-28.274334],dtype=np.float64)
		self.high=np.array([1.,1.,1.,1.,12.566371,28.274334],dtype=np.float64)
		self.dtype=np.ones(6,dtype=np.float64).dtype
		self.shape=np.ones(6).shape
		self.n=np.inf

class action_space(object):
	def __init__(self):
		self.low=None
		self.high=None
		self.dtype=np.ones(10000000,dtype=np.int64).dtype
		self.shape=np.array(10000000).shape
		self.n=3
	def sample(self):
		a=np.random.randint(3)
		return a

class corl_acrobot(object):
	def __init__(self,env_seed=0,goal_height=1.):
		self.env_name='corl_Acrobot-v1'
		torch.manual_seed(env_seed)
		self.gym_env=gym.make('Acrobot-v1')
		self.gym_env.seed(env_seed)
		self.state_dim=6
		self.observation_space=observation_space()
		self.action_space=action_space()
		self.reward_range=(-np.inf,np.inf)
		self.metadata=None
		self.horizon=500
		self.goal_height=goal_height

		self.acrobot_model_path='./trans_model_data/Acrobot-v1_model_lr0.0001_nodes512_seed0_epochs_50'
		self.norm_path='./trans_model_data/Acrobot-v1_normalization/normalization_arr'
		with open(self.acrobot_model_path, 'rb') as pickle_file:
			self.acrobot_model = torch.load(pickle_file, map_location='cpu')
		with open(self.norm_path, 'rb') as pickle_file:
			x_norm_arr, y_norm_arr = pickle.load(pickle_file)
			self.x_mean_arr, self.x_std_arr = x_norm_arr[0], x_norm_arr[1]
			self.y_mean_arr, self.y_std_arr = y_norm_arr[0], y_norm_arr[1]
		

	def reset(self):
		self.init_state=gen_init_state(self.gym_env)
		self.num_steps=0
		self.cur_state=self.init_state
		return self.cur_state

	def compare_reset(self,cur_state):
		self.cur_state=cur_state

	def step(self,ac):
		info={}
		self.num_steps+=1
		
		sa=np.concatenate([self.cur_state,np.array([ac])])
		inpt = normalize(sa,self.x_std_arr,self.x_mean_arr)
		inpt = torch.tensor(inpt, dtype=torch.float)
		state_delta = self.acrobot_model(inpt)    
		state_delta = state_delta.detach().numpy()
		state_delta = denormalize(state_delta,self.y_std_arr,self.y_mean_arr)
		next_state = sa[:6] + state_delta
		self.cur_state = next_state

		reward=-1.

		#if (self.cur_state[:4]>1.1).any() or (self.cur_state[:4]<-1.1).any():
		#	reward=-10000
		#	done=True
		#elif np.abs(-self.cur_state[0]-(self.cur_state[0]*self.cur_state[2]-self.cur_state[1]*self.cur_state[3]))>2.2:
		#	reward=-10000
		#	done=True
		if -self.cur_state[0]-(self.cur_state[0]*self.cur_state[2]-self.cur_state[1]*self.cur_state[3])>=self.goal_height:
			reward=0.
			done=True
		elif self.num_steps>=self.horizon:
			done=True
		else:
			done=False

		return self.cur_state,reward,done,info


	def get_info(self):
		return self.num_steps, self.cur_state
