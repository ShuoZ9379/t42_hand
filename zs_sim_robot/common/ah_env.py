import numpy as np
import torch
import pickle
from scipy.spatial import Delaunay

def normalize(data,x_std_arr,x_mean_arr):
	return (data - x_mean_arr[:data.shape[-1]]) / x_std_arr[:data.shape[-1]]

def denormalize(data,y_std_arr,y_mean_arr):
	return data * y_std_arr[:data.shape[-1]] + y_mean_arr[:data.shape[-1]]

def in_hull(p,H1D,H2D):
	if H1D.find_simplex(p)>=0 and H2D.find_simplex(p)<0:
		return True
	else:
		return False

def gen_init_state():
	factor=np.random.randn()
	init=np.zeros(4)
	init=np.random.uniform(0.015,0.045,4)
	init[1]=np.random.uniform(118.06,118.26)
	init[-2:]=np.array([16,16])
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

class ah_env_noobs(object):
	def __init__(self,env_seed=0,ah_goal_loc_idx=8,ctrl_rwd=False,ctrl_rwd_coef=1,with_horizon_terminal=True,with_reach_goal_terminal=True,state_with_goal_loc=False,state_with_goal_radius=False,sparse=0,final_rwd=0,horizon=2000):
		self.env_name='ah'
		self.with_obs=0
		self.seed=env_seed
		self.final_rwd=final_rwd
		self.sparse=sparse
		torch.manual_seed(env_seed)
		self.ctrl_rew=ctrl_rwd
		self.ctrl_rwd_coef=ctrl_rwd_coef
		self.with_horizon_terminal=with_horizon_terminal
		self.with_reach_goal_terminal=with_reach_goal_terminal
		self.state_with_goal_loc=state_with_goal_loc
		self.state_with_goal_radius=state_with_goal_radius
		self.goal_loc_idx=ah_goal_loc_idx
		if state_with_goal_loc:
			if state_with_goal_radius:
				self.state_dim=7
			else:
				self.state_dim=6
		else:
			self.state_dim=4
		self.observation_space=observation_space(self.state_dim)
		self.action_space=action_space()
		self.reward_range=(-np.inf,np.inf)
		self.metadata=None
		self.horizon=horizon
		self.goals = np.array([[-66, 80],
		[-41, 100], 
		[-62, 96], 
		[-49, 86], 
		[-55, 92],
		[59, 78],
		[31, 102],
		[60, 100],
		[52, 95],
		[-78, 67],
		[31, 125],
		[-26, 125],
		[0, 107],
		[3, 130],
		[-48, 114],
		[69, 78],
		])
		if not self.state_with_goal_loc:
			self.goal_loc=self.goals[self.goal_loc_idx]

		self.ah_model_path='./trans_model_data/sim_cont_trajT_bs512_model512_BS64_loadT.pkl'
		self.norm_path='./trans_model_data/gazebo_ah_normalization/normalization_arr_sim_cont_trajT_bs512_model512_BS64_loadT_py2'
		with open(self.ah_model_path, 'rb') as pickle_file:
			self.ah_model = torch.load(pickle_file, map_location='cpu')
		with open(self.norm_path, 'rb') as pickle_file:
			x_norm_arr, y_norm_arr = pickle.load(pickle_file)
			self.x_mean_arr, self.x_std_arr = x_norm_arr[0], x_norm_arr[1]
			self.y_mean_arr, self.y_std_arr = y_norm_arr[0], y_norm_arr[1]
		self.H1=np.array([[ 88.67572021,  44.43453217],
			[ 89.80430603,  49.65908432],
       		[ 90.23077393,  52.36616516],
       		[ 90.37576294,  55.98774719],
       		[ 89.2946167 ,  59.5026474 ],
			[ 87.69602966,  64.31713104],
			[ 85.16108704,  71.19532013],
			[ 82.13684845,  77.89694977],
			[ 74.24691772,  91.19889069],
			[ 68.09080505,  98.77561188],
			[ 61.46546173, 106.65620422],
			[ 55.63877487, 112.83303833],
			[ 53.02430725, 114.92677307],
			[ 43.14427567, 122.59031677],
			[ 43.12343216, 122.6015625 ],
			[ 28.351017  , 130.32281494],
			[ 18.74747467, 134.30844116],
			[ 11.96526051, 135.81428528],
			[  8.20428085, 135.91555786],
			[  2.36519504, 135.85865784],
			[-13.29637909, 135.5484314 ],
			[-25.39010048, 134.3369751 ],
			[-39.37775421, 125.64316559],
			[-53.93115997, 112.47859192],
			[-65.12301636, 100.51941681],
			[-73.16171265,  90.56554413],
			[-88.19309998,  71.29073334],
			[-88.44422913,  70.54364777],
			[-89.6594696 ,  56.80038452],
			[-89.75466156,  55.28162766],
			[-89.63751221,  50.12192154],
			[-89.49487305,  48.43606567],
			[-89.34468079,  46.8845253 ],
			[-89.0162735 ,  46.1090126 ],
			[-88.13287354,  44.4129982 ],
			[-87.78145599,  43.91517639]])
		self.H2 = np.array([[-87,41],[-83,46],[-76,52],[-60,67],[-46,79],[-32,90],[-18,100],[0,105],[18,100],[32,90],[46,79],[60,67],[76,52],[83,46],[87,41]])
		self.H1D=Delaunay(self.H1)
		self.H2D=Delaunay(self.H2)

	def reset(self,big_goal_radius=8):
		#torch.manual_seed(self.seed)
		if not self.state_with_goal_loc:
			self.goal_radius=0.6875*big_goal_radius
		else:
			if not self.state_with_goal_radius:
				idx=np.random.randint(self.goals.shape[0])
				self.goal_loc=self.goals[idx,:]
				self.goal_radius=0.6875*big_goal_radius
			else:
				raise NotImplementedError

		self.init_state=gen_init_state()
		self.num_steps=0
		if self.state_with_goal_loc:
			if not self.state_with_goal_radius:
				self.cur_state=np.concatenate((self.init_state,self.goal_loc))
			else:
				self.cur_state=np.concatenate((self.init_state,self.goal_loc,np.array([self.goal_radius])))
		else:
			self.cur_state=self.init_state
		return self.cur_state

	def compare_reset(self,goal_loc,cur_state):
		self.cur_state=cur_state
		self.goal_loc=goal_loc

	def step(self,ac):
		info={}
		self.num_steps+=1
		ac = np.nan_to_num(ac)
		ac = np.clip(ac, self.action_space.low, self.action_space.high)

		sa=np.concatenate([self.cur_state[:4],ac.reshape(-1,)])
		inpt = normalize(sa,self.x_std_arr,self.x_mean_arr)
		inpt = torch.tensor(inpt, dtype=torch.float)
		state_delta = self.ah_model(inpt)    
		state_delta = state_delta.detach().numpy()
		state_delta = denormalize(state_delta,self.y_std_arr,self.y_mean_arr)
		next_state = sa[:4] + state_delta
		self.cur_state = np.concatenate((next_state,self.cur_state[4:]))

		if not self.sparse:
			if not self.ctrl_rew:
				reward=-np.linalg.norm(self.goal_loc-self.cur_state[:2])
			else:
				reward=-np.linalg.norm(self.goal_loc-self.cur_state[:2])-self.ctrl_rwd_coef*np.square(ac).sum()

			#reward=-1.

			#if (self.cur_state[2:4]>120).any() or (self.cur_state[2:4]<1).any():
			#	reward+=-1e6
			#	done=True
			#elif not in_hull(self.cur_state[:2],self.H1D,self.H2D):
			#	reward+=-1e6
			#	done=True
			if self.with_reach_goal_terminal and np.linalg.norm(self.goal_loc-self.cur_state[:2])<=self.goal_radius:
				#reward=0.
				#print('With reach goal terminal: reached within horizon')
				reward+=self.final_rwd
				done=True
			elif self.with_horizon_terminal and self.num_steps>=self.horizon:
				#if np.linalg.norm(self.goal_loc-self.cur_state[:2])<=self.goal_radius:
					#print('No reach goal terminal: reached at horizon')
				#else:
					#print('not reached within horizon')
				done=True
			else:
				done=False
		else:
			if self.with_reach_goal_terminal and np.linalg.norm(self.goal_loc-self.cur_state[:2])<=self.goal_radius:
			#	print('With reach goal terminal: reached within horizon')
				reward=0
				done=True
			elif self.with_horizon_terminal and self.num_steps>=self.horizon:
			#	if np.linalg.norm(self.goal_loc-self.cur_state[:2])<=self.goal_radius:
			#		print('No reach goal terminal: reached at horizon')
			#	else:
			#		print('not reached within horizon')
				reward=-1
				done=True
			else:
				reward=-1
				done=False

		return self.cur_state,reward,done,info


	def get_info(self):
		return self.num_steps, self.cur_state

class ah_env_withobs(object):
	def __init__(self,obs_idx=20,env_seed=0,ah_goal_loc_idx=8,ctrl_rwd=False,ctrl_rwd_coef=1,with_horizon_terminal=True,with_reach_goal_terminal=True,state_with_goal_loc=False,state_with_goal_radius=False,with_obs_end=1,sparse=0,obs_pen=1e6,final_rwd=0,horizon=2000):
		self.env_name='ah'
		self.with_obs=1
		self.with_obs_end=with_obs_end
		self.obs_pen=obs_pen
		self.final_rwd=final_rwd
		self.sparse=sparse
		self.seed=env_seed
		torch.manual_seed(env_seed)
		self.ctrl_rew=ctrl_rwd
		self.ctrl_rwd_coef=ctrl_rwd_coef
		self.with_horizon_terminal=with_horizon_terminal
		self.with_reach_goal_terminal=with_reach_goal_terminal
		self.state_with_goal_loc=state_with_goal_loc
		self.state_with_goal_radius=state_with_goal_radius
		self.goal_loc_idx=ah_goal_loc_idx
		if state_with_goal_loc:
			if state_with_goal_radius:
				self.state_dim=7
			else:
				self.state_dim=6
		else:
			self.state_dim=4
		self.observation_space=observation_space(self.state_dim)
		self.action_space=action_space()
		self.reward_range=(-np.inf,np.inf)
		self.metadata=None
		self.horizon=horizon
		self.goals = np.array([[-66, 80],
		[-41, 100], 
		[-62, 96], 
		[-49, 86], 
		[-55, 92],
		[59, 78],
		[31, 102],
		[60, 100],
		[52, 95],
		[-78, 67],
		[31, 125],
		[-26, 125],
		[0, 107],
		[3, 130],
		[-48, 114],
		[69, 78],
		])
		self.obs_idx=obs_idx
		self.obs_filename='./obs_'+str(obs_idx)+'.pkl'
		with open(self.obs_filename, 'rb') as f: 
			self.Obs = pickle.load(f,encoding='latin')[:,:2]
		if self.obs_idx==14:
			obs_dist=0.5
		else:
			obs_dist=0.75
		self.obs_dist=obs_dist
		if not self.state_with_goal_loc:
			self.goal_loc=self.goals[self.goal_loc_idx]

		self.ah_model_path='./trans_model_data/sim_cont_trajT_bs512_model512_BS64_loadT.pkl'
		self.norm_path='./trans_model_data/gazebo_ah_normalization/normalization_arr_sim_cont_trajT_bs512_model512_BS64_loadT_py2'
		with open(self.ah_model_path, 'rb') as pickle_file:
			self.ah_model = torch.load(pickle_file, map_location='cpu')
		with open(self.norm_path, 'rb') as pickle_file:
			x_norm_arr, y_norm_arr = pickle.load(pickle_file)
			self.x_mean_arr, self.x_std_arr = x_norm_arr[0], x_norm_arr[1]
			self.y_mean_arr, self.y_std_arr = y_norm_arr[0], y_norm_arr[1]
		self.H1=np.array([[ 88.67572021,  44.43453217],
			[ 89.80430603,  49.65908432],
       		[ 90.23077393,  52.36616516],
       		[ 90.37576294,  55.98774719],
       		[ 89.2946167 ,  59.5026474 ],
			[ 87.69602966,  64.31713104],
			[ 85.16108704,  71.19532013],
			[ 82.13684845,  77.89694977],
			[ 74.24691772,  91.19889069],
			[ 68.09080505,  98.77561188],
			[ 61.46546173, 106.65620422],
			[ 55.63877487, 112.83303833],
			[ 53.02430725, 114.92677307],
			[ 43.14427567, 122.59031677],
			[ 43.12343216, 122.6015625 ],
			[ 28.351017  , 130.32281494],
			[ 18.74747467, 134.30844116],
			[ 11.96526051, 135.81428528],
			[  8.20428085, 135.91555786],
			[  2.36519504, 135.85865784],
			[-13.29637909, 135.5484314 ],
			[-25.39010048, 134.3369751 ],
			[-39.37775421, 125.64316559],
			[-53.93115997, 112.47859192],
			[-65.12301636, 100.51941681],
			[-73.16171265,  90.56554413],
			[-88.19309998,  71.29073334],
			[-88.44422913,  70.54364777],
			[-89.6594696 ,  56.80038452],
			[-89.75466156,  55.28162766],
			[-89.63751221,  50.12192154],
			[-89.49487305,  48.43606567],
			[-89.34468079,  46.8845253 ],
			[-89.0162735 ,  46.1090126 ],
			[-88.13287354,  44.4129982 ],
			[-87.78145599,  43.91517639]])
		self.H2 = np.array([[-87,41],[-83,46],[-76,52],[-60,67],[-46,79],[-32,90],[-18,100],[0,105],[18,100],[32,90],[46,79],[60,67],[76,52],[83,46],[87,41]])
		self.H1D=Delaunay(self.H1)
		self.H2D=Delaunay(self.H2)

	def reset(self,big_goal_radius=8.):
		#torch.manual_seed(self.seed)
		if self.obs_idx==14:
			self.goal_loc = np.array([21., 123.])
			big_goal_radius = 5.
		if not self.state_with_goal_loc:
			self.goal_radius=0.6875*big_goal_radius
		else:
			if not self.state_with_goal_radius:
				if self.obs_idx!=14:
					idx=np.random.randint(self.goals.shape[0])
					self.goal_loc=self.goals[idx,:]
				self.goal_radius=0.6875*big_goal_radius
			else:
				raise NotImplementedError

		self.init_state=gen_init_state()
		self.num_steps=0
		if self.state_with_goal_loc:
			if not self.state_with_goal_radius:
				self.cur_state=np.concatenate((self.init_state,self.goal_loc))
			else:
				self.cur_state=np.concatenate((self.init_state,self.goal_loc,np.array([self.goal_radius])))
		else:
			self.cur_state=self.init_state
		return self.cur_state

	def compare_reset(self,goal_loc,cur_state):
		self.cur_state=cur_state
		self.goal_loc=goal_loc

	def step(self,ac):
		info={}
		self.num_steps+=1
		ac = np.nan_to_num(ac)
		ac = np.clip(ac, self.action_space.low, self.action_space.high)

		sa=np.concatenate([self.cur_state[:4],ac.reshape(-1,)])
		inpt = normalize(sa,self.x_std_arr,self.x_mean_arr)
		inpt = torch.tensor(inpt, dtype=torch.float)
		state_delta = self.ah_model(inpt)    
		state_delta = state_delta.detach().numpy()
		state_delta = denormalize(state_delta,self.y_std_arr,self.y_mean_arr)
		next_state = sa[:4] + state_delta
		self.cur_state = np.concatenate((next_state,self.cur_state[4:]))

		if not self.sparse:
			if not self.ctrl_rew:
				reward=-np.linalg.norm(self.goal_loc-self.cur_state[:2])
			else:
				reward=-np.linalg.norm(self.goal_loc-self.cur_state[:2])-self.ctrl_rwd_coef*np.square(ac).sum()


			#if (self.cur_state[2:4]>120).any() or (self.cur_state[2:4]<1).any():
			#	print('invalid load')
			#	reward+=-1e6
			#	done=True
			#elif not in_hull(self.cur_state[:2],self.H1D,self.H2D):
			#	print('invalid state')
			#	reward+=-1e6
			#	done=True
			if not (np.linalg.norm(self.Obs-self.cur_state[:2],axis=1)>self.obs_dist*1.2).all():
			#	print('obstacle collision')
				reward+=-self.obs_pen
				if self.with_obs_end:
					print('aaa')
					done=True
				else:
					done=False
			elif self.with_reach_goal_terminal and np.linalg.norm(self.goal_loc-self.cur_state[:2])<=self.goal_radius:
			#	print('With reach goal terminal: reached within horizon')
				if self.obs_idx==14:
					print('FOUND!')
					raise
				print('OK!')
				reward+=self.final_rwd
				done=True
			elif self.with_horizon_terminal and self.num_steps>=self.horizon:
			#	if np.linalg.norm(self.goal_loc-self.cur_state[:2])<=self.goal_radius:
			#		print('No reach goal terminal: reached at horizon')
			#	else:
			#		print('not reached within horizon')
				print('bbb')
				done=True
			else:
				done=False

		else:
			if not (np.linalg.norm(self.Obs-self.cur_state[:2],axis=1)>self.obs_dist*1.2).all():
			#	print('obstacle collision')
				reward=-1-self.horizon
				if self.with_obs_end:
					print('aaa')
					done=True
				else:
					done=False
			elif self.with_reach_goal_terminal and np.linalg.norm(self.goal_loc-self.cur_state[:2])<=self.goal_radius:
			#	print('With reach goal terminal: reached within horizon')
				reward=0
				done=True
			elif self.with_horizon_terminal and self.num_steps>=self.horizon:
			#	if np.linalg.norm(self.goal_loc-self.cur_state[:2])<=self.goal_radius:
			#		print('No reach goal terminal: reached at horizon')
			#	else:
			#		print('not reached within horizon')
				reward=-1
				done=True
			else:
				reward=-1
				done=False

		return self.cur_state,reward,done,info


	def get_info(self):
		return self.num_steps, self.cur_state


