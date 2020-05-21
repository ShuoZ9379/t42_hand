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

def gen_init_state(run_idx):
	if run_idx==0:
		init=np.array([0.03238881511894898396,118.18717713766936583397,16.00000000000000000000,16.00000000000000000000])
		init[-2:]=np.array([16,16])
		return init
	elif run_idx==1:
		init=np.array([0.03661012901440022227,118.24889091229067616950,16.00000000000000000000,16.00000000000000000000])
		init[-2:]=np.array([16,16])
		return init
	else:
		np.random.seed(run_idx)
		factor=np.random.randn()
		init=np.random.uniform(0.015,0.045,4)
		init[2]=np.random.uniform(118.06,118.26)
		init[-2:]=np.array([16,16])
		return init

class action_space(object):
	def __init__(self,low,high):
		self.low=low
		self.high=high

class ah_env_noobs(object):
	def __init__(self,goal_loc,big_goal_radius,env_seed,state_with_goal_radius=False):
		torch.manual_seed(env_seed)
		self.init_state=gen_init_state(env_seed)
		self.goal_loc=goal_loc
		self.goal_radius=0.6875*big_goal_radius
		self.action_space=action_space(-1,1)
		self.num_steps=0
		if not state_with_goal_radius:
			self.state_dim=6
			self.cur_state=np.concatenate((self.init_state,self.goal_loc))
		else:
			self.state_dim=7
			self.cur_state=np.concatenate((self.init_state,self.goal_loc,self.goal_radius))

		self.ah_model_path='trans_model_data/sim_cont_trajT_bs512_model512_BS64_loadT.pkl'
		self.norm_path='trans_model_data/normalization_arr_sim_cont_trajT_bs512_model512_BS64_loadT_py2'
		with open(self.ah_model_path, 'rb') as pickle_file:
			self.ah_model = torch.load(pickle_file, map_location='cpu')
		with open(self.norm_path, 'rb') as pickle_file:
			x_norm_arr, y_norm_arr = pickle.load(pickle_file)
			self.x_mean_arr, self.x_std_arr = x_norm_arr[0], x_norm_arr[1]
			self.y_mean_arr, self.y_std_arr = y_norm_arr[0], y_norm_arr[1]

		H1=np.array([[ 88.67572021,  44.43453217],
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
		H2 = np.array([[-87,41],[-83,46],[-76,52],[-60,67],[-46,79],[-32,90],[-18,100],[0,105],[18,100],[32,90],[46,79],[60,67],[76,52],[83,46],[87,41]])
		self.H1D=Delaunay(H1)
		self.H2D=Delaunay(H2)


	def step(self,ac):
		self.num_steps+=1

		sa=np.concatenate([self.cur_state[:4],ac])
		inpt = normalize(sa,self.x_std_arr,self.x_mean_arr)
		inpt = torch.tensor(inpt, dtype=torch.float)
		state_delta = self.ah_model(inpt)    
		state_delta = state_delta.detach().numpy()
		state_delta = denormalize(state_delta,self.y_std_arr,self.y_mean_arr)
		next_state = sa[:4] + state_delta
		self.cur_state = np.concatenate((next_state,self.cur_state[4:]))

		reward=-np.linalg.norm(self.goal_loc-self.cur_state[:2])

		if -reward<=self.goal_radius:
			done=True
		elif (self.cur_state[3:5]>120).any() or (self.cur_state[3:5]<1).any():
			reward=-10000
			done=True
		elif not in_hull(self.cur_state[:2],self.H1D,self.H2D):
			reward=-10000
			done=True
		else:
			done=False

		return self.cur_state,reward,done


	def get_info(self):
		return self.num_steps, self.cur_state

