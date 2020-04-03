import pickle
import numpy as np


def convert(in_name, out_name):
	with open(in_name, 'rb') as f1:
	    trajectory = pickle.load(f1, encoding='latin1')

	with open(out_name, 'wb') as f2:
		pickle.dump(trajectory, f2, 2)


save_path = '/home/pracsys/catkin_ws/src/t42_control/nn_node/models/'
convert(save_path + 'normalization_arr', save_path+ 'normalization_arr_py2')

# trajectory_path = 'data/robotic_hand_real/A/testpaths_cyl35_d_v0.pkl'
# dumppath = 'data/robotic_hand_real/A/testpaths_py2.pkl'
# convert(trajectory_path, dumppath)