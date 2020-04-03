""" 
Author: Avishai Sintov
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from common.data_normalization import *
import pickle
import random

class predict_nn:
    def __init__(self):

        base_path = ''
        save_path = base_path + 'save_model/robotic_hand_real/pytorch/'
        model_name = 'real_A_1.pkl' # Name of the model we want to depickle
        model_name = 'real_A_heldout0.1_1.pkl'
        self.model_path = save_path + model_name

        print('[predict_nn] Loading training data...')
        with open(save_path+'/normalization_arr/normalization_arr_py2', 'rb') as pickle_file:
            x_norm_arr, y_norm_arr = pickle.load(pickle_file)

        self.x_mean_arr, self.x_std_arr = x_norm_arr[0], x_norm_arr[1]
        self.y_mean_arr, self.y_std_arr = y_norm_arr[0], y_norm_arr[1]

        with open(self.model_path, 'rb') as pickle_file:
            self.model = torch.load(pickle_file, map_location='cpu')


    def normalize(self, data):
        return (data - self.x_mean_arr[:data.shape[-1]]) / self.x_std_arr[:data.shape[-1]]

    def denormalize(self, data, ):
        return data * self.y_std_arr[:data.shape[-1]] + self.y_mean_arr[:data.shape[-1]]


    def predict(self, sa):

        inpt = self.normalize(sa)
        inpt = torch.tensor(inpt, dtype=torch.float)
        state_delta = self.model(inpt)
        state_delta = state_delta.detach().numpy()#*2.5
        state_delta = self.denormalize(state_delta)

        next_state = (sa[...,:4] + state_delta)
        return next_state





if __name__ == "__main__":
    task = 'real_A' 
    held_out = .1
    test_traj = 0
    # _ , arg1, arg2, arg3 = argv
    nn_type = '1'
    method = ''
    save_path = 'save_model/robotic_hand_real/pytorch/'

    trajectory_path_map = {
        'real_A': 'data/robotic_hand_real/A/testpaths_cyl35_d_v0.pkl', 
        'real_B': 'data/robotic_hand_real/B/testpaths_cyl35_red_d_v0.pkl',
        'transferA2B': 'data/robotic_hand_real/B/testpaths_cyl35_red_d_v0.pkl',
        'transferB2A': 'data/robotic_hand_real/A/testpaths_cyl35_d_v0.pkl',
        }
    # trajectory_path = trajectory_path_map[task]
    trajectory_path = 'data/robotic_hand_real/A/testpaths_py2.pkl'

    with open(trajectory_path, 'rb') as pickle_file:
        trajectory = pickle.load(pickle_file)#, encoding='latin1')

        # u = pickle._Unpickler(pickle_file)
        # u.encoding = 'latin1'
        # trajectory = u.load()

    def make_traj(trajectory, test_traj):
        real_positions = trajectory[0][test_traj]
        acts = trajectory[1][test_traj]    
        return np.append(real_positions, acts, axis=1)

    NN = predict_nn()
    state_dim = 4
    action_dim = 6




    # BATCH = True
    BATCH = False
    for test_traj in range(4):

        states = []
        if BATCH:
            batch_size = 4
            out=[make_traj(trajectory, i) for i in range(batch_size)]

            lengths = [len(traj) for traj in out]
            min_length = min(lengths)
            batches = [traj[0:min_length] for traj in out]
            traj = np.stack(batches,0)

            true_states = traj[:,:,:state_dim]
            state = traj[:,0,:state_dim]

            actions = traj[..., state_dim:state_dim+action_dim]


            for i in range(traj.shape[1]):
                states.append(state)
                action = actions[:,i]
                sa = np.concatenate((state, action), -1)
                state = NN.predict(sa)
            states = np.stack(states, 1)


        else:
            traj = make_traj(trajectory, test_traj)
            true_states = traj[:,:state_dim]
            state = traj[0][:state_dim]
        

            for i, point in enumerate(traj):
                states.append(state)
                action = point[state_dim:state_dim+action_dim]
                # if cuda: action = action.cuda() 
                # pdb.set_trace()
                sa = np.concatenate((state, action), 0)
                state = NN.predict(sa)
            states = np.stack(states, 0)



        if BATCH:
            plt.figure(1)
            plt.scatter(traj[test_traj, 0, 0], traj[test_traj, 0, 1], marker="*", label='start')
            plt.plot(traj[test_traj, :, 0], traj[test_traj, :, 1], color='blue', label='Ground Truth', marker='.')
            plt.plot(states[test_traj, :, 0], states[test_traj, :, 1], color='red', label='NN Prediction')
            plt.axis('scaled')
            plt.title('Bayesian NN Prediction -- pos Space')
            plt.legend()
            plt.show()


        else:
            plt.figure(1)
            plt.scatter(traj[0, 0], traj[0, 1], marker="*", label='start')
            plt.plot(traj[:, 0], traj[:, 1], color='blue', label='Ground Truth', marker='.')
            plt.plot(states[:, 0], states[:, 1], color='red', label='NN Prediction')
            plt.axis('scaled')
            plt.title('Bayesian NN Prediction -- pos Space')
            plt.legend()
            plt.show()

