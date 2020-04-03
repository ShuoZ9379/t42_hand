#!/usr/bin/env python

import rospy
from std_srvs.srv import Empty, EmptyResponse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Ellipse, Polygon
import pickle
from nn_predict.srv import StateAction2State
import time

# np.random.seed(10)

state_dim = 4
stepSize = 1
version = 0
Obj = 'cyl45'

nn_srv = rospy.ServiceProxy('/nn/predict', StateAction2State)
rospy.init_node('NN_t42', anonymous=True)

path = '/home/pracsys/catkin_ws/src/t42_control/nn_predict/src/results/'
test_path = '/home/pracsys/catkin_ws/src/t42_control/hand_control/data/dataset/'

def medfilter(x, W):
    w = int(W/2)
    x_new = np.copy(x)
    for i in range(0, x.shape[0]):
        if i < w:
            x_new[i] = np.mean(x[:i+w])
        elif i > x.shape[0]-w:
            x_new[i] = np.mean(x[i-w:])
        else:
            x_new[i] = np.mean(x[i-w:i+w])
    return x_new

if 1:
    with open(test_path + 'testpaths_' + Obj + '_d_v' + str(version) + '.pkl', 'r') as f: 
        action_seq, test_paths, Obj, Suc = pickle.load(f)
    
    action_seq = action_seq[:1]
    test_paths = test_paths[:1]
    # Suc = Suc[:1]
    # action_seq[0] = action_seq[0][20:,:]
    # test_paths[0] = test_paths[0][20:,:]

    GP_batch = []
    NN_naive = []
    filtered_test_paths = []
    for A, R in zip(action_seq, test_paths):
        if np.any(Obj == np.array(['sqr30','poly10','poly6','elp40'])): # Include orientation angle
            R = R[:,[0,1,11,12,2]]
        else:
            R = R[:,[0,1,11,12]]
        state_dim = R.shape[1]
        # R = R[20:,:]
        # A = A[20:,:]

        print('Smoothing data...')
        for i in range(state_dim):
            R[:,i] = medfilter(R[:,i], 20)

        filtered_test_paths.append(R)

        s_start = R[0,:]
        sigma_start = np.ones((1,state_dim))*1e-3
       

        ######################################## naive propagation ###############################################

        print "Running Naive."
        Np = 1 # Number of particles
        t_naive = 0

        s = np.copy(s_start)# + np.random.normal(0, sigma_start)
        # s = np.tile(s, (Np,1)) + np.random.normal(0, sigma_start, (Np, state_dim))
        Ypred_naive = s.reshape(1,state_dim)

        print("Running (open loop) path...")
        p_naive = 1
        for i in range(0, A.shape[0]):
            print("[Naive] Step " + str(i) + " of " + str(A.shape[0]))
            a = A[i,:]
            print s, a

            st = time.time()
            res = nn_srv(s.reshape(-1,1), a)
            t_naive += (time.time() - st) 

            s_next = np.array(res.next_state)
            s = np.copy(s_next)

            Ypred_naive = np.append(Ypred_naive, s_next.reshape(1,state_dim), axis=0)

        NN_naive.append(Ypred_naive)

        t_naive /= A.shape[0]

    ######################################## Save ###########################################################

    with open(path + 'testpaths_' + Obj + '_d_v' + str(version) + '.pkl', 'w') as f: 
        pickle.dump([action_seq, filtered_test_paths, NN_naive], f)

    ######################################## Plot ###########################################################
else:
    with open(path + 'testpaths_' + Obj + '_d_v' + str(version) + '.pkl', 'r') as f: 
        action_seq, filtered_test_paths, NN_naive = pickle.load(f)  

ix = [0, 1]
# plt.figure(1)
# ax1 = plt.subplot(2,1,1)
# ax1.plot(t[:-1], Smean[:,ix[0]], '-b', label='rollout mean')
# ax1.fill_between(t[:-1], Smean[:,ix[0]]+Sstd[:,ix[0]], Smean[:,ix[0]]-Sstd[:,ix[0]], facecolor='blue', alpha=0.5, label='rollout std.')
# ax1.plot(t, Ypred_mean_gp[:,ix[0]], '-r', label='BPP mean')
# ax1.fill_between(t, Ypred_mean_gp[:,ix[0]]+Ypred_std_gp[:,ix[0]], Ypred_mean_gp[:,ix[0]]-Ypred_std_gp[:,ix[0]], facecolor='red', alpha=0.5, label='BGP std.')
# ax1.plot(t, Ypred_mean_gpup[:,0], '--c', label='GPUP mean')
# ax1.fill_between(t, Ypred_mean_gpup[:,0]+Ypred_std_gpup[:,0], Ypred_mean_gpup[:,0]-Ypred_std_gpup[:,0], facecolor='cyan', alpha=0.5, label='GPUP std.')
# ax1.plot(t, Ypred_naive[:,0], '-k', label='Naive')
# ax1.plot(t, Ypred_bmean[:,0], '-m', label='Batch mean')
# ax1.legend()
# plt.title('Path ' + tr)
# ax2 = plt.subplot(2,1,2)
# ax2.plot(t[:-1], Smean[:,ix[1]], '-b')
# ax2.fill_between(t[:-1], Smean[:,ix[1]]+Sstd[:,ix[1]], Smean[:,ix[1]]-Sstd[:,ix[1]], facecolor='blue', alpha=0.5)
# ax2.plot(t, Ypred_mean_gp[:,ix[1]], '-r')
# ax2.fill_between(t, Ypred_mean_gp[:,ix[1]]+Ypred_std_gp[:,ix[1]], Ypred_mean_gp[:,ix[1]]-Ypred_std_gp[:,ix[1]], facecolor='red', alpha=0.5)
# ax2.plot(t, Ypred_mean_gpup[:,1], '--c')
# ax2.fill_between(t, Ypred_mean_gpup[:,1]+Ypred_std_gpup[:,1], Ypred_mean_gpup[:,1]-Ypred_std_gpup[:,1], facecolor='cyan', alpha=0.5)
# ax2.plot(t, Ypred_naive[:,1], '-k')
# ax2.plot(t, Ypred_bmean[:,1], '-m')

ix = [0, 1]
for S, Snaive in zip(filtered_test_paths, NN_naive):

    plt.plot(S[:,ix[0]], S[:,ix[1]], '.-k', label='rollout')
    plt.plot(Snaive[:,ix[0]], Snaive[:,ix[1]], '.-c', label='Naive')
    plt.axis('equal')
    plt.legend()
    # plt.show()



# plt.savefig('/home/pracsys/catkin_ws/src/t42_control/gpup_gp_node/src/results/path_' + tr + '.png', dpi=300) #str(np.random.randint(100000))
plt.show()

