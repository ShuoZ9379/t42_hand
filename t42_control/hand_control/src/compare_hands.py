#!/usr/bin/env python

import rospy
from std_srvs.srv import Empty, EmptyResponse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Ellipse, Polygon
import pickle
from rollout_t42.srv import rolloutReq
import time
# import var

# np.random.seed(10)

state_dim = 4
tr = '2'
stepSize = 1

rollout_srv = rospy.ServiceProxy('/rollout/rollout', rolloutReq)

path = '/home/pracsys/catkin_ws/src/t42_control/hand_control/src/results/compare/'

rospy.init_node('compare_hands', anonymous=True)


if tr == '1':
    A = np.tile(np.array([1.,-1.]), (600*1./stepSize,1))
if tr == '2':
    A = np.concatenate( (np.array([[1.,  -1.] for _ in range(int(140*1./stepSize))]), 
            np.array([[ -1., 1.] for _ in range(int(120*1./stepSize))]),
            np.array([[1., -1.] for _ in range(int(120*1./stepSize))]) ), axis=0 )
if tr == '3':
    A = np.concatenate( (np.array([[1.,  -1.] for _ in range(int(200*1./stepSize))]), 
            np.array([[1.,  1.] for _ in range(int(110*1./stepSize))]), 
            np.array([[ -1., 1.] for _ in range(int(250*1./stepSize))]),
            np.array([[0., 1.] for _ in range(int(150*1./stepSize))]) ), axis=0 )
if tr == '4':
    A = np.concatenate( (np.array([[-1.,  1.] for _ in range(int(450*1./stepSize))]), 
            np.array([[1.,  1.] for _ in range(int(110*1./stepSize))]), 
            np.array([[ 1., -1.] for _ in range(int(300*1./stepSize))]),
            np.array([[ -1., -1.] for _ in range(int(50*1./stepSize))]),
            np.array([[-1., 1.] for _ in range(int(150*1./stepSize))]) ), axis=0 )

hand = 'blue'

File = path + 'rollout_' + hand + '_a' + tr + '_d' + str(state_dim) + '.pkl'

if 1:
    Af = A.reshape((-1,))
    Pro = []
    Suc = []
    for j in range(10):
        print("Rollout number " + str(j) + ".")
        
        roll = rollout_srv(Af)
        R = np.array(roll.states).reshape(-1,state_dim)
        suc = roll.success

        Pro.append(R)
        Suc.append(suc)

        with open(File, 'w') as f: 
            pickle.dump([Pro, Suc], f)
with open(File, 'r') as f:  
    Pro, Suc = pickle.load(f) 

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


plt.figure(1)

for R in Pro:
    for i in range(4):
        R[:,i] = medfilter(R[:,i], 20)

    ax = plt.subplot(1,2,1)
    plt.plot(R[:,0], R[:,1])

    ax = plt.subplot(1,2,2)
    plt.plot(R[:,2], R[:,3])

plt.show()