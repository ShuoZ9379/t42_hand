#!/usr/bin/env python

import rospy
from cl_control.srv import pathTrackReq
from std_srvs.srv import Empty, EmptyResponse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Ellipse, Polygon
import pickle
import time
from sklearn.neighbors import NearestNeighbors

# np.random.seed(10)

def medfilter(X, W):
    w = int(W/2)

    for k in range(X.shape[1]):
        x = np.copy(X[:,k])
        x_new = np.copy(x)
        for i in range(1, x.shape[0]-1):
            if i < w:
                x_new[i] = np.mean(x[:i+w])
            elif i > x.shape[0]-w:
                x_new[i] = np.mean(x[i-w:])
            else:
                x_new[i] = np.mean(x[i-w:i+w])
        X[:,k] = np.copy(x_new)
    return X

def tracking_error(Sref, S):
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(Sref)

    Sum = 0.
    for s in S:
        _, i = nbrs.kneighbors(s.reshape(1,-1))
        Sum += np.linalg.norm(s[:2]-Sref[i,:2])**2

    return np.sqrt(Sum / S.shape[0])

track_srv = rospy.ServiceProxy('/control', pathTrackReq)

Obj = 'str40'
path = '/home/pracsys/catkin_ws/src/t42_control/cl_control/results/'

test_path = '/home/pracsys/catkin_ws/src/t42_control/hand_control/data/dataset/'
with open(test_path + 'testpaths_' + Obj + '_d_v' + str(1) + '.pkl', 'r') as f: 
    action_seq, test_paths, Obj, _ = pickle.load(f)

# i = 4 # str40
i = 1 # cyl35
# i = 5 # sqr30
# i = 2 # poly10

S = test_paths[i]#[:1030,:]
A = action_seq[i]#[:1030,:]

if 1:
    S = S[:,:4]

    res = track_srv(S.reshape((-1,)))
    Sreal = np.array(res.real_path).reshape(-1, 13)
    Areal = np.array(res.actions).reshape(-1, 2)
    Treal = np.array(res.time)
    success = res.success
    i_path = res.i_path

    with open(path + 'control_track_' + Obj + '_p' + str(i) + '.pkl', 'w') as f: 
        pickle.dump([S, Sreal, A, Areal, Treal, i_path, success], f)
    sref, sreal, A, Areal, Treal, i_path, success = S, Sreal, A, Areal, Treal, i_path, success
else:
    with open(path + 'control_track_' + Obj + '_p' + str(i) + '.pkl', 'r') as f: 
        sref, sreal, A, Areal, Treal, i_path, success = pickle.load(f)

## Plot ##

sref = medfilter(sref[:,:4], 10)
sreal = medfilter(sreal[:,:4], 10)
ds = sreal[0,:2] - sref[0,:2]
sref[:,:2] += ds
sref = sref[:i_path,:]

plt.plot(sref[:,0]-w,sref[:,1], '--r', linewidth = 3, label='reference path')
plt.plot(sreal[:,0]-w,sreal[:,1], '-k', linewidth = 3, label='tracking path')
plt.plot(sref[0,0]-w,sref[0,1], 'ob', linewidth = 3, label='start position')

plt.axis('equal')
plt.xlabel('x (mm)', fontsize=20)
plt.ylabel('y (mm)', fontsize=20)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=20)
plt.savefig(path + 'cltrack_path_' + Obj + '_p' + str(i) + '.png', dpi=150) #str(np.random.randint(100000))

# plt.show()


