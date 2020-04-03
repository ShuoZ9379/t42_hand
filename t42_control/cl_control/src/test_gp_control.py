#!/usr/bin/env python

import rospy
from control.srv import pathTrackReq
from std_srvs.srv import Empty, EmptyResponse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Ellipse, Polygon
import pickle
import time

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


rp = 7.
r = 10.


track_srv = rospy.ServiceProxy('/control', pathTrackReq)

File = 'naive_goal-35.0_92.0_n17482_traj'

traj = '/home/pracsys/catkin_ws/src/t42_control/hand_control/plans/' + File + '.txt' 
path = '/home/pracsys/catkin_ws/src/t42_control/cl_control/results/'

j = File.find('goal')+4
j1 = j
while File[j] != '_':
    j += 1
ctr = np.array([float(File[j1:j]), 0, 0, 0])
j1 = j + 1
j = j1
while File[j] != '_':
    j += 1
ctr[1] = float(File[j1:j])

S = np.loadtxt(traj, delimiter=',')
# S = medfilter(S, 20)
S = np.append(S, [ctr], axis=0)

if 1:
    res = track_srv(S.reshape((-1,)))
    Sreal = np.array(res.real_path).reshape(-1, S.shape[1])
    Areal = np.array(res.actions).reshape(-1, 2)
    success = res.success

    with open(path + 'control_' + File + '.pkl', 'w') as f: 
        pickle.dump([Sreal, Areal, S, success], f)
else:
    with open(path + 'control_' + File + '.pkl', 'r') as f: 
        Sreal, Areal, S, success = pickle.load(f)


## Plot ##

plt.figure(1)
ax = plt.subplot()
goal = plt.Circle((ctr[0], ctr[1]), r, color='m')
ax.add_artist(goal)
goal_plan = plt.Circle((ctr[0], ctr[1]), rp, color='w')
ax.add_artist(goal_plan)

plt.plot(S[:-1,0],S[:-1,1],'-r', label='reference')
plt.plot(Sreal[:,0],Sreal[:,1],'-k', label='Rolled-out')
plt.title(File + ", success: " + str(success))
plt.legend()
plt.axis('equal')
# plt.savefig(path + File, dpi=300)

# plt.figure(2)
# plt.plot(S[:-1,2],S[:-1,3],'-r', label='reference')
# plt.plot(Sreal[:,2],Sreal[:,3],'-k', label='Rolled-out')

plt.show()


