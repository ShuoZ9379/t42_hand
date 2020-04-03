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

Obj = 'cyl30'
path = '/home/pracsys/catkin_ws/src/t42_control/cl_control/results/'

test_path = '/home/pracsys/catkin_ws/src/t42_control/hand_control/data/dataset/'
with open(test_path + 'testpaths_' + Obj + '_d_v' + str(1) + '.pkl', 'r') as f: 
    action_seq, test_paths, Obj, _ = pickle.load(f)

if 0:

    j = 0
    Pro = []
    Aro = []
    Suc = []
    I = []
    for S in test_paths:
        print("Rolling out path number %d..."%j)
        S = S[:,:4]

        P = []
        for k in range(10):
            print('Trial %d.'%k)
            res = track_srv(S.reshape((-1,)))
            Sreal = np.array(res.real_path).reshape(-1, 13)
            Areal = np.array(res.actions).reshape(-1, 2)
            success = res.success
            i_path = res.i_path
            P += [(j, Sreal, Areal, success, i_path, S)]
            print('Reached point %d out of %d.'%(i_path, S.shape[0]))
            
        Pro.append(P)

        with open(path + 'control_path_' + Obj + '_d_v' + str(1) + '.pkl', 'w') as f: 
            pickle.dump(Pro, f)
        j += 1
else:
    with open(path + 'control_path_' + Obj + '_d_v' + str(1) + '.pkl', 'r') as f: 
        Pro = pickle.load(f)


## Plot ##

plt.figure(1)

Err = []
i = 0
for P in Pro:
    # j = np.array([item[0] for item in P])
    Sreal = [item[1] for item in P]
    Sref = [item[5] for item in P]
    I = [item[4] for item in P]

    print "----", i
    i += 1

    for sreal, sref, i_path in zip(Sreal, Sref, I):
        if sreal.shape[0] < 20:
            continue

        print sref.shape, i_path

        sreal = sreal[:-20,:]

        sreal = medfilter(sreal[:,:4], 10)
        ds = sreal[0,:2] - sref[0,:2]
        sref[:,:2] += ds
        sref = sref[:i_path,:]

        Err.append(tracking_error(sref, sreal))

        plt.plot(sref[:,0],sref[:,1],'--k')
        # plt.plot(sref[i_path,0],sref[i_path,1],'or')
        # plt.plot(sref[:,0],sref[:,1],':m')
        plt.plot(sreal[:,0],sreal[:,1],'-r')
plt.show()

print np.mean(np.array(Err))
exit(1)

fig, ax = plt.subplots(figsize=(12,6))

# from matplotlib.patches import Ellipse, Polygon
w = 23.4
H1 = np.array([[109.43661928,  45.45193231],
    [109.50734019,  45.48163721],
    [109.55533981,  45.51312672],
    [109.66525674,  45.61027214],
    [109.76466537,  45.71999609],
    [109.7899437 ,  45.7532702 ],
    [109.818762  ,  45.8920449 ],
    [103.75124216,  63.26119602],
    [101.20272636,  68.47952306],
    [ 91.55581395,  85.36463135],
    [ 82.23831654,  95.83953023],
    [ 56.97445869, 114.53938466],
    [ 55.3663969 , 115.56509882],
    [ 40.51083326, 119.81970072],
    [ 37.26506233, 119.99034882],
    [  9.94575024, 119.83285844],
    [ -9.41121578, 113.63378167],
    [-26.0669589 , 102.4928391 ],
    [-39.59908932,  88.99509944],
    [-39.83750641,  88.75419796],
    [-39.90573734,  88.68383244],
    [-39.97552544,  88.6100959 ],
    [-40.42713046,  88.1207196 ],
    [-41.84839576,  86.53383218],
    [-43.51329803,  84.62289628],
    [-51.62924528,  75.13655722],
    [-56.84250593,  64.16174769],
    [-62.06816435,  39.81572762],
    [-61.94485589,  39.46453519],
    [-61.91357613,  39.40508559],
    [-61.40811145,  38.84640774],
    [-61.20847613,  38.67814187],
    [-60.8488217 ,  38.51114213],
    [-60.75637937,  38.47118448],
    [-60.48326939,  38.43883537],
    [-60.46220064,  38.43718059],
    [-60.29364914,  38.4239424 ],
    [-60.07242203,  38.40656728],
    [-59.83011425,  38.38753738]])
# H2 = np.array([[-44,55],[-26,68],[-12,76],[23,86],[47,78],[65,69],[81,62],[97,50]])
# H = np.concatenate((np.array(H1), H2), axis=0)
# # H[:,0] -= w
# pgon = plt.Polygon(H, color='y', alpha=1, zorder=0)
# ax.add_patch(pgon)

# i = 4; j = 0 # str40
i = 0; j = 0 # cyl35
# i = 5; j = 2 # sqr30
# i = 2; j = 1 # poly10

P = Pro[i]
Sreal = [item[1] for item in P]
Sref = [item[5] for item in P]
I = [item[4] for item in P]

sreal = Sreal[j]
sref = Sref[j]
i_path = I[j]

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
# plt.savefig(path + 'cltrack_' + Obj + '.png', dpi=150) #str(np.random.randint(100000))

plt.show()


