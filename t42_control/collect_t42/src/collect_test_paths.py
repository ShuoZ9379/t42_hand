#!/usr/bin/env python

import rospy
from std_srvs.srv import Empty, EmptyResponse
import numpy as np
import matplotlib.pyplot as plt
import pickle
from rollout_t42.srv import rolloutReq
import time
import random

import sys
sys.path.insert(0, '/home/pracsys/catkin_ws/src/t42_control/gpup_gp_node/src/')
import var

state_dim = 13
stepSize = var.stepSize_
version = 1#var.data_version_

#v0
# action_seq = []
# # 0
# A = np.tile(np.array([-1.,1.]), (800*1/stepSize,1))
# action_seq.append(A)
# # 1
# A = np.concatenate(  (np.array([[1.,  -1.] for _ in range(int(700*1./stepSize))]), 
#         np.array([[ -1., 1.] for _ in range(int(200*1./stepSize))]),
#         np.array([[ 1., 1.] for _ in range(int(150*1./stepSize))]),
#         np.array([[ 0, -1.5] for _ in range(int(200*1./stepSize))]),
#         ), axis=0 )
# action_seq.append(A)
# # 2
# A = np.concatenate( (np.array([[-1.,  1.] for _ in range(int(400*1./stepSize))]), 
#         np.array([[ 1., -1.] for _ in range(int(400*1./stepSize))]),
#         np.array([[-1., 1.] for _ in range(int(400*1./stepSize))]) ), axis=0 )
# action_seq.append(A)
# # 3
# A = np.concatenate( (np.array([[1.,  -1.] for _ in range(int(200*1./stepSize))]), 
#         np.array([[1.,  1.] for _ in range(int(200*1./stepSize))]), 
#         np.array([[ -1., 1.] for _ in range(int(500*1./stepSize))])), axis=0 )
# action_seq.append(A)
# # 4
# A = np.concatenate( (np.array([[-1.,  1.] for _ in range(int(200*1./stepSize))]), 
#         np.array([[1.,  1.] for _ in range(int(170*1./stepSize))]), 
#         np.array([[ 1., -1.] for _ in range(int(250*1./stepSize))]),
#         np.array([[ -1., -1.] for _ in range(int(250*1./stepSize))])), axis=0 )
# action_seq.append(A)
# # 5
# A = np.concatenate(  (np.array([[-1.,  1.] for _ in range(int(600*1./stepSize))]), 
#         np.array([[1.5,  0.] for _ in range(int(300*1./stepSize))]), 
#         np.array([[-1.,  -1.] for _ in range(int(120*1./stepSize))]), 
#         np.array([[0.0,  1.5] for _ in range(int(300*1./stepSize))]), 
#         np.array([[ 0., -1.5] for _ in range(int(200*1./stepSize))])), axis=0 )
# action_seq.append(A)

discrete = False
test_path = '/home/pracsys/catkin_ws/src/t42_control/hand_control/data/dataset/'

#v1
if discrete:
    org_paths = 'testpaths_' + 'cyl35' + '_d_v' + str(1) + '.pkl'
    md = 'd'
else:
    org_paths = 'testpaths_' + 'cyl35' + '_c_v' + str(1) + '.pkl'
    md = 'c'
with open(test_path + org_paths, 'r') as f: 
    action_seq, _, _, _ = pickle.load(f)

rollout_srv = rospy.ServiceProxy('/rollout/rollout', rolloutReq)
rospy.init_node('collect_test_paths', anonymous=True)

'''
d:
TODO: 

c - data:
HAVE: elp40, poly10, sqr30, str40, cyl30, poly6, egg50, cyl45, tri50
TOFINISH: cyl35 (250,000)
TODO: rec10, rec60, sem60, cre55

c - test paths
TODO: str40

got cl:
cyl35, elp40, poly6, poly10, sqr30, str40, tri50
'''

Obj = 'str40'
path = '/home/pracsys/catkin_ws/src/t42_control/hand_control/data/dataset/'

if 1:
    test_paths = []
    Suc = []
    i = 0
    for A in action_seq:
        Af = A.reshape((-1,))
        print("Rollout number " + str(i) + ".")
        i += 1       
        
        roll = rollout_srv(Af)
        print len(roll.states)
        S = np.array(roll.states).reshape(-1,state_dim)
        suc = roll.success
        print("Got %d points with a %s trial."%(S.shape[0], 'successful' if suc else 'failed'))

        test_paths.append(S)
        Suc.append(suc)
        
        # with open(path + 'testpaths_' + Obj + '_' + md + '_v' + str(version) + '.pkl', 'w') as f: 
        #     pickle.dump([action_seq, test_paths, Obj, Suc], f)
else:
    with open(path + 'testpaths_' + Obj + '_' + md + '_v' + str(version) + '.pkl', 'r') as f: 
        action_seq, test_paths, Obj, Suc = pickle.load(f)


plt.figure(1)
plt.title('Object position')
i = 0
for S in test_paths:
    plt.plot(S[:,0], S[:,1], color=(random.random(), random.random(), random.random()), label='path ' + str(i))
    i += 1
plt.legend()

plt.figure(2)
plt.title('Angle')
i = 0
for S in test_paths:
    plt.plot(np.rad2deg(S[:,2]), color=(random.random(), random.random(), random.random()), label='path ' + str(i))
    i += 1
plt.legend()
plt.show()




