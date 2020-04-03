#!/usr/bin/env python

import rospy
from std_srvs.srv import Empty, EmptyResponse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Polygon
import pickle
from rollout_t42.srv import rolloutReq
import time
import glob

rollout_srv = rospy.ServiceProxy('/rollout/rollout', rolloutReq)
rospy.init_node('run_rollout_set', anonymous=True)

path = '/home/pracsys/catkin_ws/src/t42_control/gpup_gp_node/data/dataset_processed/'
pklfile = 't42_cyl35_red_plan_data_discrete_v0.2_d4_m1_episodes'

with open(path + pklfile + '.obj', 'r') as f: 
    episodes = pickle.load(f)

i = 0
for e in episodes:
    if i <= 4:
        i += 1
        continue
    print("Rollout path " + str(i) + ".")
    A = e[:,4:6]
    A = np.fliplr(A)

    Af = A.reshape((-1,))
    for j in range(1):
        print("Rollout number " + str(j) + ".")
        
        Sro = np.array(rollout_srv(Af).states).reshape(-1,4)

        # Pro.append(Sro)

        with open(path + 'roll_p' + str(i) + '.pkl', 'w') as f: 
            pickle.dump(Sro, f)

    i += 1
