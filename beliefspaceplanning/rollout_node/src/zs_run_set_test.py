#!/usr/bin/env python

import rospy
from std_srvs.srv import Empty, EmptyResponse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Polygon
import pickle
from rollout_node.srv import rolloutReq
import time
import glob
from scipy.io import loadmat
from scipy.spatial import ConvexHull, convex_hull_plot_2d

rollout = 1

comp = 'szhang'
Set = '19c_zstest'
set_modes = ['astar']

############################# Rollout ################################
if rollout:
    rollout_srv = rospy.ServiceProxy('/rollout/rollout', rolloutReq)
    rospy.init_node('run_rollout_set', anonymous=True)
    state_dim = 4

    while 1:
    # for _ in range(10):
        for set_mode in set_modes:
            path = '/home/' + comp + '/catkin_ws/src/beliefspaceplanning/rollout_node/set/set' + Set + '/'

            files = glob.glob(path + set_mode + "*.txt")
            files_pkl = glob.glob(path + set_mode + "*.pkl")

            if len(files) == 0:
                continue

            for i in range(len(files)):

                action_file = files[i]
                if action_file.find('traj') > 0:
                    continue
                files_pkl = glob.glob(path + set_mode + "*.pkl")
                if any(action_file[:-3] + 'pkl' in f for f in files_pkl):
                    continue
                # if int(action_file[action_file.find('run')+3]) > 0:
                #     continue
                # if action_file.find(set_modes_no[0]) > 0 or action_file.find(set_modes_no[1]) > 0:
                #     continue
                pklfile = action_file[:-3] + 'pkl'

                # To distribute rollout files between computers
                ja = pklfile.find('goal')+4
                jb = ja + 1
                while not (pklfile[jb] == '_'):
                    jb += 1
                num = int(pklfile[ja:jb])#int(pklfile[ja]) if pklfile[ja+1] == '_' else int(pklfile[ja:ja+2])
                # if num == 124:
                #     continue

                print('Rolling-out goal number ' + str(num) + ': ' + action_file + '.')

                try:
                    A = np.loadtxt(action_file, delimiter=',', dtype=float)[:,:2]
                except:
                    A = np.loadtxt(action_file, delimiter=',', dtype=float)
                    print A.shape

                Af = A.reshape((-1,))
                Pro = []
                for j in range(10):
                    print("Rollout number " + str(j) + ".")
                    
                    Sro = np.array(rollout_srv(Af, [0,0,0,0]).states).reshape(-1,state_dim)

                    Pro.append(Sro)

                    with open(pklfile, 'w') as f: 
                        pickle.dump(Pro, f)
