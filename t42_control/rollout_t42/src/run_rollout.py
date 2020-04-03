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
from scipy.io import savemat, loadmat

rollout_srv = rospy.ServiceProxy('/rollout/rollout', rolloutReq)

rospy.init_node('run_rollout_set', anonymous=True)
rate = rospy.Rate(15) # 15hz
state_dim = 4

path = '/home/pracsys/catkin_ws/src/hand_control/plans/'

rollout = 0

############################# Rollout ################################
if rollout:

    files = glob.glob(path + "*.txt")

    for i in range(len(files)):

        action_file = files[i]
        matfile = action_file[:-3]
        if action_file.find('traj') > 0 or action_file.find('roll') > 0:
            continue

        print('Rolling-out file number ' + str(i+1) + ': ' + action_file + '.')

        A = np.loadtxt(action_file, delimiter=',', dtype=float)[:,:2]

        Af = A.reshape((-1,))
        Pro = []
        for j in range(5):
            # print("Rollout number " + str(j) + ".")
            
            Sro = np.array(rollout_srv(Af).states).reshape(-1,state_dim)

            Pro.append(Sro)

            savemat(matfile + 'mat', {'D': Pro})
            with open(matfile + 'pkl', 'w') as f: 
                pickle.dump(Pro, f)

############################# Function ################################

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


def clean(D):
    print('[run_rollout] Cleaning data...')

    i = 0
    inx = []
    while i < D.shape[0]:
        if i > 0 and np.linalg.norm( D[i, 0:2] - D[i-1, 0:2] ) > 4:
            i += 1
            continue
        if D[i,0] < -50. or D[i,0] > 120:
            i += 1
            continue
        inx.append(i)
        i += 1

    return D[inx,:]

############################# Plot ################################

if not rollout:
    files = glob.glob(path + "*.pkl")
    K = []
    for k in range(len(files)):
        pklfile = files[k]
        for j in range(len(pklfile)-1, 0, -1):
            if pklfile[j] == '/':
                break
        file_name = pklfile[j+1:-4]
        if file_name == 'rollout_output':
            continue
        print('Plotting file number ' + str(k) + ': ' + file_name)

        Straj = np.loadtxt(pklfile[:-8] + 'traj.txt', delimiter=',', dtype=float)[:,:2]
        
        with open(pklfile) as f:  
            Pro = pickle.load(f)

        fig = plt.figure(k)

        for i in range(len(Pro)):
            K.append(Pro[i][0,:]*1000)
            S = clean(Pro[i][:,:4]*1000)
            for i in range(4):
                S[:,i] = medfilter(S[:,i], 20)
            plt.plot(S[:,0], S[:,1], '.-b', label='rollout')
            plt.plot(S[0,0], S[0,1], 'oc', label='rollout')

        np.savetxt(pklfile[:-8] + 'roll.txt', S, delimiter=',')
        
        plt.plot(Straj[:,0], Straj[:,1], '.-r', label='planned')
        plt.plot(Straj[0,0], Straj[0,1], 'o-r', label='planned')

    print np.mean(np.array(K), 0)
    print np.std(np.array(K), 0)
    plt.show()



if 0 and not rollout:

    files = glob.glob(path + "*.pkl")

    for k in range(len(files)):

        pklfile = files[k]

        for j in range(len(pklfile)-1, 0, -1):
            if pklfile[j] == '/':
                break
        file_name = pklfile[j+1:-4]
        
        print('Plotting file number ' + str(k+1) + ': ' + file_name)
        
        with open(pklfile) as f:  
            Pro = pickle.load(f) 

        maxR = np.max([x.shape[0] for x in Pro])
        c = np.sum([(1 if x.shape[0]==maxR else 0) for x in Pro])

        Smean = []
        Sstd = []
        for i in range(maxR):
            F = []
            for j in range(len(Pro)): 
                if Pro[j].shape[0] > i:
                    F.append(Pro[j][i])
            Smean.append( np.mean(np.array(F), axis=0) )
            Sstd.append( np.std(np.array(F), axis=0) )
        Smean = np.array(Smean)
        Sstd = np.array(Sstd)

        print("Roll-out success rate: " + str(float(c) / len(Pro)*100) + "%")

        fig = plt.figure(k)
        for S in Pro:
            plt.plot(S[:,0], S[:,1], 'r')

        plt.plot(Smean[:,0], Smean[:,1], '-b', label='rollout mean')
        # X = np.concatenate((Smean[:,0]+Sstd[:,0], np.flip(Smean[:,0]-Sstd[:,0])), axis=0)
        # Y = np.concatenate((Smean[:,1]+Sstd[:,1], np.flip(Smean[:,1]-Sstd[:,1])), axis=0)
        # plt.fill( X, Y , alpha = 0.5 , color = 'b')
        # plt.plot(Smean[:,0]+Sstd[:,0], Smean[:,1]+Sstd[:,1], '--b', label='rollout mean')
        # plt.plot(Smean[:,0]-Sstd[:,0], Smean[:,1]-Sstd[:,1], '--b', label='rollout mean')       
        # plt.title(file_name + ", suc. rate: " + str(float(c) / len(Pro)*100) + "%")
        
    plt.show()