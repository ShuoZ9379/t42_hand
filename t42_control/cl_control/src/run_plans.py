#!/usr/bin/env python

import rospy
from std_srvs.srv import Empty, EmptyResponse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Polygon
import pickle
from cl_control.srv import pathTrackReq
import time
import glob

rollout = 0

comp = 'pracsys'

Set = '1'
# set_modes = ['naive']
set_modes = ['robust', 'naive']

############################# Rollout ################################
if rollout:
    track_srv = rospy.ServiceProxy('/control', pathTrackReq)
    rospy.init_node('run_contol_set', anonymous=True)
    state_dim = 4

    for set_mode in set_modes:
        path = '/home/' + comp + '/catkin_ws/src/t42_control/hand_control/plans/' # /set' + Set + '/'

        files = glob.glob(path + set_mode + "*.txt")
        files_pkl = glob.glob(path + set_mode + "*.pkl")

        for i in range(len(files)):

            traj_file = files[i]
            
            if traj_file.find('_plan') > 0:
                continue
            if any(traj_file[:-3] + 'pkl' in f for f in files_pkl):
                continue
            pklfile = traj_file[:-3] + 'pkl'

            # To distribute rollout files between computers
            # ja = pklfile.find('goal')+4
            # if int(pklfile[ja]) <= 4:
            #     continue

            print('Rolling-out file number ' + str(i+1) + ': ' + traj_file + '.')

            try:
                S = np.loadtxt(traj_file, delimiter=',', dtype=float)[:,:4]
            except:
                continue

            if S.shape[0] < 10:
                continue

            Sf = S.reshape((-1,))
            Pro = []
            c = 0
            f = 0
            while c < 10:
                print("Rollout number " + str(c) + ".")
                
                res = track_srv(Sf.reshape((-1,)))
                Sreal = np.array(res.real_path).reshape(-1, state_dim)
                Areal = np.array(res.actions).reshape(-1, 2)
                success = res.success
                i_path = res.i_path

                if i_path > 60 or f > 4:
                    Pro += [(c, Sreal, Areal, success, i_path, S)]
                    with open(pklfile, 'w') as f: 
                        pickle.dump(Pro, f)
                    c += 1
                    f = 0
                else:
                    f += 1

############################# Plot ################################

def medfilter(x, W):
    w = int(W/2)
    x_new = np.copy(x)
    for i in range(1, x.shape[0]-1):
        if i < w:
            x_new[i] = np.mean(x[:i+w])
        elif i > x.shape[0]-w:
            x_new[i] = np.mean(x[i-w:])
        else:
            x_new[i] = np.mean(x[i-w:i+w])
    return x_new


rp = 7.
r = 10.

set_num = Set
set_modes = ['naive', 'robust']

if not rollout and 1:
    results_path = '/home/' + comp + '/catkin_ws/src/t42_control/hand_control/plans/results_control/' # /set' + Set + '/'

    for set_mode in set_modes:

        path = '/home/' + comp + '/catkin_ws/src/t42_control/hand_control/plans/' # /set' + Set + '/'

        fo  = open(results_path + set_mode + '.txt', 'wt') 

        files = glob.glob(path + "*.pkl")

        for k in range(len(files)):

            pklfile = files[k]
            if pklfile.find(set_mode) < 0:
                continue
            if pklfile.find('_plan') > 0:
                continue

            o = pklfile.find('goal')+4
            o1 = pklfile[o:].find('_') + 1 + o
            o2 = pklfile[o1:].find('_') +  o1
            ctr = [float(pklfile[o:o1-1]), float(pklfile[o1:o2])] # Goal center
            print ctr

            for j in range(len(pklfile)-1, 0, -1):
                if pklfile[j] == '/':
                    break
            file_name = pklfile[j+1:-4]

            trajfile = pklfile[:-8] + 'traj.txt'
            Straj = np.loadtxt(trajfile, delimiter=',', dtype=float)[:,:2]

            print('Plotting file number ' + str(k+1) + ': ' + file_name)
            
            with open(pklfile) as f:  
                Pro = pickle.load(f)

            # Apply filter to episode
            for P in Pro:
                S = P[1]
                for i in range(4):
                    S[:,i] = medfilter(S[:,i], 20)

            i = 0
            while i < len(Pro):
                if Pro[i][1].shape[0] == 1:
                    del Pro[i]
                else:
                    i += 1 

            A = np.loadtxt(pklfile[:-3] + 'txt', delimiter=',', dtype=float)[:,:2]
            maxR = A.shape[0]
            # maxX = np.max([x.shape[0] for x in Pro])

            c = np.sum([(1 if maxR - x[1].shape[0] < 10 else 0) for x in Pro])

            # Smean = []
            # Sstd = []
            # for i in range(min(maxR, maxX)):
            #     F = []
            #     for j in range(len(Pro)): 
            #         if Pro[j].shape[0] > i:
            #             F.append(Pro[j][i])
            #     Smean.append( np.mean(np.array(F), axis=0) )
            #     Sstd.append( np.std(np.array(F), axis=0) )
            # Smean = np.array(Smean)
            # Sstd = np.array(Sstd)
            
            c = float(c) / len(Pro)*100
            print("Finished episode success rate: " + str(c) + "%")

            # fig = plt.figure(k)
            fig, ax = plt.subplots()
            p = 0
            for P in Pro:
                S = P[1]
                plt.plot(S[:,0], S[:,1], 'r')
                if not P[3]:
                    plt.plot(S[-1,0], S[-1,1], 'or')
                else:
                    plt.plot(S[-1,0], S[-1,1], 'og')

                if np.linalg.norm(S[-1,:2]-ctr) <= r:
                    p += 1
            p = float(p) / len(Pro)*100
            print("Reached goal success rate: " + str(p) + "%")

            plt.plot(ctr[0], ctr[1], 'om')
            goal = plt.Circle((ctr[0], ctr[1]), r, color='m')
            ax.add_artist(goal)
            goal_plan = plt.Circle((ctr[0], ctr[1]), 8, color='w')
            ax.add_artist(goal_plan)

            try:
                for o in Obs:
                    obs = plt.Circle(o[:2], o[2])#, zorder=10)
                    ax.add_artist(obs)
            except:
                pass

            plt.plot(Straj[:,0], Straj[:,1], '-k', linewidth=3.5, label='Planned path')

            # plt.plot(Smean[:,0], Smean[:,1], '-b', label='rollout mean')
            # X = np.concatenate((Smean[:,0]+Sstd[:,0], np.flip(Smean[:,0]-Sstd[:,0])), axis=0)
            # Y = np.concatenate((Smean[:,1]+Sstd[:,1], np.flip(Smean[:,1]-Sstd[:,1])), axis=0)
            # plt.fill( X, Y , alpha = 0.5 , color = 'b')
            # plt.plot(Smean[:,0]+Sstd[:,0], Smean[:,1]+Sstd[:,1], '--b', label='rollout mean')
            # plt.plot(Smean[:,0]-Sstd[:,0], Smean[:,1]-Sstd[:,1], '--b', label='rollout mean')       
            plt.title(file_name + ", CL, suc. rate: " + str(c) + "%, " + "goal suc.: " + str(p) + "%")
            plt.axis('equal')

            for i in range(len(pklfile)-1, 0, -1):
                if pklfile[i] == '/':
                    break

            fo.write(pklfile[i+1:-4] + ': ' + str(c) + ', ' + str(p) + '\n')
            plt.savefig(results_path + '/' + pklfile[i+1:-4] + '.png')

        fo.close()
        
    # plt.show()

if not rollout and 0:
    results_path = '/home/' + comp + '/catkin_ws/src/beliefspaceplanning/rollout_node/set/set' + Set + '/results_goal/'
    PL = {set_modes[0]: 0., set_modes[1]: 0., set_modes[2]: 0.}

    fo  = open(results_path + 'set' + set_num + '.txt', 'wt') 

    for goal_num in range(C.shape[0]):
        ctr = C[goal_num]
        print "ctr: ", ctr

        fo.write('\ngoal ' + str(goal_num) + ': ' + str(C[goal_num]) + '\n')
        
        fig = plt.figure(figsize=(20,7))

        a = 0
        for set_mode in set_modes:

            path = '/home/' + comp + '/catkin_ws/src/beliefspaceplanning/rollout_node/set/set' + Set + '/'

            files = glob.glob(path + set_mode + "*.pkl")

            found = False
            for k in range(len(files)):
                pklfile = files[k]
                ja = pklfile.find('goal')+4
                ja = pklfile[ja:ja+2] if not pklfile[ja+1] == '_' else pklfile[ja]
                if int(ja) == goal_num:
                    found = True
                    break

            n = len(fig.axes)
            for i in range(n):
                fig.axes[i].change_geometry(1, n+1, i+1)
            ax = fig.add_subplot(1, n+1, n+1)#, aspect='equal')
            plt.plot(ctr[0], ctr[1], 'om')
            goal = plt.Circle((ctr[0], ctr[1]), r, color='m')
            ax.add_artist(goal)
            goal_plan = plt.Circle((ctr[0], ctr[1]), rp, color='w')
            ax.add_artist(goal_plan)
            # plt.ylim([40,130])
            # plt.xlim([-100, 100])

            if not found:
                plt.title(set_mode + ", No plan")
                fo.write(set_mode + ': No plan\n')
                continue
            PL[set_mode] += 1.
            pklfile = files[k]

            for j in range(len(pklfile)-1, 0, -1):
                if pklfile[j] == '/':
                    break
            file_name = pklfile[j+1:-4]

            trajfile = pklfile[:-8] + 'traj.txt'
            Straj = np.loadtxt(trajfile, delimiter=',', dtype=float)[:,:2]

            with open(pklfile) as f:  
                Pro = pickle.load(f) 

            i = 0
            while i < len(Pro):
                if Pro[i].shape[0] == 1:
                    del Pro[i]
                else:
                    i += 1

            A = np.loadtxt(pklfile[:-3] + 'txt', delimiter=',', dtype=float)[:,:2]
            maxR = A.shape[0]+1
            maxX = np.max([x.shape[0] for x in Pro])
            
            c = np.sum([(1 if x.shape[0]==maxR else 0) for x in Pro])

            Smean = []
            Sstd = []
            for i in range(min(maxR, maxX)):
                F = []
                for j in range(len(Pro)): 
                    if Pro[j].shape[0] > i:
                        F.append(Pro[j][i])
                Smean.append( np.mean(np.array(F), axis=0) )
                Sstd.append( np.std(np.array(F), axis=0) )
            Smean = np.array(Smean)
            Sstd = np.array(Sstd)

            c = float(c) / len(Pro)*100

            p = 0
            t = True
            for S in Pro:
                if t:
                    plt.plot(S[:,0], S[:,1], 'r', label='rollouts')
                    t = False
                else:
                    plt.plot(S[:,0], S[:,1], 'r')

                if S.shape[0] < maxR:
                    plt.plot(S[-1,0], S[-1,1], 'oc')

                if np.linalg.norm(S[-1,:2]-ctr) <= r:
                    p += 1
            p = float(p) / len(Pro)*100

            try:
                for o in Obs:
                    obs = plt.Circle(o[:2], o[2])#, zorder=10)
                    ax.add_artist(obs)
            except:
                pass

            plt.plot(Straj[:,0], Straj[:,1], '-k', linewidth=3.5, label='Planned path')

            plt.plot(Smean[:,0], Smean[:,1], '-b', linewidth=3.0, label='rollout mean')
            # X = np.concatenate((Smean[:,0]+Sstd[:,0], np.flip(Smean[:,0]-Sstd[:,0])), axis=0)
            # Y = np.concatenate((Smean[:,1]+Sstd[:,1], np.flip(Smean[:,1]-Sstd[:,1])), axis=0)
            # plt.fill( X, Y , alpha = 0.5 , color = 'b')
            # plt.plot(Smean[:,0]+Sstd[:,0], Smean[:,1]+Sstd[:,1], '--b', label='rollout mean')
            # plt.plot(Smean[:,0]-Sstd[:,0], Smean[:,1]-Sstd[:,1], '--b', label='rollout mean')       
            plt.title(set_mode + ", suc. rate: " + str(c) + "%, " + "goal suc.: " + str(p) + "%")
            # plt.axis('equal')

            plt.plot(Smean[0,0], Smean[0,1], 'og', markersize=14 , label='start state')
            
            plt.legend()

            for i in range(len(pklfile)-1, 0, -1):
                if pklfile[i] == '/':
                    break

            fo.write(set_mode + ': ' + str(c) + ', ' + str(p) + '\n')
        plt.savefig(results_path + 'set' + str(set_num) + '_goal' + str(goal_num) + '.png', dpi=300) 
        # plt.show()   
    
    fo.write('\n\nPlanning success rate: \n')
    for k in list(PL.keys()):
        fo.write(k + ': ' + str(PL[k]/13.*100.) + '\n')
    
    fo.close()
    # plt.show()


        
        

    