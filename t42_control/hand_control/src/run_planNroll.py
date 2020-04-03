#!/usr/bin/env python

import rospy
from std_srvs.srv import Empty, EmptyResponse
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
import glob
from hand_control.srv import RegraspObject, close, observation, planroll, TargetAngles
from rollout_t42.srv import rolloutReq
from std_msgs.msg import String, Float32MultiArray, Bool

# def callbackObjectDrop(msg):
#     drop = msg.data

pr_srv = rospy.ServiceProxy('/planNroll', planroll)
pr_proc_srv = rospy.ServiceProxy('/planNroll/process', Empty)
rollout_srv = rospy.ServiceProxy('/rollout/rollout', rolloutReq)

open_srv = rospy.ServiceProxy('/OpenGripper', Empty) 
close_srv = rospy.ServiceProxy('/CloseGripper', close) 
move_srv = rospy.ServiceProxy('/MoveGripper', TargetAngles)
# rospy.Subscriber('/cylinder_drop', Bool, callbackObjectDrop)
# drop = True

from gpup_gp_node_exp.srv import gpup_transition, batch_transition, one_transition
gp_srv = rospy.ServiceProxy('/gp/transition', batch_transition)
naive_srv = rospy.ServiceProxy('/gp/transitionOneParticle', one_transition)

rospy.init_node('run_planNroll', anonymous=True)

goals = np.array([[-25, 75]])
set_modes = ['naive', 'critic']

msg = planroll()

state_dim = 4

def slow_open():
    for _ in range(30):
        move_srv(np.array([-6.,-6.]))
        rospy.sleep(0.1)

def ResetArm():
    open_srv()
    print('[] Press key to insert object...')
    raw_input()
    rospy.sleep(3.)
    print('[] Waiting to grasp object...')
    close_srv()
    print('[] Lift object and press key...')
    raw_input()

def medfilter(x, W = 40):
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

if 1:
    for goal in goals:
        for set_mode in set_modes:
            print "Running " + set_mode + " with goal " + str(goal) + "..."
            msg.goal = goal
            msg.planning_algorithm = set_mode

            # First rollout is from the known start state
            res = pr_srv(goal, set_mode)
            File = res.file
            # File = '/home/pracsys/catkin_ws/src/t42_control/hand_control/plans/critic_goal-30.0_60.0_n61981_plan.'# Overload
            # File = '/home/pracsys/catkin_ws/src/t42_control/hand_control/plans/critic_goal-30.0_60.0_n33397_plan.'# Overload
            # File = '/home/pracsys/catkin_ws/src/t42_control/hand_control/plans/critic_goal-30.0_72.0_n94352_plan.'# 

            # Now, more runs from approximately the start state
            A = np.loadtxt(File + 'txt', delimiter=',', dtype=float)[:,:2]

            P = []
            Af = A.reshape((-1,))
            for j in range(1):
                print("Rollout number " + str(j) + ".")

                # ResetArm()
                
                Sro = np.array(rollout_srv(Af).states).reshape(-1,state_dim)

                P.append(Sro)

                # slow_open()

                with open(File + '.pkl', 'w') as f: 
                    pickle.dump(P, f)
exit(1)
# pr_proc_srv()

path = '/home/pracsys/catkin_ws/src/t42_control/hand_control/plans/'
results_path = '/home/pracsys/catkin_ws/src/t42_control/hand_control/plans/rollout_results/'

def Plot():
    files = glob.glob(path + "*.pkl")
    r = 8

    for pklfile in files:
        print
        print pklfile

        with open(pklfile) as f:  
            Pro, goal, Straj, A = pickle.load(f)

        A = np.loadtxt(pklfile[:-3] + 'txt', delimiter=',', dtype=float)[:,:2]
        maxR = A.shape[0]+1
        # maxX = np.max([x.shape[0] for x in Pro])

        c = np.sum([(1 if x.shape[0]==maxR else 0) for x in Pro])
        c = float(c) / len(Pro)*100
        print("Finished episode success rate: " + str(c) + "%")

        fig, ax = plt.subplots(figsize=(12,12))
        goal_plan = plt.Circle((goal[0], goal[1]), r, color='m')
        ax.add_artist(goal_plan)

        p = 0
        for S in Pro:
            for i in range(2):
                S[:,i] = medfilter(S[:,i], 20)
            if S.shape[0] < maxR or np.linalg.norm(S[-1,:2]-goal) > r:
                plt.plot(S[:,0], S[:,1], '-r')
                plt.plot(S[-1,0], S[-1,1], 'or')
            else:
                plt.plot(S[-1,0], S[-1,1], 'ob')
                plt.plot(S[:,0], S[:,1], '-b')
                p += 1
        p = float(p) / len(Pro)*100
        print("Reached goal success rate: " + str(p) + "%")

        for i in range(len(pklfile)-1, 0, -1):
            if pklfile[i] == '/':
                break

        plt.plot(Straj[:,0], Straj[:,1], '-k', linewidth = 2.7, label='Planned path')
        plt.savefig(results_path + pklfile[i+1:-4] + '.png', dpi=300)


if 0:
    state_dim = 12

    for _ in range(1):

        files = glob.glob(path + "*.txt")
        files_pkl = glob.glob(path + "*.pkl")

        if len(files) == 0:
            continue

        for i in range(len(files)):

            action_file = files[i]
            if action_file.find('traj') > 0 or action_file.find('_plan.') < 0:
                continue
            if any(action_file[:-3] + 'pkl' in f for f in files_pkl):
                continue
            pklfile = action_file[:-3] + 'pkl'

            # if action_file.find('naive_goal90.0_72.0_n47414_plan') < 0:
            #     continue

            trajfile = pklfile[:-8] + 'traj.txt'
            Straj = np.loadtxt(trajfile, delimiter=',', dtype=float)[:,:2]

            print('Rolling-out file number ' + str(i+1) + ': ' + action_file + '.')

            A = np.loadtxt(action_file, delimiter=',', dtype=float)[:,:2]
            # A = np.tile(A, (3,1))

            # Get goal
            j = pklfile.find('goal')+4
            j1 = j
            while pklfile[j] != '_':
                j += 1
            goal = np.array([float(pklfile[j1:j]), 0])
            j1 = j + 1
            j = j1
            while pklfile[j] != '_':
                j += 1
            goal[1] = float(pklfile[j1:j])
            print goal

            # if np.all(goal == np.array([90.,72.])):
            #     A = A[:995,:]
            A = A[range(0,A.shape[0],3),:]

            Af = A.reshape((-1,))
            Pro = []
            for j in range(10):
                print("Rollout number " + str(j) + ".")

                ResetArm()
                
                print("Rolling out...")
                Sro = np.array(rollout_srv(Af).states).reshape(-1,state_dim)
                Pro.append(Sro)

                print("Dropping...")
                slow_open()

                print("Saving to " + pklfile)
                with open(pklfile, 'w') as f: 
                    pickle.dump([Pro, goal, Straj, A], f)

            Plot()

Plot()


# pklfile = '/home/pracsys/catkin_ws/src/t42_control/hand_control/plans/robust_goal90.0_72.0_n29801_plan/robust_goal90.0_72.0_n29801_plan.pkl'

# with open(pklfile) as f:  
#     Pro, goal, Straj, A = pickle.load(f)

# s = []
# for S in Pro:
#     s.append(S[0,:])
# print np.mean(np.array(s), 0)

# A = A[range(0, A.shape[0], 10), :]
# A = np.tile(np.array([1,-1]), (500,1))

# action_file = '/home/pracsys/catkin_ws/src/t42_control/hand_control/plans/robust_goal80.0_81.0_n68930_plan.txt'
# A = np.loadtxt(action_file, delimiter=',', dtype=float)[:,:2]

# traj_file = '/home/pracsys/catkin_ws/src/t42_control/hand_control/plans/robust_goal80.0_81.0_n68930_traj.txt'
# Straj = np.loadtxt(traj_file, delimiter=',', dtype=float)[:,:4]

# # S = Pro[0][:,:4]
# # S[:,4:] *= 1000.

# s_start = Straj[0,:]
# state_dim = 4#S.shape[1]

# print s_start
# s = np.copy(s_start)# + np.random.normal(0, sigma_start)
# Ypred = s.reshape(1,state_dim)

# Sp = np.tile(s, (100,1))

# print("Running (open loop) path...")
# p_naive = 1
# for i in range(0, A.shape[0]):
#     print("[] Step " + str(i) + " of " + str(A.shape[0]))
#     a = A[i,:]
#     print s, a

#     # res = naive_srv(s.reshape(-1,1), a)
#     # s_next = np.array(res.next_state)
#     # s = np.copy(s_next)

#     res = gp_srv(Sp.reshape(-1,1), a)
#     S_next = np.array(res.next_states).reshape(-1, state_dim)
#     s_next = np.mean(S_next, 0)
#     Sp = np.copy(S_next)
#     s = np.copy(s_next)

#     Ypred = np.append(Ypred, s_next.reshape(1, state_dim), axis=0)

# # h = [40, 40, 100, 100]
# # for i in range(state_dim):
# #     try:
# #         S[:,i] = medfilter(S[:,i], h[i])
# #     except:
# #         S[:,i] = medfilter(S[:,i], 40)

# # plt.plot(S[:,0], S[:,1], '.-k', label='rollout')
# plt.plot(Straj[:,0], Straj[:,1], '.-r', label='Plan')
# plt.plot(Ypred[:,0], Ypred[:,1], '.-c', label='Naive')

# plt.show()