#!/usr/bin/env python

import rospy
import numpy as np
from hand_control.msg import plan_for_goal_request, plan_for_goal_response
from rollout_t42.srv import rolloutReq
from hand_control.srv import RegraspObject, close, observation, planroll, TargetAngles
from std_msgs.msg import String, Float32MultiArray, Bool
from std_srvs.srv import Empty, EmptyResponse
import pickle
import matplotlib.pyplot as plt

# Example: rosservice call /planNroll '{goal: [47,102],planning_algorithm: naive}'

class planRoll():

    planning_done = False

    path = '/home/pracsys/catkin_ws/src/t42_control/hand_control/plans/'

    goal_counter = 0
    publishToPlanner = True
    planning_algorithm = 'naive'
    drop = True
    arm_status = ' '
    trigger = False # Enable collection
    radius = 8.0
    state_dim = 4

    def __init__(self):
        rospy.init_node('planNroll', anonymous=True)

        self.planning_request_pub = rospy.Publisher('/planning/request', plan_for_goal_request, queue_size=1)
        rospy.Subscriber('/planning/response', plan_for_goal_response, self.planningResponseCallback)
        self.rollout_srv = rospy.ServiceProxy('/rollout/rollout', rolloutReq)
        self.arm_reset_srv = rospy.ServiceProxy('/RegraspObject', RegraspObject)
        rospy.Service('/planNroll', planroll, self.run)
        rospy.Service('/planNroll/process', Empty, self.process)
        rospy.Subscriber('/cylinder_drop', Bool, self.callbackObjectDrop)
        rospy.Subscriber('/ObjectIsReset', String, self.callbackTrigger)
        self.obs_srv = rospy.ServiceProxy('/observation', observation)
        self.open_srv = rospy.ServiceProxy('/OpenGripper', Empty) 
        self.close_srv = rospy.ServiceProxy('/CloseGripper', close) 
        self.move_srv = rospy.ServiceProxy('/MoveGripper', TargetAngles)

        self.results_path = self.path + 'results/'

        self.rate = rospy.Rate(10)
        print('[plan_call] Ready to plan and roll...')
        rospy.spin()

        # while not rospy.is_shutdown():
        #     rate.sleep()
        #     if (self.publishToPlanner):
        #         print('[plan_call] Calling planner with goal ' + str(self.goals[self.goal_counter]) + '...')
        #         self.plan_for_goal(self.goals[self.goal_counter])
        #     rate.sleep()

    def ResetArm(self):
        # while 1:
        #     if not self.trigger and self.arm_status == 'waiting':
        #         print('[rollout_action_publisher] Waiting for arm to grasp object...')
        #         self.arm_reset_srv()
        #         rospy.sleep(1.0)
        #     self.rate.sleep()
        #     if self.arm_status != 'moving' and self.trigger:
        #         self.rate.sleep()
        #         if self.drop: # Check if really grasped
        #             self.trigger = False
        #             print('[rollout_action_publisher] Grasp failed. Restarting')
        #             continue
        #         else:
        #             break
        self.open_srv()
        print('[planNroll] Press key to insert object...')
        raw_input()
        rospy.sleep(3.)
        print('[planNroll] Waiting to grasp object...')
        self.close_srv()
        print('[planNroll] Press key to start...')
        raw_input()
        # self.trigger = False

    def run(self, msg):
        goal = msg.goal
        self.goal = goal
        self.planning_algorithm = msg.planning_algorithm

        # Reset arm
        # self.trigger = False
        # self.ResetArm()
        # rospy.sleep(2.)

        # Get state
        for _ in range(10):
            start = np.array(self.obs_srv().state)
            self.rate.sleep()
        start = start[[0,1,11,12]] # For cylinder
        start[:2] *= 1000
        # start = np.array([  2.03328312e+01,   1.11295141e+02,   9.91000000e+01,  -9.86000000e+01])
        # start = start[:self.state_dim]

        print "[plan_call] Start state: " + str(start)
        print "[plan_call] Goal state: " + str(goal)

        self.id = str(np.random.randint(100000))

        # Plan
        print('[plan_call] Calling planner...')
        self.plan_for_goal(start, goal)

        while not self.planning_done:
            self.rate.sleep()

        # Rollout
        # print('[plan_call] Got path. Ready to roll out. Press key...')
        # raw_input()
        # print('[plan_call] Got path. Rolling out...')
        # A = self.planning_actions.reshape((-1,2))
        # if A.size == 0:
        #     rospy.logerr('[plan_call] Recieved empty path.')
        #     return
        # rospy.sleep(2.)
        # S = np.array(self.rollout_srv(A).states).reshape(-1, self.state_dim)
        # while not self.drop:
        #     self.move_srv(np.array([-3.,-3.]))
        #     self.rate.sleep()

        print('Finished, getting states...')
        
        # Save
        # print('[plan_call] Saving rollout...')
        pklfile = self.path + self.planning_algorithm + '_goal' + str(goal[0]) + '_' + str(goal[1]) + '_n' + self.id + '_plan.pkl'
        # with open(pklfile, 'w') as f: 
        #     pickle.dump(S, f)
        # File = self.path + self.planning_algorithm + '_goal' + str(goal[0]) + '_' + str(goal[1]) + '_n' + self.id + '_rollout.txt'
        # np.savetxt(File, S, delimiter = ', ')

        print('[plan_call] Process ended.')
        # self.open_srv()

        if 0:
            fig, ax = plt.subplots()
            St = np.copy(S)
            for i in range(2):
                St[:,i] = self.medfilter(St[:,i], 20)
            plt.plot(St[:,0],St[:,1], '-b', label='rollout filtered')
            plt.plot(S[:,0],S[:,1], '--r', label='rollout')
            plt.plot(self.planned_path[:,0],self.planned_path[:,1], '-k', label='plan')
            g = plt.Circle((self.goal[0], self.goal[1]), self.radius, color='m', label='goal region')
            ax.add_artist(g)
            plt.legend()
            plt.savefig(self.results_path + '/' + self.planning_algorithm + '_goal' + str(goal[0]) + '_' + str(goal[1]) + '_n' + self.id + '_init.png')
            plt.show()

        return {'suc': self.planning_succeeded, 'file': pklfile[:-3]}

    def callbackObjectDrop(self, msg):
        self.drop = msg.data

    def callbackTrigger(self, msg):
        self.arm_status = msg.data
        if not self.trigger and self.arm_status == 'finished':
            self.trigger = True

    def medfilter(self, x, W):
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

            
    def plan_for_goal(self, start, goal):
        
        msg = plan_for_goal_request()

        msg.start_state = start
        msg.goal_state = goal
        msg.goal_radius = self.radius
        msg.time_limit = 1500 #seconds
        msg.probability_success_threshold = 0.7 #affects SVM validity check 
        msg.planning_algorithm = self.planning_algorithm

        print('Requesting goal %d with %s algorithm...'%(self.goal_counter, self.planning_algorithm))
        self.planning_request_pub.publish(msg)

        self.planning_done = False


    def planningResponseCallback(self, msg):

        print "Something returned..."
        self.planning_actions = msg.planned_actions
        self.planned_path = msg.planned_path
        self.planning_succeeded = msg.reached_goal

        self.planning_done = True

        self.planning_actions = np.array(self.planning_actions).reshape((-1,2))
        self.planned_path = np.array(self.planned_path).reshape((-1, self.state_dim))

        File = self.planning_algorithm + '_goal' + str(self.goal[0]) + '_' + str(self.goal[1]) + '_n' + self.id + '_plan.txt'
        np.savetxt(self.path + File, self.planning_actions, delimiter = ', ')
        File =  self.planning_algorithm + '_goal' + str(self.goal[0]) + '_' + str(self.goal[1]) + '_n' + self.id + '_traj.txt'
        np.savetxt(self.path + File, self.planned_path, delimiter = ', ')
        
        F  = open(self.path + 'log.txt', 'a')
        F.write("Number: " + self.id + ", " + self.planning_algorithm + ": goal - " + str(self.goal) + ", success: " + str(self.planning_succeeded*1) + "\n")
        F.close()

        print('[plan_call] Planned received and saved in ' + File + '.')

    def process(self, msg):
        import glob

        
        set_modes = ['robust']#, 'naive']

        for set_mode in set_modes:

            fo  = open(self.results_path + set_mode + '.txt', 'wt') 

            files = glob.glob(self.path + "*.pkl")

            for k in range(len(files)):

                pklfile = files[k]
                if pklfile.find(set_mode) < 0:
                    continue
                j = pklfile.find('goal')+4
                j1 = j
                while pklfile[j] != '_':
                    j += 1
                ctr = np.array([float(pklfile[j1:j]), 0])
                j1 = j + 1
                j = j1
                while pklfile[j] != '_':
                    j += 1
                ctr[1] = float(pklfile[j1:j])
                print ctr

                for j in range(len(pklfile)-1, 0, -1):
                    if pklfile[j] == '/':
                        break
                file_name = pklfile[j+1:-4]

                trajfile = pklfile[:-8] + 'traj.txt'
                print trajfile
                Straj = np.loadtxt(trajfile, delimiter=',', dtype=float)[:,:2]

                print('Plotting file number ' + str(k+1) + ': ' + file_name)
                
                with open(files[k]) as f:  
                    Pu = pickle.load(f)

                # Apply filter to episode
                P = []
                Ss = []
                for S in Pu:
                    Ss.append(S[0,:])
                    for i in range(4):
                        S[:,i] = self.medfilter(S[:,i], 20)
                    P.append(S)

                print np.mean(np.array(Ss), 0)

                A = np.loadtxt(pklfile[:-3] + 'txt', delimiter=',', dtype=float)[:,:2]
                maxR = A.shape[0]-1
                
                # Smean = S
                
                # fig = plt.figure(k)
                fig, ax = plt.subplots()
                p = 0
                su = 0
                for S in P:
                    plt.plot(S[:,0], S[:,1], 'r')
                    if S.shape[0] < maxR:
                        plt.plot(S[-1,0], S[-1,1], 'oc')
                    else:
                        p += 1

                    if np.linalg.norm(S[-1,:2]-ctr) <= 1.2*self.radius:
                        su += 1
                        
                plt.plot(ctr[0], ctr[1], 'om')
                goal = plt.Circle((ctr[0], ctr[1]), 1.2*self.radius, color='m')
                ax.add_artist(goal)

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
                plt.title(file_name)
                plt.axis('equal')

                for i in range(len(pklfile)-1, 0, -1):
                    if pklfile[i] == '/':
                        break

                fo.write(pklfile[i+1:-4] + ': ' + str(su) + '/' + str(len(P)) + '\n')
                plt.savefig(self.results_path + '/' + pklfile[i+1:-4] + '.png')

            fo.close()
        

if __name__ == '__main__':
    
    try:
        planRoll()
    except rospy.ROSInterruptException:
        pass