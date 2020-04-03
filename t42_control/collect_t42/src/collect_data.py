#!/usr/bin/env python

import rospy
import numpy as np
import time
import random
from std_msgs.msg import String, Float32MultiArray, Bool
from std_srvs.srv import Empty, EmptyResponse
from hand_control.srv import observation, IsDropped, TargetAngles, RegraspObject, close
from transition_experience import *
# from common_msgs_gl.srv import SendBool, SendDoubleArray
import glob
from bowen_pose_estimate.srv import recordHandPose

class collect_data():

    gripper_closed = False
    trigger = True # Enable collection
    discrete_actions = True # Discrete or continuous actions
    arm_status = ' '
    global_trigger = True

    num_episodes = 0
    episode_length = 1000000 # !!!
    desired_action = np.array([0.,0.])
    drop = True

    A = np.array([[1.0,1.0],[-1.,-1.],[-1.,1.],[1.,-1.],[1.5,0.],[-1.5,0.],[0.,-1.5],[0.,1.5]])

    texp = transition_experience(Load = False, discrete = discrete_actions, postfix = 'bu')

    def __init__(self):
        rospy.init_node('collect_data', anonymous=True)

        rospy.Subscriber('/gripper/gripper_status', String, self.callbackGripperStatus)
        pub_gripper_action = rospy.Publisher('/collect/action', Float32MultiArray, queue_size=10)
        rospy.Service('/collect/trigger_episode', Empty, self.callbackManualTrigger)
        rospy.Service('/collect/save', Empty, self.callbackSave)
        self.recorderSave_srv = rospy.ServiceProxy('/actor/save', Empty)
        obs_srv = rospy.ServiceProxy('/observation', observation)
        drop_srv = rospy.ServiceProxy('/IsObjDropped', IsDropped)
        self.move_srv = rospy.ServiceProxy('/MoveGripper', TargetAngles)
        rospy.Subscriber('/ObjectIsReset', String, self.callbackTrigger)
        arm_reset_srv = rospy.ServiceProxy('/RegraspObject', RegraspObject)
        # record_srv = rospy.ServiceProxy('/record_hand_pose', recordHandPose)
        rospy.Subscriber('/cylinder_drop', Bool, self.callbackObjectDrop)
        recorder_srv = rospy.ServiceProxy('/actor/trigger', Empty)

        self.collect_mode = 'plan' # 'manual' or 'auto' or 'plan'
        
        if self.collect_mode == 'manual':
            rospy.Subscriber('/keyboard/desired_action', Float32MultiArray, self.callbackDesiredAction)
            ResetKeyboard_srv = rospy.ServiceProxy('/ResetKeyboard', Empty)

        # if self.collect_mode == 'plan':
        # filest = glob.glob('/home/pracsys/catkin_ws/src/hand_control/plans/*.txt')
        # files = []
        # for f in filest:
        #     if f.find('traj') == -1:
        #         files.append(f)
        
        close_srv = rospy.ServiceProxy('/CloseGripper', close)
        open_srv = rospy.ServiceProxy('/OpenGripper', Empty) 

        msg = Float32MultiArray()

        # msgd = record_srv()
        open_srv()
        time.sleep(2.)

        print('[collect_data] Ready to collect...')
        print('[collect_data] Position object and press key...')
        raw_input()

        rate = rospy.Rate(2.5) # 15hz
        count_fail = 0
        self.first = False
        while not rospy.is_shutdown():

            if self.global_trigger:

                if self.collect_mode != 'manual':
                    if np.random.uniform() > 0.: #0.5
                        self.collect_mode = 'plan'
                        self.first = True
                        # files = glob.glob('/home/pracsys/catkin_ws/src/t42_control/hand_control/plans/*.txt')
                        # if len(files)==0:
                        #     self.collect_mode = 'auto'
                    else:
                        self.collect_mode = 'auto'

                if self.trigger:# and self.arm_status == 'waiting':
                    
                    # if self.collect_mode == 'manual': 
                    #     ResetKeyboard_srv()
                    if 1:#drop_srv().dropped:
                        print "Closing"
                        close_srv()
                        time.sleep(1.0)
                        # arm_reset_srv()
                        # print('[collect_data] Waiting for arm to grasp object...')
                        # print('[collect_data] Position object and press key...')
                        # raw_input()
                    else:
                        self.trigger = True

                if self.trigger:# self.arm_status != 'moving' and self.trigger:

                    print('[collect_data] Verifying grasp...')
                    if drop_srv().dropped: # Check if really grasped
                        self.trigger = False
                        print('[collect_data] Grasp failed. Restarting')
                        open_srv()
                        count_fail += 1
                        if count_fail == 60:
                            self.global_trigger = False
                        continue

                    count_fail = 0

                    print('[collect_data] Starting episode %d...' % self.num_episodes)

                    self.num_episodes += 1
                    Done = False

                    if self.collect_mode == 'plan':
                        # ia = np.random.randint(len(files))
                        # print('[collect_data] Rolling out file: ' + files[ia])
                        # Af = np.loadtxt(files[ia], delimiter = ',', dtype=float)[:,:2]
                        if np.random.uniform() > 0.:
                            Af = np.tile(np.array([-1.,1.]), (np.random.randint(50,160), 1))
                        else:
                            Af = np.tile(np.array([1.,-1.]), (np.random.randint(100,500), 1))
                        print('[collect_data] Rolling out shooting with %d steps.'%Af.shape[0])
                    
                    # Start episode
                    recorder_srv()
                    n = 0
                    action = np.array([0.,0.])
                    state = np.array(obs_srv().state)
                    T = rospy.get_time()
                    for ep_step in range(self.episode_length):

                        if self.collect_mode == 'plan' and Af.shape[0] == ep_step and self.first: # Finished planned path and now applying random actions
                            # self.collect_mode = 'auto'
                            self.first = False
                            n = 0
                            print('[collect_data] Running random actions...')
                            episode_cur = 0
                            episode_max = np.random.randint(200)
                            print('[collect_data] Rolling out random actions with %d steps.'%episode_max)
                        
                        if n == 0:
                            if self.collect_mode == 'auto':
                                action, n = self.choose_action()
                            elif self.collect_mode == 'manual':
                                action = self.desired_action
                                n = 1
                            else: # 'plan'
                                if self.first:
                                    n = 1
                                    action = Af[ep_step, :] 
                                else:
                                    action, n = self.choose_action()                           
                        print action, ep_step
                        if not self.first:
                            episode_cur += 1
                        
                        msg.data = action
                        pub_gripper_action.publish(msg)
                        suc = self.move_srv(action).success
                        n -= 1

                        # Get observation
                        next_state = np.array(obs_srv().state)

                        # if self.drop:
                            # raw_input()

                        if suc:
                            fail = False#self.drop # drop_srv().dropped # Check if dropped - end of episode
                            c = 0
                            while self.drop:
                                if c == 3:
                                    fail = True
                                    break
                                c += 1
                                rate.sleep()
                        else:
                            # End episode if overload or angle limits reached
                            rospy.logerr('[collect_data] Failed to move gripper. Episode declared failed.')
                            fail = True
                            recorder_srv()

                        self.texp.add(rospy.get_time()-T, state, action, next_state, not suc or fail)
                        state = np.copy(next_state)

                        if not suc or fail:
                            Done = True
                            break

                        if not self.first and episode_cur > episode_max:
                            recorder_srv()
                            print('[collect_data] Simulated drop.')
                            break
                        
                        rate.sleep()
                
                    self.trigger = False
                    print('[collect_data] Finished running episode %d with total number of collected points: %d' % (self.num_episodes, self.texp.getSize()))
                    print('[collect_data] Waiting for next episode initialization...')
                    rospy.sleep(5.0)

                    self.slow_open()
                    self.recorderSave_srv()
                    open_srv()
                    print('[collect_data] Position object and press key...')
                    raw_input()
                    self.trigger = True

                    # if self.num_episodes > 0 and not (self.num_episodes % 5):
                    #     open_srv()
                    #     # self.texp.save()
                    #     self.recorderSave_srv()
                    #     if (self.num_episodes % 20 == 0):
                    #         print('[collect_data] Cooling down.')
                    #         rospy.sleep(120)


    def callbackGripperStatus(self, msg):
        self.gripper_closed = msg.data == "closed"

    def callbackTrigger(self, msg):
        self.arm_status = msg.data
        if not self.trigger and self.arm_status == 'finished':
            self.trigger = True

    def callbackManualTrigger(self, msg):
        self.global_trigger = not self.global_trigger

    def callbackObjectDrop(self, msg):
        self.drop = msg.data

    def slow_open(self):
        print "Opening slowly."
        for _ in range(30):
            self.move_srv(np.array([-6.,-6.]))
            rospy.sleep(0.1)

    def choose_action(self):
        if self.discrete_actions:
            a = self.A[np.random.randint(self.A.shape[0])]
            if np.random.uniform(0,1,1) > 0.85:
                if np.random.uniform(0,1,1) > 0.5:
                    a = self.A[0]
                else:
                    a = self.A[1]
            elif np.random.uniform(0,1,1) > 0.85:
                if np.random.uniform(0,1,1) > 0.5:
                    a = self.A[2]
                else:
                    a = self.A[3]

            n = np.random.randint(60)
            if self.collect_mode == 'plan':
                if self.first:
                    n = np.random.randint(200)
                else:
                    n = np.random.randint(12, 50)
                    a = self.A[np.random.randint(self.A.shape[0])]
                    print "Running " + str(n) + " times action " + str(a) + " ..."
            else:
                if np.all(a == self.A[0]) or np.all(a == self.A[1]):
                    n = np.random.randint(50)
                elif np.random.uniform() > 0.7:
                    n = np.random.randint(150)
                else:
                    n = np.random.randint(85)
            return a, n
        else:
            a = np.random.uniform(-1.,1.,2)
            if np.random.uniform(0,1,1) > 0.6:
                if np.random.uniform(0,1,1) > 0.5:
                    a[0] = np.random.uniform(-1.,-0.8,1)
                    a[1] = np.random.uniform(-1.,-0.8,1)
                else:
                    a[0] = np.random.uniform(0.8,1.,1)
                    a[1] = np.random.uniform(0.8,1.,1)

            n = np.random.randint(60)
            if self.collect_mode == 'plan':
                if self.first:
                    n = np.random.randint(200)
                else:
                    n = np.random.randint(15, 80)
                    a = np.random.uniform(-1.,1.,2)
                    print "Running " + str(n) + " times action " + str(a) + " ..."
            else:
                if np.all(a == self.A[0]) or np.all(a == self.A[1]):
                    n = np.random.randint(50)
                elif np.random.uniform() > 0.7:
                    n = np.random.randint(250)
                else:
                    n = np.random.randint(100)

            return a, n

    def callbackDesiredAction(self, msg):
        self.desired_action = msg.data

    def callbackSave(self, msg):
        # self.texp.save()
        self.recorderSave_srv()

if __name__ == '__main__':
    
    try:
        collect_data()
    except rospy.ROSInterruptException:
        pass
