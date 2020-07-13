#!/usr/bin/env python

import rospy,sys
import numpy as np
import time, sys
import random
from std_msgs.msg import String, Float32MultiArray, Bool
from std_srvs.srv import Empty, EmptyResponse
from hand_control.srv import observation, IsDropped, TargetAngles, RegraspObject, close
from zs_transition_experience_v2 import *
import glob
from bowen_pose_estimate.srv import recordHandPose

class collect_data():
    discrete_actions = True # Discrete or continuous actions
    gripper_closed = False
    trigger = True # Enable collection
    arm_status = ' '
    global_trigger = True
    recorder_running=True

    num_episodes = 0
    episode_length = 1000000 # !!!
    desired_action = np.array([0.,0.])
    A = np.array([[1.0,1.0],[-1.,-1.],[-1.,1.],[1.,-1.],[1.5,0.],[-1.5,0.],[0.,-1.5],[0.,1.5]])
    B = np.array([[1.0,1.0],[-1.,1.],[1.,-1.],[1.5,0.],[0.,1.5]])

    def __init__(self):
        rospy.init_node('zs_collect_data_v2', anonymous=True)
        if rospy.has_param('~object_name'):
            self.collect_idx=rospy.get_param('~collect_idx')
            Obj=rospy.get_param('~object_name')
            ver=rospy.get_param('~version')
            col=rospy.get_param('~color')
        self.texp = zs_transition_experience_v2(Object = Obj, version=ver, color=col,Load = False, discrete = self.discrete_actions, postfix = 'bu')
        
        rospy.Subscriber('/gripper/gripper_status', String, self.callbackGripperStatus)
        rospy.Subscriber('/recorder_running',Bool,self.callbackRecorderRunning)
        pub_gripper_action = rospy.Publisher('/collect/action', Float32MultiArray, queue_size=10)
        rospy.Service('/collect/save', Empty, self.callbackSave)
        obs_srv = rospy.ServiceProxy('/observation', observation)
        drop_srv = rospy.ServiceProxy('/IsObjDropped', IsDropped)
        self.move_srv = rospy.ServiceProxy('/MoveGripper', TargetAngles)
        self.recorderSave_srv = rospy.ServiceProxy('/actor/save', Empty)
        recorder_srv = rospy.ServiceProxy('/actor/trigger', Empty)

        self.collect_mode = 'plan' # 'manual' or 'auto' or 'plan'
        
        if self.collect_mode == 'manual':
            rospy.Subscriber('/keyboard/desired_action', Float32MultiArray, self.callbackDesiredAction)
            ResetKeyboard_srv = rospy.ServiceProxy('/ResetKeyboard', Empty)

        
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
        self.first = False
        while not rospy.is_shutdown():

            if self.global_trigger:

                if self.collect_mode != 'manual':
                    if np.random.uniform() > 1.0: #0.5
                        self.collect_mode = 'plan'
                        self.first = True
                    else:
                        self.collect_mode = 'auto'

                if self.trigger:
                    if 1:#drop_srv().dropped:
                        print "Closing"
                        close_srv()
                        time.sleep(1.0)

                if self.trigger:# self.arm_status != 'moving' and self.trigger:
                    print('[collect_data] Verifying grasp...')
                    if drop_srv().dropped: # Check if really grasped
                        self.trigger = False
                        print('[collect_data] Grasp failed. Restarting')
                        self.slow_open()
                        continue

                    print('[collect_data] Starting episode %d...' % self.num_episodes)
                    self.num_episodes += 1

                    if self.collect_mode == 'plan':
                        if np.random.uniform() > 0.5:
                            Af = np.tile(np.array([-1.,1.]), (np.random.randint(50,130), 1))
                        else:
                            Af = np.tile(np.array([1.,-1.]), (np.random.randint(50,130), 1))
                        print('[collect_data] Rolling out shooting with %d steps.'%Af.shape[0])
                    
                    # Start episode
                    recorder_srv()
                    n = 0
                    action = np.array([0.,0.])
                    T = rospy.get_time()
                    Done=False
                    state = np.array(obs_srv().state)
                    count_fail=0
                    self.init=True

                    for ep_step in range(self.episode_length):
                        if self.collect_mode == 'plan' and Af.shape[0] == ep_step and self.first: # Finished planned path and now applying random actions
                            # self.collect_mode = 'auto'
                            self.first = False
                            n = 0
                            print('[collect_data] Running random actions...')
                            episode_cur = 0
                            episode_max = np.random.randint(200)
                            #episode_max = 1
                            #print('[collect_data] Rolling out random actions with %d steps.'%episode_max)
                        
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
                        if self.collect_mode == 'plan' and not self.first:
                            episode_cur += 1
                        
                        msg.data = action
                        pub_gripper_action.publish(msg)
                        suc = self.move_srv(action).success
                        n -= 1

                        # Get observation
                        next_state = np.array(obs_srv().state)

                        if suc:
                            if next_state[-1] or next_state[-2] or next_state[-3]:
                                count_fail+=1
                                if count_fail>=3:
                                    Done=True
                                    if self.recorder_running==True:
                                        recorder_srv()
                            else:
                                count_fail=0
                        else:
                            # End episode if overload or angle limits reached
                            rospy.logerr('[collect_data] Failed to move gripper. Episode declared failed.')
                            Done = True
                            if self.recorder_running==True:
                                recorder_srv()

                        self.texp.add(self.collect_idx, rospy.get_time()-T, state, action, next_state, Done)
                        state = np.copy(next_state)

                        if Done:
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


    def callbackGripperStatus(self, msg):
        self.gripper_closed = msg.data == "closed"

    def callbackRecorderRunning(self,msg):
        self.recorder_running=msg.data

    def slow_open(self):
        print "Opening slowly."
        for _ in range(30):
            self.move_srv(np.array([-6.,-6.]))
            rospy.sleep(0.1)

    def choose_action(self):
        if self.discrete_actions:
            if self.init==False:
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
            else:
                a=self.B[np.random.randint(self.B.shape[0])]
                self.init=False

            n = np.random.randint(60)
            if self.collect_mode == 'plan':
                if self.first:
                    n = np.random.randint(200)
                else:
                    n = np.random.randint(15, 80)
                    a = self.A[np.random.randint(self.A.shape[0])]
                    print "Running " + str(n) + " times action " + str(a) + " ..."
            #else:
            #    if np.all(a == self.A[0]) or np.all(a == self.A[1]):
            #        n = np.random.randint(50)
            #    elif np.random.uniform() > 0.7:
            #        n = np.random.randint(150)
            #    else:
            #        n = np.random.randint(85)
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
