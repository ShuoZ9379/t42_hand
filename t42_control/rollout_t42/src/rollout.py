#!/usr/bin/env python

import rospy
from std_msgs.msg import String, Float32MultiArray, Bool
from std_srvs.srv import Empty, EmptyResponse, SetBool
from rollout_t42.srv import rolloutReq, rolloutReqFile, plotReq, observation, IsDropped, TargetAngles, gets
from hand_control.srv import RegraspObject, close
import numpy as np
import matplotlib.pyplot as plt
import pickle

class rolloutPublisher():

    states = []
    plot_num = 0
    arm_status = ' '
    trigger = False # Enable collection
    drop = True
    suc = True
    drop_counter = 0
    fail = False
    actor_running = False

    def __init__(self):
        rospy.init_node('rollout_t42', anonymous=True)

        rospy.Service('/rollout/rollout', rolloutReq, self.CallbackRollout)
        rospy.Service('/rollout/rollout_from_file', rolloutReqFile, self.CallbackRolloutFile)
        rospy.Service('/rollout/run_trigger', SetBool, self.callbackStop)
        self.record_srv = rospy.ServiceProxy('/rollout/record_trigger', SetBool)
        self.action_pub = rospy.Publisher('/rollout/action', Float32MultiArray, queue_size = 10)
        # rospy.Subscriber('/rollout/move_success', Bool, self.callbackSuccess)
        rospy.Subscriber('/rollout/fail', Bool, self.callbacFail)
        rospy.Subscriber('/rollout_actor/runnning', Bool, self.callbackActorRunning)

        self.rollout_actor_srv = rospy.ServiceProxy('/rollout/run_actor', SetBool)

        self.arm_reset_srv = rospy.ServiceProxy('/RegraspObject', RegraspObject)
        rospy.Subscriber('/ObjectIsReset', String, self.callbackTrigger)
        # rospy.Subscriber('/cylinder_drop', Bool, self.callbackObjectDrop)
        self.drop_srv = rospy.ServiceProxy('/IsObjDropped', IsDropped)
        self.move_srv = rospy.ServiceProxy('/MoveGripper', TargetAngles)
        self.obs_srv = rospy.ServiceProxy('/observation', observation)
        self.gets_srv = rospy.ServiceProxy('/rollout/get_states', gets)
        self.open_srv = rospy.ServiceProxy('/OpenGripper', Empty) 
        self.close_srv = rospy.ServiceProxy('/CloseGripper', close) 

        self.state_dim = 4
        self.action_dim = 2
        self.stepSize = 1

        print('[rollout] Ready to rollout...')

        self.rate = rospy.Rate(10) # 15hz
        rospy.spin()

    def ResetArm(self):
        while 1:
            if not self.trigger and self.arm_status == 'waiting':
                print('[rollout_action_publisher] Waiting for arm to grasp object...')
                self.arm_reset_srv()
                rospy.sleep(1.0)
            self.rate.sleep()
            if self.arm_status != 'moving' and self.trigger:
                self.rate.sleep()
                if self.drop_srv().dropped: # Check if really grasped
                    self.trigger = False
                    print('[rollout_action_publisher] Grasp failed. Restarting')
                    continue
                else:
                    break
        self.trigger = False

    def run_rollout(self, A):
        self.rollout_transition = []
        self.trigger = False
        # self.ResetArm()  
        self.fail = False  

        print("[rollout_action_publisher] Place object and press key...")
        raw_input()
        self.close_srv()

        msg = Float32MultiArray()  

        state = np.array(self.obs_srv().state)
        self.S = []
        self.S.append(np.copy(state))  

        print("[rollout_action_publisher] Rolling-out actions...")
        self.rollout_actor_srv(True)
        self.record_srv(True)
        
        # Publish episode actions
        self.running = True
        success = True
        n = 0
        i = 0
        while self.running:

            if n == 0:
                action = A[i,:]
                i += 1
                n = self.stepSize
                # print('[rollout] Applying action (' + str(action) + ') ' + str(i) + ' out of ' + str(A.shape[0]) + '.' )

            msg.data = action
            self.action_pub.publish(msg)
            n -= 1
            print(i, action, A.shape[0])

            if self.fail: # not suc or fail:
                success = False
                print("[rollout] Drop Fail.")
                self.record_srv(False)
                self.rollout_actor_srv(False)
                break

            if i == A.shape[0] and n == 0:
                print("[rollout] Complete.")
                success = True
                self.record_srv(False)
                self.rollout_actor_srv(False)
                # self.slow_open()
                break

            self.rate.sleep()

        rospy.sleep(1.)
        self.open_srv()

        return success

    def slow_open(self):
        print "Opening slowly."
        for _ in range(30):
            self.move_srv(np.array([-6.,-6.]))
            rospy.sleep(0.1)

    def callbacFail(self, msg):
        self.fail = msg.data

    def callbackSuccess(self, msg):
        self.suc = msg.data

    def callbackTrigger(self, msg):
        self.arm_status = msg.data
        if not self.trigger and self.arm_status == 'finished':
            self.trigger = True

    def callbackStop(self, msg):
        self.running = msg.data

        return {'success': True, 'message': ''}

    def callbackActorRunning(self, msg):
        self.actor_running = msg.data

    def CallbackRollout(self, req):

        print('[rollout_action_publisher] Rollout request received.')
        
        actions = np.array(req.actions).reshape(-1, self.action_dim)
        success = self.run_rollout(actions)

        # states = np.array(self.S)
        # states[:,:2] *= 1000.

        SA = self.gets_srv()
        states = np.array(SA.states)
        # actions = SA.actions 

        return {'states': states.reshape((-1,)), 'success' : success}

    def CallbackRolloutFile(self, req):

        file_name = req.file

        actions = np.loadtxt(file_name, delimiter=',', dtype=float)[:,:2]
        success = self.run_rollout(actions)

        states = np.array(self.S)

        return {'states': states.reshape((-1,)), 'success' : True}

if __name__ == '__main__':
    try:
        rolloutPublisher()
    except rospy.ROSInterruptException:
        pass