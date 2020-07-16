#!/usr/bin/env python

import rospy
from std_msgs.msg import String, Float32MultiArray, Bool
from std_srvs.srv import Empty, EmptyResponse, SetBool
from rollout_t42.srv import rolloutReq, rolloutReqFile, plotReq, observation, IsDropped, TargetAngles, gets
from hand_control.srv import RegraspObject, close
import numpy as np
import matplotlib.pyplot as plt
import pickle, time

class rolloutPublisher():
    fail = False
    #one_fail=False
    S = A = T = []

    def __init__(self):
        rospy.init_node('rollout_t42', anonymous=True)
        if rospy.has_param('~ubuntu_cali_path'):
            self.ubuntu_cali_path=rospy.get_param('~ubuntu_cali_path')

        rospy.Service('/rollout/rollout', rolloutReq, self.CallbackRollout)
        rospy.Service('/rollout/rollout_v2', rolloutReq, self.CallbackRolloutV2)
        rospy.Service('/rollout/rollout_from_file', rolloutReqFile, self.CallbackRolloutFile)

        self.action_pub = rospy.Publisher('/rollout/action', Float32MultiArray, queue_size = 10)

        rospy.Subscriber('/rollout/fail', Bool, self.callbacFail)
        #rospy.Subscriber('/rollout/one_fail', Bool, self.callbacOneFail)

        self.rollout_actor_srv = rospy.ServiceProxy('/rollout/run_actor', SetBool)
        self.drop_srv = rospy.ServiceProxy('/IsObjDropped', IsDropped)
        self.move_srv = rospy.ServiceProxy('/MoveGripper', TargetAngles)
        self.obs_srv = rospy.ServiceProxy('/observation', observation)
        self.open_srv = rospy.ServiceProxy('/OpenGripper', Empty) 
        self.close_srv = rospy.ServiceProxy('/CloseGripper', close) 

        self.state_dim = 4
        self.action_dim = 2
        self.stepSize = 1
        print('[rollout] Ready to rollout...')

        self.rate = rospy.Rate(10) # 10hz
        rospy.spin()

    def run_rollout(self, A):
        finished=False
        while not finished:
            self.rollout_transition = []
            self.fail = False  

            print("[rollout_action_publisher] Place object and press key...")
            raw_input()
            self.close_srv()
            time.sleep(1.0)
            print('[rollout] Verifying grasp...')
            if self.drop_srv().dropped: # Check if really grasped
                print('[rollout] Grasp failed. Restarting')
                self.slow_open()
                self.open_srv()
                continue

            msg = Float32MultiArray()  

            state = np.array(self.obs_srv().state)
            self.S = []
            self.S.append(np.copy(state))  

            print("[rollout_action_publisher] Rolling-out actions...")
            
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
                n -= 1
                msg.data = action
                self.action_pub.publish(msg)

                if i==1 and n==self.stepSize-1:
                    self.rollout_actor_srv(True)

                print(i, action, A.shape[0])
                next_state=np.array(self.obs_srv().state)
                self.S.append(np.copy(next_state))
                self.rollout_transition.append([state,action,next_state,self.fail])
                state=np.copy(next_state)

                if self.fail: # not suc or detection/drop fail:
                    print("[rollout] Move or Detection or Drop Fail.")
                    success = False
                    self.rollout_actor_srv(False)
                    break

                if i == A.shape[0] and n == 0:
                    print("[rollout] Complete.")
                    success = True
                    self.rollout_actor_srv(False)
                    break
                self.rate.sleep()
            rospy.sleep(1.)
            self.slow_open()
            self.open_srv()
            finished=True

        return success

    def run_rollout_v2(self, A):
        finished=False
        while not finished:
            self.rollout_transition = []
            self.fail = False  

            print("[rollout_action_publisher] Place object and press key...")
            raw_input()
            self.close_srv()
            time.sleep(1.0)
            print('[rollout] Verifying grasp...')
            if self.drop_srv().dropped: # Check if really grasped
                print('[rollout] Grasp failed. Restarting')
                self.slow_open()
                self.open_srv()
                continue

            state = np.array(self.obs_srv().state)
            self.S = []
            self.S.append(np.copy(state))  

            print("[rollout_action_publisher] Rolling-out actions...")
            
            # Publish episode actions
            self.running = True
            success = True
            n = 0
            i = 0
            count_fail=0
            while self.running:
                if n == 0:
                    action = A[i,:]
                    i += 1
                    n = self.stepSize
                n -= 1
                print(i, action, A.shape[0])

                if i % 4 == 1:
                    suc=self.move_srv(self.action).success

                next_state=np.array(self.obs_srv().state)
                self.S.append(np.copy(next_state))
                self.rollout_transition.append([state,action,next_state, not suc or ((next_state[-1] or next_state[-2] or next_state[-3]) and count_fail>=11)])
                state=np.copy(next_state)
                if not suc:
                    print("[rollout] Move Fail.")
                    success = False
                    count_fail=0
                    break
                else:
                    if next_state[-1] or next_state[-2] or next_state[-3]:
                        count_fail+=1
                        if count_fail>=12:
                            print("[rollout] Detection or Drop Fail.")
                            success = False
                            count_fail=0
                            break
                    else:
                        count_fail=0
                if i == A.shape[0] and n == 0:
                    print("[rollout] Complete.")
                    success = True
                    break
                self.rate.sleep()
            rospy.sleep(1.)
            self.slow_open()
            self.open_srv()
            finished=True

        return success

    def slow_open(self):
        print "Opening slowly."
        for _ in range(30):
            self.move_srv(np.array([-6.,-6.]))
            rospy.sleep(0.1)

    def callbacFail(self, msg):
        self.fail = msg.data
        
    #def callbacOneFail(self, msg):
    #    self.one_fail = msg.data

    def CallbackRollout(self, req):
        print('[rollout_action_publisher] Rollout request received.')
        actions = np.array(req.actions).reshape(-1, self.action_dim)
        success = self.run_rollout(actions)
        states = np.array(self.S)

        return {'states': states.reshape((-1,)), 'success' : success}

    def CallbackRolloutV2(self, req):
        print('[rollout_action_publisher] Rollout request received.')
        actions = np.array(req.actions).reshape(-1, self.action_dim)
        success = self.run_rollout_v2(actions)
        states = np.array(self.S)

        return {'states': states.reshape((-1,)), 'success' : success}

if __name__ == '__main__':
    try:
        rolloutPublisher()
    except rospy.ROSInterruptException:
        pass