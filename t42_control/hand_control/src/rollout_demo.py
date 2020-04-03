#!/usr/bin/env python

import rospy
from std_msgs.msg import String, Float32MultiArray, Bool
from std_srvs.srv import Empty, EmptyResponse
from rollout_node.srv import rolloutReq, rolloutReqFile, plotReq, observation, IsDropped, TargetAngles
from hand_control.srv import RegraspObject, close
import numpy as np
import matplotlib.pyplot as plt
import pickle


class rollout_demo():

    states = []
    plot_num = 0
    arm_status = ' '
    trigger = False # Enable demo
    resetf = False
    gripper_pos = np.array([0., 0.])
    A = np.array([[-1.,1.], [1.,-1.], [-1.,-1.]])*2.
    drop = True

    def __init__(self):
        rospy.init_node('rollout_node', anonymous=True)

        rospy.Service('/run_demo', Empty, self.callbackTrigger)
        rospy.Subscriber('/ObjectIsReset', String, self.callbackArm)
        rospy.Subscriber('/cylinder_drop', Bool, self.callbackObjectDrop)

        self.obs_srv = rospy.ServiceProxy('/observation', observation)
        self.drop_srv = rospy.ServiceProxy('/IsObjDropped', IsDropped)
        self.move_srv = rospy.ServiceProxy('/MoveGripper', TargetAngles)
        self.arm_reset_srv = rospy.ServiceProxy('/RegraspObject', RegraspObject)

        self.close_srv = rospy.ServiceProxy('/CloseGripper', close)
        self.open_srv = rospy.ServiceProxy('/OpenGripper', Empty) 

        rospy.Subscriber('/gripper/pos', Float32MultiArray, self.callbackGripperPos)

        self.state_dim = 4
        self.action_dim = 2
        self.stepSize = 1 

        self.rate = rospy.Rate(15) # 15hz
        ia = 1
        count = 0
        drop_count = []
        while not rospy.is_shutdown():

            if self.trigger:
                self.ResetArm()
                # self.resetf = True
                # self.trigger = False

            if self.resetf:
                print self.A[ia], count

                suc = self.move_srv(self.A[ia]).success
                self.rate.sleep()

                if self.gripper_pos[1] > 0.51:
                    ia = 1
                if self.gripper_pos[1] < 0.33:
                    ia = 0
                # if count > 200:
                #     ia = 2

                if not suc:# or self.drop:
                    if 0:#self.drop:
                        drop_count.append(count)
                        print "Drop!!! Avg.: ", np.mean(drop_count)
                        raw_input()
                    self.trigger = True
                    self.resetf = False 
                    count = 0
                    ia = 1

                count += 1

            self.rate.sleep()

    def ResetArm(self):
        while 1:
            if self.trigger and self.arm_status == 'waiting':
                print('[rollout_action_publisher] Waiting for arm to grasp object...')
                self.arm_reset_srv()
                rospy.sleep(1.0)
            self.rate.sleep()
            if self.arm_status != 'moving' and self.trigger:
                self.rate.sleep()
                if 0:#self.drop: # Check if really grasped
                    self.trigger = True
                    print('[rollout_action_publisher] Grasp failed. Restarting')
                    continue
                else:
                    break

        self.resetf = True
        self.trigger = False
        

    def callbackGripperPos(self, msg):
        self.gripper_pos = np.array(msg.data)

    def run_rollout(self, A):
        self.rollout_transition = []
        self.trigger = False
        # self.ResetArm()            

        print("[rollout] Rolling-out...")
        
        # Start episode
        success = True
        S = []
        for i in range(A.shape[0]):
            # Get observation and choose action
            state = np.array(self.obs_srv().state)
            action = A[i,:]
            print action

            S.append(state)
            
            suc = True
            state_tmp = state
            for _ in range(self.stepSize):
                suct = self.move_srv(action).success

                next_state = np.array(self.obs_srv().state)
                self.rollout_transition += [(state_tmp, action, next_state, not suct or self.drop_srv().dropped)]
                state_tmp = next_state

                self.rate.sleep()
                if not suct:
                    suc = False

            # Get observation
            next_state = np.array(self.obs_srv().state)

            if suc:
                fail = self.drop # Check if dropped - end of episode
            else:
                # End episode if overload or angle limits reached
                rospy.logerr('[rollout] Failed to move gripper. Episode declared failed.')
                fail = True

            state = np.copy(next_state)

            if not suc or fail:
                print("[rollout] Fail")
                S.append(state)
                success = False
                break

        file_pi = open('/home/pracsys/catkin_ws/src/hand_control/plans/rollout_output.pkl', 'wb')
        pickle.dump(self.rollout_transition, file_pi)
        file_pi.close()

        print("[rollout] Rollout done.")

        return np.array(S), success

    def callbackArm(self, msg):
        self.arm_status = msg.data

    def callbackTrigger(self, msg):
        self.trigger = not self.trigger

    def callbackObjectDrop(self, msg):
        self.drop = msg.data

    def CallbackRollout(self, req):
        
        actions = np.array(req.actions).reshape(-1, self.action_dim)
        success = True
        self.states, success = self.run_rollout(actions)

        return {'states': self.states.reshape((-1,)), 'success' : success}

    def Plot(self, req):
        planned = np.array(req.states).reshape(-1, self.state_dim)
        plt.clf()
        plt.plot(self.states[:,0], self.states[:,1],'b', label='Rolled-out path')
        plt.plot(planned[:,0], planned[:,1],'r', label='Planned path')
        # plt.legend()
        if (req.filename):
            plt.savefig(req.filename, bbox_inches='tight')
        else:
            plt.show()

        return EmptyResponse()


if __name__ == '__main__':
    try:
        rollout_demo()
    except rospy.ROSInterruptException:
        pass