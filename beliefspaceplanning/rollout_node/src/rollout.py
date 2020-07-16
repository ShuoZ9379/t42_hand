#!/usr/bin/env python

import rospy
from std_srvs.srv import Empty, EmptyResponse
from std_msgs.msg import Bool, String, Float32MultiArray
from rollout_node.srv import rolloutReq, rolloutReqFile, plotReq, observation, IsDropped, TargetAngles, reset, StepOnlineReq, CheckOnlineStatus
import numpy as np
import matplotlib.pyplot as plt
import pickle
from rollout_node.srv import gets

import sys
sys.path.insert(0, '/home/szhang/catkin_ws/src/beliefspaceplanning/gpup_gp_node/src/')
import var

class rollout():

    states = []
    actions = []
    plot_num = 0
    drop = True
    obj_pos = np.array([0., 0.])

    def __init__(self):
        rospy.init_node('rollout_node', anonymous=True)
        self.normal_goals=([[-66, 80],[-41, 100], [-62, 96], [-49, 86], [-55, 92],[59, 78],[31, 102],[60, 100],[52, 95],[-78, 67],[31, 125],[-26, 125],[0, 107],[3, 130],[-48, 114],[69, 78]])
        self.horseshoe_goals=np.array([[21, 123]])

        rospy.Service('/rollout/rollout', rolloutReq, self.CallbackRollout)
        rospy.Service('/rollout/rollout_from_file', rolloutReqFile, self.CallbackRolloutFile)
        rospy.Service('/rollout/plot', plotReq, self.Plot)
        rospy.Subscriber('/hand_control/cylinder_drop', Bool, self.callbackDrop)
        rospy.Subscriber('/hand_control/gripper_status', String, self.callbackGripperStatus)
        rospy.Subscriber('/hand/obj_pos', Float32MultiArray, self.callbackObj)
        self.action_pub = rospy.Publisher('/rollout/gripper_action', Float32MultiArray, queue_size = 10)

        self.obs_srv = rospy.ServiceProxy('/hand_control/observation', observation)
        self.drop_srv = rospy.ServiceProxy('/hand_control/IsObjDropped', IsDropped)
        self.move_srv = rospy.ServiceProxy('/hand_control/MoveGripper', TargetAngles)
        self.reset_srv = rospy.ServiceProxy('/hand_control/ResetGripper', reset)


        rospy.Service('/rollout/ResetOnline', reset, self.CallbackResetOnline)
        rospy.Service('/rollout/StepOnline', StepOnlinetReq, self.CallbackStepOnline)
        self.move_online_srv = rospy.ServiceProxy('/hand_control/MoveGripperOnline', TargetAngles)
        self.check_srv=rospy.ServiceProxy('/hand_control/CheckStatusOnline', CheckOnlineStatus)


        self.state_dim = var.state_dim_
        self.action_dim = var.state_action_dim_-var.state_dim_
        self.stepSize = var.stepSize_ # !!!!!!!!!!!!!!!!!!!!!!!!!!

        print("[rollout] Ready to rollout...")

        self.rate = rospy.Rate(2) 
        # while not rospy.is_shutdown():
        rospy.spin()

    def run_rollout(self, A, obs_idx, obs_size=np.array([0.75]),goal_idx=np.array([8]),big_goal_radius=np.array([8])):
        self.rollout_transition = []

        # Reset gripper
        while 1:
            self.reset_srv(obs_idx[0],obs_size[0],goal_idx[0],big_goal_radius[0])
            while not self.gripper_closed:
                print('caocaocao')
                self.rate.sleep()

            print self.obj_pos, np.abs(self.obj_pos[0]-0.03) < 0.015 , np.abs(self.obj_pos[1]-118.16) < 0.1
            if np.abs(self.obj_pos[0]-0.03) < 0.015 and np.abs(self.obj_pos[1]-118.16) < 0.1:
                break
        
        print("[rollout] Rolling-out...")

        msg = Float32MultiArray()

        # Start episode
        success = True
        state = np.array(self.obs_srv().state)
        S = []
        S.append(np.copy(state))
        stepSize = var.stepSize_
        n = 0
        i = 0
        while 1:
            if n == 0:
                action = A[i,:]
                i += 1
                n = stepSize
            
            msg.data = action
            self.action_pub.publish(msg)
            suc = self.move_srv(np.concatenate((action,obs_size))).success
            n -= 1
            
            next_state = np.array(self.obs_srv().state)

            if suc:
                fail = self.drop
            else:
                # End episode if overload or angle limits reached
                rospy.logerr('[rollout] Failed to move gripper. Episode declared failed.')
                fail = True

            if n == 0 or (not suc or fail):
                # print len(S), len(self.rollout_transition)
                S.append(np.copy(next_state))
                self.rollout_transition += [(state, action, next_state, not suc or fail)]

            state = np.copy(next_state)

            if not suc or fail:
                print("[rollout] Fail.")
                success = False
                break
            if i == A.shape[0] and n == 0:
                print("[rollout] Complete.")
                success = True
                break

            self.rate.sleep()

        print("[rollout] Rollout done.")

        self.states = np.array(S).reshape((-1,))
        return success
    

    def callbackGripperStatus(self, msg):
        self.gripper_closed = msg.data == "closed"

    def callbackDrop(self, msg):
        self.drop = msg.data

    def callbackObj(self, msg):
        Obj_pos = np.array(msg.data)
        self.obj_pos = Obj_pos[:2] * 1000

    def CallbackResetOnline(self, req):
        self.obs_idx = req.obs_idx
        self.obs_size = req.obs_size
        self.goal_idx = req.goal_idx
        self.big_goal_radius = req.big_goal_radius
        if self.obs_idx==14:
            self.goal=self.horseshoe_goals[self.goal_idx]
            self.obs_size=0.5
            self.big_goal_radius=5
        elif self.obs_idx==20:
            self.goal=self.normal_goals[self.goal_idx]
            self.obs_size=0.75
            self.big_goal_radius=8

        while 1:
            st=self.reset_srv(self.obs_idx,self.obs_size,self.goal_idx,self.big_goal_radius).states
            while not self.gripper_closed:
                print('caocaocao')
                self.rate.sleep()
            print self.obj_pos, np.abs(self.obj_pos[0]-0.03) < 0.015 , np.abs(self.obj_pos[1]-118.16) < 0.1
            if np.abs(self.obj_pos[0]-0.03) < 0.015 and np.abs(self.obj_pos[1]-118.16) < 0.1:
                break
        return {'states': np.array(self.obs_srv().state)}

    def CallbackStepOnline(self, req):
        S,rwd_history,Done_history=[]
        suc_history,object_grasped_history,no_hit_obs_history,goal_reached_history=[],[],[],[]
        actions_nom = np.array(req.actions).reshape(-1, self.action_dim)
        actions_nom = np.clip(actions_nom,np.array([-1,-1]),np.array([1,1]))
        for i in range(actions_nom.shape[0]):
            self.move_online_srv(actions_nom[i])
            self.rate.sleep()
            next_state = np.array(self.obs_srv().state)
            S.append(next_state)
            res=self.check_srv()
            suc=res.success
            suc_history.append(int(suc))
            object_grasped=res.grasped
            object_grasped_history.append(int(object_grasped))
            no_hit_obs=res.avoid_obs
            no_hit_obs_history.append(int(no_hit_obs))
            goal_reached=res.goal_reach
            goal_reached_history.append(int(goal_reached))
            failed = not (suc and object_grasped and no_hit_obs)
            Done=goal_reached or failed
            Done_history.append(int(Done))
            rwd=-np.linalg.norm(self.goal-next_state[:2])-np.square(actions_nom[i]).sum()
            #if failed:
            if not no_hit_obs:
                rwd-=1e6
            rwd_history.append(rwd)
            if Done:
                print('Episode Finished')
                break
        return {'states': next_state, 'states_history': np.array(S).reshape((-1,)), 'success': suc, 'success_history': np.array(suc_history), 'grasped': object_grasped, 'grasped_history': np.array(object_grasped_history), 'avoid_obs': no_hit_obs, 'avoid_obs_history': np.array(no_hit_obs_history), 'goal_reach': goal_reached, 'goal_reach_history': np.array(goal_reached_history), 'reward': rwd, 'reward_history': np.array(rwd_history), 'done': Done, 'done_history': np.array(Done_history)}

    def CallbackRollout(self, req):
        obs_idx = np.array(req.obs_idx)
        obs_size = np.array(req.obs_size)
        goal_idx = np.array(req.goal_idx)
        big_goal_radius = np.array(req.big_goal_radius)
        success = self.run_rollout(actions_nom,obs_idx,obs_size,goal_idx,big_goal_radius)

        return {'states': self.states, 'actions_res': self.actions, 'success' : success}

    def CallbackRolloutFile(self, req):

        file_name = req.file

        actions = np.loadtxt(file_name, delimiter=',', dtype=float)[:,:2]
        success = True
        success = self.run_rollout(actions)

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
        rollout()
    except rospy.ROSInterruptException:
        pass