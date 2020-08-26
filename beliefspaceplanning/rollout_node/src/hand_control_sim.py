#!/usr/bin/python 

import rospy
import numpy as np 
from std_msgs.msg import Float64MultiArray, Float32MultiArray, String, Bool
from std_srvs.srv import Empty, EmptyResponse
from rollout_node.srv import TargetAngles, IsDropped, observation, MoveServos, CheckOnlineStatus, reset
import math
import pickle


state_form = 'pos_load' # 'pos_load' or 'pos_vel' or 'pos_load_vel' or 'pos_load_vel_joints' or 'pos_load_joints' or 'all'

class hand_control():

    finger_opening_position = np.array([0.005, 0.005])
    finger_closing_position = np.array([0.08, 0.08])
    finger_move_step_size = np.array([0.00015, 0.00015])
    closed_load = np.array(20.)

    gripper_pos = np.array([0., 0.])
    gripper_load = np.array([0., 0.])
    gripper_load_prev = np.array([0., 0.])
    joint_states = np.array([0., 0., 0., 0.])
    joint_velocities = np.array([0., 0., 0., 0.])
    lift_status = False # True - lift up, False - lift down
    object_grasped = False
    base_pos = [0.,0.]
    base_theta = 0
    obj_pos = [0.,0.]
    obj_vel = [0.,0.]
    D_load = np.array([0., 0.])
    R = []
    count = 1
    OBS = True

    gripper_status = 'open'

    move_servos_srv = 0.

    def __init__(self):
        rospy.init_node('hand_control_sim', anonymous=True)
        self.normal_goals=([[-66, 80],[-41, 100], [-62, 96], [-49, 86], [-55, 92],[59, 78],[31, 102],[60, 100],[52, 95],[-78, 67],[31, 125],[-26, 125],[0, 107],[3, 130],[-48, 114],[69, 78]])
        self.horseshoe_goals=np.array([[21, 123]])
        
        rospy.Subscriber('/gripper/pos', Float32MultiArray, self.callbackGripperPos)
        rospy.Subscriber('/gripper/load', Float32MultiArray, self.callbackGripperLoad)
        rospy.Subscriber('/gripper/lift_status', String, self.callbackLiftStatus)
        rospy.Subscriber('/hand/obj_pos', Float32MultiArray, self.callbackObj)
        rospy.Subscriber('/hand/obj_vel', Float32MultiArray, self.callbackObjVel)
        rospy.Subscriber('/hand/my_joint_states', Float32MultiArray, self.callbackJoints)
        rospy.Subscriber('/hand/my_joint_velocities', Float32MultiArray, self.callbackJointsVel)
        pub_gripper_status = rospy.Publisher('/hand_control/gripper_status', String, queue_size=10)
        pub_drop = rospy.Publisher('/hand_control/cylinder_drop', Bool, queue_size=10)

        rospy.Service('/hand_control/ResetGripper', reset, self.ResetGripper)
        rospy.Service('/hand_control/MoveGripper', TargetAngles, self.MoveGripper)
        rospy.Service('/hand_control/MoveGripperOnline', TargetAngles, self.MoveGripperOnline)
        rospy.Service('/hand_control/CheckStatusOnline', CheckOnlineStatus, self.CheckStatusOnline)
        rospy.Service('/hand_control/IsObjDropped', IsDropped, self.CheckDropped)
        rospy.Service('/hand_control/observation', observation, self.GetObservation)

        self.move_servos_srv = rospy.ServiceProxy('/MoveServos', MoveServos)
        self.move_lift_srv = rospy.ServiceProxy('/LiftHand', Empty)
        self.reset_srv = rospy.ServiceProxy('/gazebo/reset_world', Empty)
        
        rate = rospy.Rate(50)
        while not rospy.is_shutdown():
            pub_gripper_status.publish(self.gripper_status)
            pub_drop.publish(not self.object_grasped)

            self.D_load = np.copy(self.gripper_load) - np.copy(self.gripper_load_prev)
            self.gripper_load_prev = np.copy(self.gripper_load)

            # print(self.obj_pos)
            # rospy.spin()
            rate.sleep()

    def callbackGripperPos(self, msg):
        self.gripper_pos = np.array(msg.data)

    def callbackGripperLoad(self, msg):
        self.gripper_load = np.array(msg.data)

    def callbackJoints(self, msg):
        self.joint_states = np.array(msg.data)

    def callbackJointsVel(self, msg):
        self.joint_velocities = np.array(msg.data)

    def callbackLiftStatus(self, msg):
        self.lift_status = True if msg.data == 'up' else False

    def callbackObj(self, msg):
        Obj_pos = np.array(msg.data)

        # b = 0.5e-3 # For screwdriver scenario
        b = -0.1 # For regular cylinder scenario

        if Obj_pos[2] < b or (self.joint_states[0] > 1.5 and self.joint_states[2] > 1.5) or self.joint_states[0] > 2.5 or self.joint_states[2] > 2.5: 
            self.object_grasped = False
        else:
            self.object_grasped = True
        # self.object_grasped = True if abs(Obj_pos[2]) < 1e-2 else False

        self.obj_pos = Obj_pos[:2]*1000 # m to mm

    def callbackObjVel(self, msg):
        Obj_vel = np.array(msg.data)

        self.obj_vel = Obj_vel[:2]*1000 # m/s to mm/s

    def ResetGripper(self, msg):
        self.gripper_status = 'open'

        ratein = rospy.Rate(15)
        self.obs_idx=msg.obs_idx
        self.obs_size=msg.obs_size
        self.goal_idx=msg.goal_idx
        self.big_goal_radius=msg.big_goal_radius
        self.sparse=msg.sparse
        if self.obs_idx==14:
            self.goal=self.horseshoe_goals[self.goal_idx]
            self.obs_size=0.5
            self.big_goal_radius=5
        elif self.obs_idx==20:
            self.goal=self.normal_goals[self.goal_idx]
            self.obs_size=0.75
            self.big_goal_radius=8
        with open('/home/szhang/catkin_ws/src/beliefspaceplanning/rollout_node/set/obs_'+str(self.obs_idx)+'.pkl', 'r') as f: 
            self.Obs = pickle.load(f)

        print('[hand_control_sim] Resetting gripper...')
        while 1:
            # Open gripper
            self.moveGripper(self.finger_opening_position)
            rospy.sleep(.5)
            # Drop lift down
            while self.lift_status:
                self.move_lift_srv.call()
                rospy.sleep(.1)
                ratein.sleep()
            # Reset object
            rospy.sleep(.2)
            self.reset_srv.call()
            # Close gripper
            no = 0#np.random.random(2)*2. - 1.
            self.moveGripper(self.finger_closing_position+no/1000.)
            rospy.sleep(0.7)
            self.move_lift_srv.call()
            rospy.sleep(2.0)
            ratein.sleep()
            if self.object_grasped:# and self.lift_s, self.obj_veltatus:
                if self.wait2initialGrasp():
                    break
        
        self.gripper_status = 'closed'
        return {'states': np.concatenate((self.obj_pos, self.gripper_load), axis=0)}

    def wait2initialGrasp(self):
        # This function waits for grasp to be stable (static) in its initial pose

        print('[hand_control_sim] Waiting for hand to stabilize...')
        ratein = rospy.Rate(15)
        c = 0
        suc = False
        while c < 150:
            c += 1
            pos1 = self.obj_pos
            rospy.sleep(0.2)
            ratein.sleep()
            # print(pos1, self.obj_pos, np.linalg.norm(pos1 - self.obj_pos))
            if np.linalg.norm(pos1 - self.obj_pos) < 2e-2:
                suc = True
                print('[hand_control_sim] Hand stable.')
                break

        return suc

    def MoveGripper(self, msg):
        # This function should accept a vector of normalized incraments to the current angles: msg.angles = [dq1, dq2], where dq1 and dq2 can be equal to 0 (no move), 1,-1 (increase or decrease angles by finger_move_step_size)

        inc = np.array(msg.angles)[:2]
        obs_size = None
        if self.OBS:
            if np.array(msg.angles).shape[0]==3:
                obs_size = np.array(msg.angles)[-1]

        inc_angles = np.multiply(self.finger_move_step_size, inc)

        desired = self.gripper_pos + inc_angles

        suc = self.moveGripper(desired,obs_size=obs_size)

        return {'success': suc}
    
    def moveGripper(self, angles, obs_size=None):

        if angles[0] > 0.7 or angles[1] > 0.7 or angles[0] < 0.003 or angles[1] < 0.003:
            rospy.logerr('[hand_control_sim] Desired angles out of bounds.')
            return False

        if abs(self.gripper_load[0]) > 120 or abs(self.gripper_load[1]) > 120:
            rospy.logerr('[hand_control_sim] Pre-overload.')
            return False

        self.move_servos_srv.call(angles)

        if self.OBS:
            if obs_size is None:
                obs_size=self.Obs[0,2]
            for obs in self.Obs:
                if np.linalg.norm(self.obj_pos-obs[:2]) < obs_size:
                    print('[hand_control_sim] Collision.')
                    return False
        return True

    def MoveGripperOnline(self, msg):
        inc = np.array(msg.angles)[:2]
        inc_angles = np.multiply(self.finger_move_step_size, inc)
        desired = self.gripper_pos + inc_angles

        self.move_servos_srv.call(desired)
        return {'success': True}

    def CheckStatusOnline(self,msg):
        if self.gripper_pos[0] > 0.7 or self.gripper_pos[1] > 0.7 or self.gripper_pos[0] < 0.003 or self.gripper_pos[1] < 0.003:
            rospy.logerr('[hand_control_sim] Gripper angles out of bounds.')
            suc=False

        elif abs(self.gripper_load[0]) > 120 or abs(self.gripper_load[1]) > 120:
            rospy.logerr('[hand_control_sim] Gripper overload.')
            suc=False
        else:
            suc=True

        no_hit_obs=True
        if self.OBS:
            for obs in self.Obs:
                if np.linalg.norm(self.obj_pos-obs[:2]) < self.obs_size:
                    print('[hand_control_sim] Collision.')
                    no_hit_obs=False
                    break

        goal_reached=False
        if np.linalg.norm(self.obj_pos-self.goal) < self.big_goal_radius:
            print('[hand_control_sim] Goal Reached.')
            goal_reached=True
        return {'state': np.concatenate((self.obj_pos, self.gripper_load), axis=0) ,'success': suc, 'grasped': self.object_grasped, 'avoid_obs': no_hit_obs, 'goal_reach': goal_reached}


    def CheckDropped(self, msg):

        return {'dropped': not self.object_grasped}

    def GetObservation(self, msg):
        if state_form == 'all':   
            obs = np.concatenate((self.obj_pos, self.gripper_load, self.joint_states, self.joint_velocities), axis=0)
        elif state_form == 'pos_load':
            obs = np.concatenate((self.obj_pos, self.gripper_load), axis=0)
        elif state_form == 'pos_vel':   
            obs = np.concatenate((self.obj_pos, self.obj_vel), axis=0)
        elif state_form == 'pos_load_vel':   
            obs = np.concatenate((self.obj_pos, self.gripper_load, self.obj_vel, self.D_load), axis=0)
        elif state_form == 'pos_load_vel_joints':   
            obs = np.concatenate((self.obj_pos, self.gripper_load, self.obj_vel, self.joint_states), axis=0)
        elif state_form == 'pos_load_joints':   
            obs = np.concatenate((self.obj_pos, self.gripper_load, self.joint_states), axis=0)

        # print obs
        return {'state': obs}


if __name__ == '__main__':
    
    try:
        hand_control()
    except rospy.ROSInterruptException:
        pass