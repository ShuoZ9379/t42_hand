#!/usr/bin/python 

'''
----------------------------
Author: Avishai Sintov
        Rutgers University
Date: October 2018
----------------------------
'''

import rospy
import numpy as np 
from std_msgs.msg import Float64, Float32MultiArray
from gazebo_msgs.msg import ModelStates, LinkStates
import PyKDL
import time

windowSize = 100

class object_state_publisher():

    obj_pos = np.array([0.,0.,0.])
    obj_pos_prev = np.array([0.,0.,0.])
    obj_vel = np.array([0.,0.,0.])
    R_obj = None
    hand_pos = np.array([0.,0.,0.])
    R_hand = None
    msg = Float32MultiArray()
    win = np.array([])
    t_prev = 0.0
    O = np.array([]) # Object last positions
    dt = 0.0001
    

    def __init__(self):
        rospy.init_node('object_state_publisher', anonymous=True)

        rospy.Subscriber('/gazebo/model_states', ModelStates, self.ObjCallback)
        rospy.Subscriber('/gazebo/link_states', LinkStates, self.HandCallback)
        object_position_pub = rospy.Publisher('/hand/obj_pos', Float32MultiArray, queue_size=10)
        object_velocity_pub = rospy.Publisher('/hand/obj_vel', Float32MultiArray, queue_size=10)
        object_orientation_pub = rospy.Publisher('/hand/obj_orientation', Float32MultiArray, queue_size=10)

        self.t_prev = rospy.get_time()

        self.freq = 50.0
        self.rate = rospy.Rate(self.freq)
        while not rospy.is_shutdown():

            if self.R_obj != None and self.R_hand != None:
                object_pos = self.obj_pos - self.hand_pos # Position relative to the hands base link
                object_pos = np.array([object_pos[1],-object_pos[0],object_pos[2]])
                obj_orientation = (self.R_hand*self.R_obj.Inverse()).GetEulerZYX() # Orientation relative to hand - I did not verify that this is correct

                # Compute object velocity using second order derivative (the gazebo measurement is much more noisy)
                self.obj_vel = (object_pos - self.obj_pos_prev) / (1/self.freq) #self.dt
                self.obj_pos_prev = np.copy(object_pos)

                if self.win.shape[0] < windowSize:
                    self.win = np.append(self.win, self.obj_vel).reshape(-1, 3)
                else:
                    v = self.obj_vel
                    self.win = np.append(self.win, v).reshape(-1, 3)
                    self.obj_vel = np.mean(self.win, axis=0)
                    self.win = np.delete(self.win, 0, axis=0)   

                # self.obj_vel[np.where(np.abs(self.obj_vel) < 1e-5)] = 0.0

                self.msg.data = object_pos
                object_position_pub.publish(self.msg)
                self.msg.data = self.obj_vel
                object_velocity_pub.publish(self.msg)
                self.msg.data = obj_orientation
                object_orientation_pub.publish(self.msg)

            self.rate.sleep()

    def ObjCallback(self, msg):

        idx = self.getNameOrder(msg.name, 'object')

        self.obj_pos = np.array([msg.pose[idx].position.x, msg.pose[idx].position.y, msg.pose[idx].position.z])
        self.R_obj = PyKDL.Rotation.Quaternion(msg.pose[idx].orientation.x, msg.pose[idx].orientation.y, msg.pose[idx].orientation.z, msg.pose[idx].orientation.w)
        
        # Apply mean filter to object velocity with windowSize
        # 

        

        # self.rate.sleep()

    def HandCallback(self, msg):

        idx = self.getNameOrder(msg.name, 'base_link')
        
        self.hand_pos = np.array([msg.pose[idx].position.x, msg.pose[idx].position.y, msg.pose[idx].position.z])
        self.R_hand = PyKDL.Rotation.Quaternion(msg.pose[idx].orientation.x, msg.pose[idx].orientation.y, msg.pose[idx].orientation.z, msg.pose[idx].orientation.w)
        
    def getNameOrder(self, names, str):

        idx = -1
        for i, name in enumerate(names):
            if name == str:
                idx = i
            if name[-9:] == str:
                idx = i

        return idx


if __name__ == '__main__':

    try:
        object_state_publisher()
    except rospy.ROSInterruptException:
        pass