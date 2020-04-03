#!/usr/bin/env python

import rospy
import numpy as np
import time
import random
from transition_experience import *
from std_msgs.msg import Float64MultiArray, Float32MultiArray, String, Bool,Float32
from std_srvs.srv import Empty, EmptyResponse
import geometry_msgs.msg

class actorPubRec():
    discrete_actions = True

    gripper_pos = np.array([0., 0.])
    gripper_load = np.array([0., 0.])
    obj_pos = np.array([0., 0.])
    drop = True
    Done = False
    running = False
    action = np.array([0.,0.])
    state = np.array([0.,0., 0., 0.])
    angle = np.array([0.])
    n = 0
    marker0 = np.array([0.,0.])
    marker1 = np.array([0.,0.])
    marker2 = np.array([0.,0.])
    marker3 = np.array([0.,0.])
    
    texp = transition_experience(Load=True, discrete = discrete_actions, postfix = '')

    def __init__(self):
        rospy.init_node('actor_pub_record', anonymous=True)

        rospy.Subscriber('/gripper/load', Float32MultiArray, self.callbackGripperLoad)
        rospy.Subscriber('/hand_control/obj_pos_mm', Float32MultiArray, self.callbackObj)
        rospy.Subscriber('/cylinder_drop', Bool, self.callbackDrop)
        rospy.Subscriber('/collect/action', Float32MultiArray, self.callbackAction)
        rospy.Subscriber('/object_orientation',Float32MultiArray, self.callbackOrientation)
        rospy.Subscriber('/finger_markers', geometry_msgs.msg.PoseArray, self.callAddFingerPos)
        rospy.Service('/actor/trigger', Empty, self.callbackTrigger)
        rospy.Service('/actor/save', Empty, self.callbackSave)

        rate = rospy.Rate(10)
        count = 1
        while not rospy.is_shutdown():

            if self.running:
                #count += 1

                self.state = np.concatenate((self.obj_pos, self.angle, self.marker0, self.marker1, self.marker2, self.marker3, self.gripper_load), axis=0)
                D = self.drop
                self.texp.add(rospy.get_time()-self.T, self.state, self.action, self.state, D)

                if self.drop:
                    c = 0
                    while self.drop:
                        if c == 10:
                            print('[actor_record] Episode ended (%d points so far).' % self.texp.getSize())
                            self.running = False
                            if not D:
                                self.texp.add(rospy.get_time()-self.T, self.state, self.action, self.state, self.drop)
                            break
                        c += 1
                        rate.sleep()

            rate.sleep()

    def callbackGripperLoad(self, msg):
        self.gripper_load = np.array(msg.data)

    def callbackObj(self, msg):
        self.obj_pos = np.array(msg.data)

    def callbackDrop(self, msg):
        self.drop = msg.data

    def callbackOrientation(self,msg):
        self.angle = msg.data

    def callbackAction(self, msg):
        self.action = np.array(msg.data)

    def callAddFingerPos(self, msg):
        tempMarkers =  msg.poses

        self.marker0[0] = tempMarkers[0].position.x
        self.marker0[1] = tempMarkers[0].position.y

        self.marker1[0] = tempMarkers[1].position.x
        self.marker1[1] = tempMarkers[1].position.y

        self.marker2[0] = tempMarkers[2].position.x
        self.marker2[1] = tempMarkers[2].position.y 

        self.marker3[0] = tempMarkers[3].position.x
        self.marker3[1] = tempMarkers[3].position.y

    def callbackTrigger(self, msg):
        self.running = not self.running

        if self.running:
            self.T = rospy.get_time()
        else:
            print('[actor_record] Episode ended (%d points so far).' % self.texp.getSize())

        return EmptyResponse()

    def callbackSave(self, msg):
        self.texp.save()

        return EmptyResponse()


if __name__ == '__main__':
    
    try:
        actorPubRec()
    except rospy.ROSInterruptException:
        pass
