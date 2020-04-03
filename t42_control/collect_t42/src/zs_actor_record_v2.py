#!/usr/bin/env python

import rospy
import numpy as np
import time, sys
import random
from zs_transition_experience_v2 import *
from hand_control.srv import observation
from std_msgs.msg import Float64MultiArray, Float32MultiArray, String, Bool,Float32
from std_srvs.srv import Empty, EmptyResponse
import geometry_msgs.msg

class actorPubRec():
    running = False
    over = False
    action = np.array([0.,0.])
    over_sign=True
    
    def __init__(self):
        rospy.init_node('zs_actor_record_v2', anonymous=True)
        if rospy.has_param('~object_name'):
            self.collect_idx=rospy.get_param('~collect_idx')
            Obj=rospy.get_param('~object_name')
            ver=rospy.get_param('~version')
            col=rospy.get_param('~color')
        discrete_actions = True
        self.texp = zs_transition_experience_v2(Object = Obj, version=ver, color=col,Load=True, discrete = discrete_actions, postfix = '')

        rospy.Subscriber('/collect/action', Float32MultiArray, self.callbackAction)
        obs_srv = rospy.ServiceProxy('/observation', observation)
        pub_recorder_running=rospy.Publisher('/recorder_running', Bool, queue_size=10)
        rospy.Service('/actor/trigger', Empty, self.callbackTrigger)
        rospy.Service('/actor/save', Empty, self.callbackSave)

        rate = rospy.Rate(10)
        count_fail = 0
        while not rospy.is_shutdown():

            if self.running:
                state=obs_srv().state
                act=self.action
                self.Done = False
                if state[-1] or state[-2] or state[-3]:
                    count_fail+=1
                    if count_fail>=12:
                        self.Done=True
                else:
                    count_fail=0
                
                if self.over_sign==False:
                    self.texp.add(self.collect_idx,rospy.get_time()-self.T, state, act, state, self.Done)
                    if self.Done:
                        self.running=False
                        self.over_sign=True
                        print('[actor_record] Episode ended (%d points so far).' % self.texp.getSize())

            pub_recorder_running.publish(self.running)
            rate.sleep()


    def callbackAction(self, msg):
        self.action = np.array(msg.data)

    def callbackTrigger(self, msg):
        self.running = not self.running

        if self.running:
            self.over_sign=False
            self.T = rospy.get_time()
        else:
            obs_srv = rospy.ServiceProxy('/observation', observation)
            if self.Done==False:
                self.over_sign=True
                state=obs_srv().state
                act=self.action
                self.texp.add(self.collect_idx,rospy.get_time()-self.T, state, act, state, True)
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
