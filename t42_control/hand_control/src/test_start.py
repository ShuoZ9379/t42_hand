#!/usr/bin/env python

import rospy
from std_srvs.srv import Empty, EmptyResponse
from std_msgs.msg import Bool, String, Float32MultiArray
from hand_control.srv import RegraspObject, close
import numpy as np
import matplotlib.pyplot as plt
import pickle

obj_pos = np.array([0., 0.])
gripper_load = np.array([0., 0.])
gripper_closed = "open"
trigger = False
arm_status = ''
drop = True


def callbackObj(msg):
    global obj_pos
    Obj_pos = np.array(msg.data)
    obj_pos = Obj_pos[:2] * 1000

def callbackGripperLoad(msg):
    global gripper_load
    gripper_load = np.array(msg.data)

def callbackObjectDrop(msg):
    global drop
    drop = msg.data

def callbackTrigger(msg):
    global arm_status
    arm_status = msg.data
    global trigger
    if not trigger and arm_status == 'finished':
        trigger = True

rospy.init_node('test_start', anonymous=True)
reset_srv = rospy.ServiceProxy('/RegraspObject', RegraspObject)
rospy.Subscriber('/ObjectIsReset', String, callbackTrigger)
rospy.Subscriber('/hand_control/obj_pos_mm', Float32MultiArray, callbackObj)
rospy.Subscriber('/gripper/load', Float32MultiArray, callbackGripperLoad)
rospy.Subscriber('/cylinder_drop', Bool, callbackObjectDrop)
open_srv = rospy.ServiceProxy('/OpenGripper', Empty) 

rate = rospy.Rate(2.5) 

if 1:
    S = []
    L = []
    # with open('t42_start_states.obj', 'rb') as f:
    #     S, L = pickle.load(f)
    while len(S) < 10:
        print "Run ", len(S)
        # Reset gripper
        while 1:
            if not trigger and arm_status == 'waiting':
                reset_srv()
                rospy.sleep(1.0)
            rate.sleep()
            if arm_status != 'moving' and trigger:
                rate.sleep()
                if drop: # Check if really grasped
                    trigger = False
                    print('[rollout_action_publisher] Grasp failed. Restarting')
                    continue
                else:
                    break

        rate.sleep()

        S.append(obj_pos)
        L.append(gripper_load)

        open_srv()
        trigger = False
        rate.sleep()

        if not (len(S) % 5):
            with open('t42_start_states.obj', 'wb') as f:
                pickle.dump([S, L], f)
else:
    with open('t42_start_states.obj', 'rb') as f:
        S, L = pickle.load(f)

S = np.array(S)
L = np.array(L)

print "Mean: ", np.mean(S, 0), np.mean(L, 0)
print "Std: ", np.std(S, 0), np.std(L, 0)


fig = plt.figure(1)
plt.plot(S[:,0], S[:,1],'.')

fig = plt.figure(2)
plt.plot(L[:,0], L[:,1],'.')

plt.show()


# Mean:  [  39.97240439  107.67179579] [ 107.4 -107.4]
# Std:  [ 0.83943551  2.56771237] [ 3.2         2.87054002]


