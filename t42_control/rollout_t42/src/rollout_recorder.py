#!/usr/bin/env python

import rospy
from std_msgs.msg import String, Float32MultiArray, Bool
from std_srvs.srv import Empty, EmptyResponse, SetBool
from rollout_t42.srv import rolloutReq, rolloutReqFile, plotReq, observation, IsDropped, TargetAngles, gets
from hand_control.srv import RegraspObject, close
import numpy as np
import matplotlib.pyplot as plt
import pickle
import geometry_msgs.msg
from sensor_msgs.msg import CompressedImage, Image

record_images = False

class rolloutRecorder():

    gripper_load = np.array([0., 0.])
    obj_pos = np.array([0., 0.])
    drop = True
    running = False
    action = np.array([0.,0.])
    S = A = T = []
    suc = True
    drop_counter = 0
    T0 = 0.
    fail = False
    angle = np.array([0.])
    marker0 = np.array([0.,0.])
    marker1 = np.array([0.,0.])
    marker2 = np.array([0.,0.])
    marker3 = np.array([0.,0.])

    images = []
    I = []
    pos_image = np.array([0,0])
    Simage = []
    Timages = []
    
    def __init__(self):
        rospy.init_node('rollout_recorder', anonymous=True)

        rospy.Subscriber('/gripper/load', Float32MultiArray, self.callbackGripperLoad)
        rospy.Subscriber('/hand_control/obj_pos_mm', Float32MultiArray, self.callbackObj)
        # rospy.Subscriber('/cylinder_drop', Bool, self.callbackObjectDrop)
        rospy.Subscriber('/rollout/action', Float32MultiArray, self.callbackAction)
        # rospy.Subscriber('/rollout/move_success', Bool, self.callbackSuccess)
        rospy.Subscriber('/rollout/fail', Bool, self.callbacFail)
        rospy.Subscriber('/object_orientation',Float32MultiArray, self.callbackOrientation)
        rospy.Subscriber('/finger_markers', geometry_msgs.msg.PoseArray, self.callAddFingerPos)

        rospy.Service('/rollout/record_trigger', SetBool, self.callbackTrigger)
        rospy.Service('/rollout/get_states', gets, self.get_states)
        rollout_actor_srv = rospy.ServiceProxy('/rollout/run_trigger', SetBool)
        rollout_srv = rospy.ServiceProxy('/rollout/run_trigger', SetBool)
        if record_images:
            rospy.Subscriber('/camera/color/image_raw', Image, self.callbackImage,  queue_size = 1)
            rospy.Subscriber('/marker2D', geometry_msgs.msg.Pose, self.callbackPosImage)

        print('[rollout_recorder] Ready ...')

        if not record_images:
            self.rate = rospy.Rate(10) # 10hz
        else:
            self.rate = rospy.Rate(2) # 2hz
        while not rospy.is_shutdown():

            if self.running:
                # self.state = np.concatenate((self.obj_pos, self.angle, self.marker0, self.marker1, self.marker2, self.marker3, self.gripper_load), axis=0)
                self.state = np.concatenate((self.obj_pos, self.gripper_load), axis=0)
                    
                self.S.append(self.state)
                self.A.append(self.action)
                self.T.append(rospy.get_time() - self.T0)

                if record_images:
                    self.images.append(self.I)
                    self.Simage.append(self.pos_image)
                    self.Timages.append(rospy.get_time() - self.T0)

                if self.fail:
                    print('[rollout_recorder] Episode ended with %d points.'%len(self.S))
                    self.running = False

            self.rate.sleep()

    def callbackGripperLoad(self, msg):
        self.gripper_load = np.array(msg.data)

    def callbackObj(self, msg):
        self.obj_pos = np.array(msg.data)*1000

    def callbackAction(self, msg):
        self.action = np.array(msg.data)

    def callbacFail(self, msg):
        self.fail = msg.data

    def callbackOrientation(self,msg):
        self.angle = msg.data

    def callbackImage(self, msg):
        # self.I = msg.data
        self.I = np.fromstring(msg.data, np.uint8).reshape(480,640,3)
        # self.I = cv2.imdecode(self.I, -1)
        # print self.I.shape
        # plt.imshow(self.I)
        # plt.show()

    def callbackPosImage(self, msg):
        self.pos_image = np.array([msg.position.x, msg.position.y])

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
        self.running = msg.data
        if self.running:
            self.S = []
            self.A = []
            self.T = []
            self.T0 = rospy.get_time()
            self.images = []
            self.Timages = []
            print('[rollout_recorder] Recording started.')
            self.fail = False
        else:
            print('[rollout_recorder] Recording stopped with %d points.'%len(self.S))

        return {'success': True, 'message': ''}

    def get_states(self, msg):

        if record_images:
            # with open('/media/pracsys/DATA/hand_images_data/rollout_blue_' + str(np.random.randint(10000)) + '.pkl', 'wb') as f:
            with open('/home/pracsys/Videos/rollout_red_' + str(np.random.randint(10000)) + '.pkl', 'wb') as f:
                pickle.dump([self.Timages, self.Simage, self.images, self.S, self.A, self.T], f)          

        print(np.array(self.S).shape)
        return {'states': np.array(self.S).reshape((-1,)), 'actions': np.array(self.A).reshape((-1,)), 'time': np.array(self.T).reshape((-1,))}

    def medfilter(self, x, W):
        print('[rollout_recorder] Smoothing data...')
        w = int(W/2)
        x_new = np.copy(x)
        for i in range(0, x.shape[0]):
            if i < w:
                x_new[i] = np.mean(x[:i+w])
            elif i > x.shape[0]-w:
                x_new[i] = np.mean(x[i-w:])
            else:
                x_new[i] = np.mean(x[i-w:i+w])
        return x_new

if __name__ == '__main__':
    try:
        rolloutRecorder()
    except rospy.ROSInterruptException:
        pass