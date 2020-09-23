#!/usr/bin/env python

import rospy,sys,cv2
import numpy as np
import time
import random
from std_msgs.msg import String, Float32MultiArray, Bool
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from std_srvs.srv import Empty, EmptyResponse
from hand_control.srv import observation, IsDropped, TargetAngles, RegraspObject, close
import glob
#from zs_transition_experience import *
#import roslib
#roslib.load_manifest('my_package')


class image_converter():
    start=False

    def __init__(self):
        rospy.init_node('zs_visual_save', anonymous=True)
        self.bridge = CvBridge()
        rospy.Subscriber("/camera/color/image_raw",Image,self.callbackImgConvert)
        rate = rospy.Rate(10) 
        ct=0
        idx=0
        while not rospy.is_shutdown():
            if self.start==True:
                #print(self.cv_image)
                cv2.imwrite("/home/szhang/Desktop/pre_vali/a"+str(idx)+".png", self.cv_image)
                print("saved")
                idx+=1
                if idx==10:
                    sys.exit()
                rate.sleep()

    def callbackImgConvert(self,data):
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            self.start=True
        except CvBridgeError as e:
            print(e)

    
if __name__ == '__main__':
    
    try:
        #pose_estimate()
        image_converter()
    except rospy.ROSInterruptException:
        pass
