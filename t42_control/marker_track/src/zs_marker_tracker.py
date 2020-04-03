#!/usr/bin/env python

import rospy,sys,cv2
import numpy as np
import time
import random
from std_msgs.msg import String, Float32MultiArray, Bool
from geometry_msgs.msg import Pose, PoseArray
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from std_srvs.srv import Empty, EmptyResponse
from hand_control.srv import observation, IsDropped, TargetAngles, RegraspObject, close
import glob


class marker_tracker():
    global_start=False
    cam_mat=np.array([[619.4442138671875, 0.0, 315.49298095703125], [0.0, 619.4443359375, 237.61129760742188], [0.0, 0.0, 1.0]])
    dist_coeff=np.array([[0.0,0.0,0.0,0.0]])
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
    parameters = cv2.aruco.DetectorParameters_create()

    def __init__(self):
        rospy.init_node('zs_marker_tracker', anonymous=True)
        self.bridge = CvBridge()
        rospy.Subscriber("/camera/color/image_raw",Image,self.callbackImgConvert)
        pub_obj_pose = rospy.Publisher('/cylinder_pose', Pose, queue_size=2)
        pub_obj_corner = rospy.Publisher('/cylinder_corner', Pose, queue_size=2)
        pub_obj_drop = rospy.Publisher('/cylinder_drop', Bool, queue_size=2)
        pub_fingers =  rospy.Publisher("/finger_markers",PoseArray,queue_size=2)

        msg = Pose()
   
        rate = rospy.Rate(10) 
        while not rospy.is_shutdown():
            if self.global_start==True:
                obj_drop=False
                ct=0
                obj_drop,im,all_corners,all_ids,base_rmat,base_tvecs=self.detect_obj_drop()

                while obj_drop:
                    ct+=1
                    if ct>0:
                        #rospy.logerr('[marker_tracker] Object Dropped is True!')
                        pub_obj_drop.publish(obj_drop)
                        print('yes')
                        break
                    obj_drop,im,all_corners,all_ids,base_rmat,base_tvecs=self.detect_obj_drop()
                       
                if obj_drop==False:
                    pub_obj_drop.publish(obj_drop)
                    finger_ids=list(all_ids)
                    finger_corners=list(all_corners)
                    obj_idx=all_ids.index(5)
                    base_idx=all_ids.index(0)

                        
                    if obj_idx > base_idx:
                        finger_ids.pop(obj_idx)
                        finger_ids.pop(base_idx)
                        finger_corners.pop(obj_idx)
                        finger_corners.pop(base_idx)
                    else:
                        finger_ids.pop(base_idx)
                        finger_ids.pop(obj_idx)
                        finger_corners.pop(base_idx)
                        finger_corners.pop(obj_idx)

                    obj_rvecs, obj_tvecs, _objPoints=cv2.aruco.estimatePoseSingleMarkers(all_corners[obj_idx], 0.02007, self.cam_mat, self.dist_coeff)
                    obj_rmat,jacobian=cv2.Rodrigues(obj_rvecs)    

                    obj_pos=base_rmat.T.dot((obj_tvecs-base_tvecs).reshape(-1))
                    msg=convert_pose(msg,obj_pos)
                    pub_obj_pose.publish(msg)
                        
                    obj_corner=base_rmat.T.dot((obj_rmat.dot(np.array([-0.02007/2,0.02007/2,0.0]))+obj_tvecs-base_tvecs).reshape(-1))
                    msg=convert_pose(msg,obj_corner)
                    pub_obj_corner.publish(msg)

                    finger_rvecs, finger_tvecs, _objPoints=cv2.aruco.estimatePoseSingleMarkers(finger_corners, 0.01295, self.cam_mat, self.dist_coeff)
                    msgArray=PoseArray()
                    for finger_id in range(1,5):
                        idx=finger_ids.index(finger_id)
                        finger_pos=base_rmat.T.dot((finger_tvecs[idx]-base_tvecs).reshape(-1))
                        msg=convert_pose(msg,finger_pos)
                        msgArray.poses.append(msg)
                    pub_fingers.publish(msgArray)

            rate.sleep()

    def callbackImgConvert(self,data):
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            self.global_start=True
        except CvBridgeError as e:
            print(e)

    def detect_markerIds(self):
        im=cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2RGB)
        all_corners, all_ids, rejectedImgPoints = cv2.aruco.detectMarkers(im, self.aruco_dict, parameters=self.parameters)
        all_ids=list(all_ids.reshape(len(all_ids)))
        return im,all_corners,all_ids

    def detect_obj_drop(self):
        im,all_corners,all_ids=self.detect_markerIds()
        if 0 not in all_ids:
            rospy.logwarn('[marker_tracker] Base marker Not Detected!')
            return True,0,0,0,0,0
        else:
            if 5 not in all_ids:
                rospy.logwarn('[marker_tracker] Object Not Detected!')
                return True,0,0,0,0,0
            else:
                if 1 not in all_ids or 2 not in all_ids or 3 not in all_ids or 4 not in all_ids:
                    rospy.logwarn('[marker_tracker] Some finger marker not detected!')
                    return True,0,0,0,0,0
                else:
                    base_idx=all_ids.index(0)
                    base_rvecs, base_tvecs, _objPoints=cv2.aruco.estimatePoseSingleMarkers(all_corners[base_idx], 0.01295, self.cam_mat, self.dist_coeff)
                    base_rmat,jacobian=cv2.Rodrigues(base_rvecs)
                    #base_rmat=np.array([[-0.06203229,0.99632318,0.05909421],[0.99103353,0.05446638,0.12200798],[0.11834073,0.06613278,0.99076835]])
                    #base_tvecs=np.array([-0.02883384,0.08533936,0.26374578])
                    obj_idx=all_ids.index(5)
                    obj_rvecs, obj_tvecs, _objPoints=cv2.aruco.estimatePoseSingleMarkers(all_corners[obj_idx], 0.02007, self.cam_mat, self.dist_coeff)
                    obj_height=base_rmat.T.dot((obj_tvecs-base_tvecs).reshape(-1))[-1]
                    x=base_rmat.T.dot((obj_tvecs-base_tvecs).reshape(-1))[0]
                    y=base_rmat.T.dot((obj_tvecs-base_tvecs).reshape(-1))[1]
                    obj_rmat,jacobian=cv2.Rodrigues(obj_rvecs)
                    print("cos(theta)",obj_rmat[:,2].dot(base_rmat[:,2]))
                    print(base_rmat)
                    print(base_tvecs)
                    print(x,y,obj_height)
                    if obj_height<-0.05 or obj_height>0.05:#0.000 start
                        rospy.logwarn('[marker_tracker] Object Dropped!')
                        return True,0,0,0,0,0
                    else:
                        return False,im,all_corners,all_ids,base_rmat,base_tvecs

def convert_pose(msg,pos):
    msg.position.x=pos[1]
    msg.position.y=-pos[0]
    msg.position.z=pos[2]
    return msg
       
    

if __name__ == '__main__':
    
    try:
        #pose_estimate()
        marker_tracker()
    except rospy.ROSInterruptException:
        pass
