#!/usr/bin/env python

import rospy,sys,cv2
import numpy as np
import time
import random
from std_msgs.msg import String, Float32MultiArray, Bool, Float32
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
        pub_obj_pose = rospy.Publisher('/cylinder_pose_incam', Pose, queue_size=2)
        pub_obj_corner = rospy.Publisher('/cylinder_corner_incam', Pose, queue_size=2)
        pub_fingers =  rospy.Publisher("/finger_markers_incam",PoseArray,queue_size=2)
        pub_no_detect = rospy.Publisher('/no_detect', Bool, queue_size=2)
        pub_no_paral = rospy.Publisher('/no_paral', Bool, queue_size=2)
        pub_obj_drop = rospy.Publisher('/cylinder_drop', Bool, queue_size=2)
        pub_base_rmat = rospy.Publisher('/base_rmat', Float32MultiArray, queue_size=2)
        pub_base_tvec = rospy.Publisher('/base_tvec', Float32MultiArray, queue_size=2)
        base_rmat_msg=Float32MultiArray()
        base_tvec_msg=Float32MultiArray()
        
        msg_default=Pose()
        msg_defalut=convert_pose(msg_default,[0,0,0])

        msgArray_default=PoseArray()
        for finger_id in range(1,5):
            msgArray_default.poses.append(msg_default)

        msg = Pose()
        msgArray=PoseArray()
        no_detect_default=False
        no_paral_default=False
        obj_drop_default=False

        rate = rospy.Rate(10) 
        while not rospy.is_shutdown():
            if self.global_start==True:
                im,all_corners,all_ids=self.detect_markerIds()
                if all_ids is None:
                    rospy.logwarn('[marker_tracker] No Marker Detected!')
                    pub_no_detect.publish(True)
                    pub_no_paral.publish(True)
                    pub_obj_drop.publish(True)
                    pub_obj_pose.publish(msg_default)
                    pub_obj_corner.publish(msg_default)
                    pub_fingers.publish(msgArray_default)
                    base_rmat_msg.data=np.zeros(9)
                    base_tvec_msg.data=np.zeros(3)
                    pub_base_rmat.publish(base_rmat_msg)
                    pub_base_tvec.publish(base_tvec_msg)
                    continue
                if 0 not in all_ids:
                    rospy.logwarn('[marker_tracker] Base marker Not Detected!')
                    pub_no_detect.publish(True)
                    pub_no_paral.publish(no_paral_default)
                    pub_obj_drop.publish(obj_drop_default)
                    pub_obj_pose.publish(msg_default)
                    pub_obj_corner.publish(msg_default)
                    pub_fingers.publish(msgArray_default)
                    base_rmat_msg.data=np.zeros(9)
                    base_tvec_msg.data=np.zeros(3)
                    pub_base_rmat.publish(base_rmat_msg)
                    pub_base_tvec.publish(base_tvec_msg)
                elif 5 not in all_ids:
                    rospy.logwarn('[marker_tracker] Object marker Not Detected!')
                    pub_no_detect.publish(True)
                    pub_no_paral.publish(no_paral_default)
                    pub_obj_drop.publish(obj_drop_default)
                    pub_obj_pose.publish(msg_default)
                    pub_obj_corner.publish(msg_default)
                    pub_fingers.publish(msgArray_default)
                    base_rmat_msg.data=np.zeros(9)
                    base_tvec_msg.data=np.zeros(3)
                    pub_base_rmat.publish(base_rmat_msg)
                    pub_base_tvec.publish(base_tvec_msg)
                elif 1 not in all_ids or 2 not in all_ids or 3 not in all_ids or 4 not in all_ids:
                    print(all_ids)
                    rospy.logwarn('[marker_tracker] Some finger marker Not Detected!')

                    pub_no_detect.publish(True)
                    pub_no_paral.publish(no_paral_default)
                    pub_obj_drop.publish(obj_drop_default)
                    pub_obj_pose.publish(msg_default)
                    pub_obj_corner.publish(msg_default)
                    pub_fingers.publish(msgArray_default)
                    base_rmat_msg.data=np.zeros(9)
                    base_tvec_msg.data=np.zeros(3)
                    pub_base_rmat.publish(base_rmat_msg)
                    pub_base_tvec.publish(base_tvec_msg)
                else:
                    base_idx=all_ids.index(0)
                    base_rvecs, base_tvecs, _objPoints=cv2.aruco.estimatePoseSingleMarkers(all_corners[base_idx], 0.01295, self.cam_mat, self.dist_coeff)
                    base_rmat,jacobian=cv2.Rodrigues(base_rvecs)
                    obj_idx=all_ids.index(5)
                    obj_rvecs, obj_tvecs, _objPoints=cv2.aruco.estimatePoseSingleMarkers(all_corners[obj_idx], 0.02007, self.cam_mat, self.dist_coeff)
                    obj_rmat,jacobian=cv2.Rodrigues(obj_rvecs)
                    obj_pos=base_rmat.T.dot((obj_tvecs-base_tvecs).reshape(-1))
                    obj_height=obj_pos[2]
                    #print("Obj Pos Cam",obj_pos)
                    z_axis_ls=[base_rmat[:,2],obj_rmat[:,2]]
                    finger_ids=list(all_ids)
                    finger_corners=list(all_corners)
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
                    finger_rvecs, finger_tvecs, _objPoints=cv2.aruco.estimatePoseSingleMarkers(finger_corners, 0.01295, self.cam_mat, self.dist_coeff)
                    for finger_id in range(1,5):
                        idx=finger_ids.index(finger_id)
                        finger_rmat,jacobian=cv2.Rodrigues(finger_rvecs[idx])
                        z_axis_ls.append(finger_rmat[:,2])
                    if get_no_paral(z_axis_ls,threshold=0.93):
                        rospy.logwarn('[marker_tracker] Some z axes Not Parallel!')
                        pub_no_detect.publish(no_detect_default)
                        pub_no_paral.publish(True)
                        pub_obj_drop.publish(obj_drop_default)
                        pub_obj_pose.publish(msg_default)
                        pub_obj_corner.publish(msg_default)
                        pub_fingers.publish(msgArray_default)
                        base_rmat_msg.data=np.zeros(9)
                        base_tvec_msg.data=np.zeros(3)
                        pub_base_rmat.publish(base_rmat_msg)
                        pub_base_tvec.publish(base_tvec_msg)
                    elif obj_height<-0.05 or obj_height>0.05:
                        rospy.logwarn('[marker_tracker] Object Dropped!')
                        pub_no_detect.publish(no_detect_default)
                        pub_no_paral.publish(no_paral_default)
                        pub_obj_drop.publish(True)
                        pub_obj_pose.publish(msg_default)
                        pub_obj_corner.publish(msg_default)
                        pub_fingers.publish(msgArray_default)
                        base_rmat_msg.data=np.zeros(9)
                        base_tvec_msg.data=np.zeros(3)
                        pub_base_rmat.publish(base_rmat_msg)
                        pub_base_tvec.publish(base_tvec_msg)
                    else:
                        pub_no_detect.publish(no_detect_default)
                        pub_no_paral.publish(no_paral_default)
                        pub_obj_drop.publish(obj_drop_default)

                        obj_pos=obj_tvecs.reshape(-1)
                        msg=convert_pose(msg,obj_pos)
                        pub_obj_pose.publish(msg)
                        
                        obj_corner=(obj_rmat.dot(np.array([-0.02007/2,0.02007/2,0.0]))+obj_tvecs).reshape(-1)
                        msg=convert_pose(msg,obj_corner)
                        pub_obj_corner.publish(msg)

                        for finger_id in range(1,5):
                            idx=finger_ids.index(finger_id)
                            finger_pos=finger_tvecs[idx].reshape(-1)
                            msg=convert_pose(msg,finger_pos)
                            msgArray.poses.append(msg)
                        pub_fingers.publish(msgArray)

                        base_rmat_msg.data=base_rmat.flatten()
                        pub_base_rmat.publish(base_rmat_msg)
                        base_tvec_msg.data=base_tvecs.reshape(-1)
                        pub_base_tvec.publish(base_tvec_msg)

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
        if all_ids is not None:
            all_ids=list(all_ids.reshape(len(all_ids)))
        return im,all_corners,all_ids

def convert_pose(msg,pos):
    msg.position.x=pos[0]
    msg.position.y=pos[1]
    msg.position.z=pos[2]
    return msg

def get_no_paral(z_axis_ls,threshold):
    base_z=z_axis_ls[0]
    #for i in range(1,len(z_axis_ls)):
    for i in range(1,2):
        if base_z.dot(z_axis_ls[i])<threshold:
            print("Not Parallel "+ str(i-1), base_z.dot(z_axis_ls[i]))
            return True
    return False

    

if __name__ == '__main__':
    
    try:
        #pose_estimate()
        marker_tracker()
    except rospy.ROSInterruptException:
        pass
