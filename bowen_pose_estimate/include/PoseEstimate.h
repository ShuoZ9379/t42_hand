#ifndef POSEESTIMATE_HH_
#define POSEESTIMATE_HH_

#include <ros/ros.h>
#include <memory>
#include <opencv2/opencv.hpp>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <boost/foreach.hpp>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include "Utils.h"
#include "ConfigParser.h"
#include <bowen_pose_estimate/getpose.h>
#include <bowen_pose_estimate/recordHandPose.h>
#include <geometry_msgs/PoseArray.h>


class PoseEstimate
{
public:
  PoseEstimate();
  PoseEstimate(ConfigParser* cfg1);
  PoseEstimate(std::shared_ptr<ros::NodeHandle> nh, ConfigParser* cfg1);
  ~PoseEstimate();
  // void cameraCallback(PointCloud::ConstPtr& msg);
  void removeSurface(PointCloudRGB::Ptr cloud_out);
  void cylinderFittingWithNormal(PointCloudRGB::Ptr cloud_in);
  Eigen::Matrix4f computePose(PointCloudRGB::Ptr cylinder_cloud);
  bool markerPose();
  bool markerPose1(const cv::Mat &image_in);

  ros::Subscriber _rgb_subscriber, _depth_subscriber, _cld_subscriber;
  ros::Publisher _cy_publisher, _drop_publisher, _cy_corner_publisher, _marker_publisher, _finger_markers_pub, _marker2d_pub;
  cv::Mat _rgb_image, _depth_image, _rgb_image1, _depth_image1, _marker_debug_img;
  Eigen::Matrix4f _cam2handmarker, _cam2handmarker1, _cy2world, _cy2world1;
  std::shared_ptr<ros::NodeHandle> _nh;
  bool _rgb_inited, _depth_inited;
  bool _recorded_hand_pose;
  PointCloudRGB::Ptr _cloud_rgb;
  ConfigParser* cfg;
  ros::ServiceServer _pose_server, _record_hand_pose_server;
  PointCloudRGB::Ptr _cloud_filtered_sparse, _dense_raw_cloud,_dense_raw_cloud1;  // supposed to be downsampled cylinder only cloud, i.e. final one
  int num_image;
  Eigen::Matrix4f _recorded_hand2cam,_cam2global, _cam2global1, _cy2cam1;
  Eigen::Vector4f _cylinder_center, _handmarker_center, _cylinder_center1;  // Finally in global
  bool _is_drop;
  int image_count;
  geometry_msgs::Pose _last_msg;
  int _miss_detect_cnt;
  float _bad_z;
  Eigen::Vector4f _cylinder_corner0, _cylinder_corner0_last;
  std::vector<int> _finger_marker_ids;
  std::vector<float> _finger_marker_sizes;
  std::vector<Eigen::Vector4f,Eigen::aligned_allocator<Eigen::Vector4f> > _fingermarkers_in_global, _fingermarkers_in_global_last;
  
  
protected:
  void subscribeMsg();
  void subscribeMsg1();
  bool poseService(bowen_pose_estimate::getpose::Request& req, bowen_pose_estimate::getpose::Response& res);
  bool handPoseRecordService(bowen_pose_estimate::recordHandPose::Request &req, bowen_pose_estimate::recordHandPose::Response &res);
  void rgbCallback(const sensor_msgs::Image::ConstPtr &msg);  
  void depthCallback(const sensor_msgs::Image::ConstPtr &msg);
  Eigen::Vector3f markerNormal(const std::vector<std::vector<cv::Point2f>> &corners);
  pcl::PointXYZRGB getMarkerCenter3D(const std::vector<std::vector<cv::Point2f>> &all_corners);
  pcl::PointXYZRGB getMarkerCenter3D1(const std::vector<std::vector<cv::Point2f>> &all_corners);
  void getCylinderInBowl(PointCloudRGB::Ptr cloud);
  pcl::PointXYZRGB getPix3D(float u, float v);
  void recordData();
  void arucoBoardDetection();
  void processOneFingerMarker(const std::vector<std::vector<cv::Point2f>> &marker_corners,const std::vector<int> &marker_ids, int query_idx, float marker_size, Eigen::Vector4f &pos_in_basemarker);


  tf::TransformBroadcaster _hand2cam_br;
  
};


#endif