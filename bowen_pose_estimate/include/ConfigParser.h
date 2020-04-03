#ifndef CONFIGPARSER_HH_
#define CONFIGPARSER_HH_ 

#include "Utils.h"

class ConfigParser
{
public:

  ConfigParser(std::string cfg_file);
  ~ConfigParser();
  void parseYMLFile(std::string filepath);

  Eigen::Matrix3f cam_intrinsic;
  Eigen::Matrix4f handmarker2endeffector, endeffector2global;
  ros::NodeHandle nh;
  std::string rgb_topic, depth_topic, rgb_topic1, depth_topic1;
  float leaf_size, distance_threshold, distance_to_table, radius, cylinder_distance_threshold, radius_limits_min, radius_limits_max, normal_distance_weight, k_search;
  int max_iterations, min_number;
  std::string rgb_path, depth_path, marker_path;
  float adaptiveThreshWinSizeMax,minDistanceToBorder,adaptiveThreshWinSizeStep,aprilTagCriticalRad,aprilTagMinClusterPixels,minCornerDistanceRate,
        minMarkerPerimeterRate,markerBorderBits,minOtsuStdDev,perpectiveRemovePixelPerCell,perspectiveRemoveIgnoredMarginPerCell,maxErroneousBitsInBorderRate,errorCorrectionRate,cornerRefinementWinSize,cornerRefinementMaxIterations,cornerRefinementMinAccuracy,cornerRefinementMethod;
  Eigen::Matrix4f recorded_hand2cam, recorded_hand2cam1;
  Eigen::Matrix4f cam_in_base;
};


#endif