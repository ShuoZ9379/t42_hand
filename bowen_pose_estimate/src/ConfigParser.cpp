#include "ConfigParser.h"


ConfigParser::ConfigParser(std::string cfg_file)
{
  parseYMLFile(cfg_file);
}

ConfigParser::~ConfigParser()
{

}


void ConfigParser::parseYMLFile(std::string filepath)
{
	system(("rosparam load " + filepath).c_str());

  std::vector<float> cam_in_base_data;
  nh.getParam("cam_in_base", cam_in_base_data);
  cam_in_base.setIdentity();
  cam_in_base.block(0,3,3,1)<<cam_in_base_data[0],cam_in_base_data[1],cam_in_base_data[2];
  Eigen::Quaternionf q(cam_in_base_data[6],cam_in_base_data[3],cam_in_base_data[4],cam_in_base_data[5]);
  Eigen::Matrix3f R(q);
  cam_in_base.block(0,0,3,3) = R;

  std::vector<float> recorded_hand2cam_data;
  nh.getParam("recorded_hand2cam", recorded_hand2cam_data);
  for (int i=0;i<16;i++)
  {
    recorded_hand2cam(i/4,i%4) = recorded_hand2cam_data[i];
  }

  std::vector<float> recorded_hand2cam_data1;
  nh.getParam("recorded_hand2cam1", recorded_hand2cam_data1);
  for (int i=0;i<16;i++)
  {
    recorded_hand2cam1(i/4,i%4) = recorded_hand2cam_data1[i];
  }
 
  std::vector<float> K;
	nh.getParam("cam_K", K);
  for (int i=0;i<9;i++)
  {
    cam_intrinsic(i/3,i%3) = K[i];
  }
  K.clear();
  nh.getParam("handmarker2endeffector", K);
  handmarker2endeffector.setIdentity();
  for (int i=0;i<16;i++)
  {
    handmarker2endeffector(i/4,i%4) = K[i];
  }
  endeffector2global.setIdentity();  // end effector has 90 deg offset from global
  Eigen::Matrix3f R_tmp;  
  R_tmp<<0,1,0,
         -1,0,0, 
         0,0,1;
  endeffector2global.block(0,0,3,3) = R_tmp;
	nh.getParam("rgb_msg_topic", rgb_topic);
  nh.getParam("depth_msg_topic", depth_topic);
  nh.getParam("rgb_msg_topic1", rgb_topic1);
  nh.getParam("depth_msg_topic1", depth_topic1);


  nh.getParam("rgb_path", rgb_path);
  nh.getParam("depth_path", depth_path);
  nh.getParam("marker_path", marker_path);

  //================== pcl options ========================
  nh.getParam("down_sample/leaf_size", leaf_size);
  nh.getParam("plane_fitting/distance_threshold", distance_threshold);
  nh.getParam("plane_fitting/max_iterations", max_iterations);
  nh.getParam("plane_fitting/distance_to_table", distance_to_table);
  nh.getParam("remove_noise/radius", radius);
  nh.getParam("remove_noise/min_number", min_number);

  nh.getParam("cylinder_fitting/k_search", k_search);
  nh.getParam("cylinder_fitting/distance_threshold", cylinder_distance_threshold);
  nh.getParam("cylinder_fitting/radius_limits_min", radius_limits_min);
  nh.getParam("cylinder_fitting/radius_limits_max", radius_limits_max);
  nh.getParam("cylinder_fitting/normal_distance_weight", normal_distance_weight);

  nh.getParam("marker/minDistanceToBorder", minDistanceToBorder);
  nh.getParam("marker/adaptiveThreshWinSizeMax", adaptiveThreshWinSizeMax);
  nh.getParam("marker/adaptiveThreshWinSizeStep", adaptiveThreshWinSizeStep);
  nh.getParam("marker/aprilTagCriticalRad", aprilTagCriticalRad);
  nh.getParam("marker/aprilTagMinClusterPixels", aprilTagMinClusterPixels);
  nh.getParam("marker/minCornerDistanceRate", minCornerDistanceRate);
  nh.getParam("marker/minMarkerPerimeterRate", minMarkerPerimeterRate);
  nh.getParam("marker/markerBorderBits", markerBorderBits);
  nh.getParam("marker/minOtsuStdDev", minOtsuStdDev);
  nh.getParam("marker/perpectiveRemovePixelPerCell", perpectiveRemovePixelPerCell);
  nh.getParam("marker/perspectiveRemoveIgnoredMarginPerCell", perspectiveRemoveIgnoredMarginPerCell);
  nh.getParam("marker/maxErroneousBitsInBorderRate", maxErroneousBitsInBorderRate);
  nh.getParam("marker/errorCorrectionRate", errorCorrectionRate);
  nh.getParam("marker/cornerRefinementWinSize", cornerRefinementWinSize);
  nh.getParam("marker/cornerRefinementMaxIterations", cornerRefinementMaxIterations);
  nh.getParam("marker/cornerRefinementMinAccuracy", cornerRefinementMinAccuracy);
  nh.getParam("marker/cornerRefinementMethod", cornerRefinementMethod);
  

}
