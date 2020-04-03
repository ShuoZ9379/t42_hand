#include "Utils.h"
#include "PoseEstimate.h"
#include "ConfigParser.h"


int main(int argc, char** argv)
{
  ros::init(argc,argv,"testService");
  ConfigParser cfg("/home/catkin_ws/src/bowen_pose_estimate/config.yaml");
  std::shared_ptr<ros::NodeHandle> nh(new ros::NodeHandle);
  PoseEstimate p(nh, &cfg);
}