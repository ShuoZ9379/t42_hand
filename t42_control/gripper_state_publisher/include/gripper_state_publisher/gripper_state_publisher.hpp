#include "ros/ros.h"
#include "std_msgs/Float32MultiArray.h"
class GripperStatePublisher{

	ros::NodeHandle n_;
	ros::ServiceClient srv_clnt_pos_, srv_clnt_load_, srv_clnt_current_, srv_clnt_temp_;
	ros::Publisher pub_pos_, pub_curr_, pub_load_, pub_pos_ref_, pub_temp_;
	ros::Rate loop_rate_;

	std_msgs::Float32MultiArray temp;
public:
	GripperStatePublisher(double frequency);
	void spin();	
};
