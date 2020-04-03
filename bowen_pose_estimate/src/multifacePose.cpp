#include "Utils.h"
#include "PoseEstimate.h"
#include "ConfigParser.h"
#include <opencv2/rgbd.hpp>
#include "bowen_pose_estimate/getMultifacePose.h"
#include "bowen_pose_estimate/multifacePose.h"

cv::Mat _rgb_image, _marker_debug_img;
Eigen::Matrix4f _marker2cam(Eigen::Matrix4f::Identity());
Eigen::Matrix4f _handbase2cam(Eigen::Matrix4f::Identity());
Eigen::Matrix4f _marker2handbase(Eigen::Matrix4f::Identity());
int _detected_id=-1;


void rgbCallback(const sensor_msgs::Image::ConstPtr &msg)
{
	cv_bridge::CvImagePtr bridge_ptr = cv_bridge::toCvCopy(msg, "bgr8");
	_rgb_image = bridge_ptr->image; //NOTE overwritting server _rgb_image
}


bool handbaseRecord(const ConfigParser &cfg, bowen_pose_estimate::recordHandPose::Request &req, bowen_pose_estimate::recordHandPose::Response &res)
{
	
	cv::Ptr<cv::aruco::DetectorParameters> marker_params = cv::aruco::DetectorParameters::create();	
	marker_params->minDistanceToBorder = cfg.minDistanceToBorder;
	marker_params->adaptiveThreshWinSizeMax = cfg.adaptiveThreshWinSizeMax;
	marker_params->adaptiveThreshWinSizeStep = cfg.adaptiveThreshWinSizeStep;
	marker_params->minCornerDistanceRate = cfg.minCornerDistanceRate;
	marker_params->minMarkerPerimeterRate = cfg.minMarkerPerimeterRate;
	marker_params->minOtsuStdDev = cfg.minOtsuStdDev;
	marker_params->perspectiveRemoveIgnoredMarginPerCell = cfg.perspectiveRemoveIgnoredMarginPerCell;
	marker_params->maxErroneousBitsInBorderRate = cfg.maxErroneousBitsInBorderRate;
	marker_params->errorCorrectionRate = cfg.errorCorrectionRate;
	marker_params->cornerRefinementWinSize = cfg.cornerRefinementWinSize;
	marker_params->cornerRefinementMaxIterations = cfg.cornerRefinementMaxIterations;
	marker_params->cornerRefinementMinAccuracy = cfg.cornerRefinementMinAccuracy;
	marker_params->cornerRefinementMethod = cfg.cornerRefinementMethod; 


	cv::Ptr<cv::aruco::Dictionary> dictionary_ = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
	std::vector<std::vector<cv::Point2f>> marker_corners_, rejected_points;
	std::vector<int> marker_ids;
	cv::aruco::detectMarkers(_rgb_image, dictionary_, marker_corners_, marker_ids, marker_params, rejected_points);
	
	const int handbase_start_id=8;
	auto it=std::find(marker_ids.begin(),marker_ids.end(),handbase_start_id);
	if (marker_ids.size()==0 || it==marker_ids.end())
	{
		ROS_ERROR("board not detected");
        return false;
	}
	int index=std::distance(marker_ids.begin(),it);

	cv::Ptr<cv::aruco::GridBoard> board;
	board = cv::aruco::GridBoard::create(2,1, 0.025, 0.005, dictionary_, handbase_start_id);
	cv::aruco::refineDetectedMarkers(_rgb_image, board, marker_corners_, marker_ids, rejected_points);

	cv::Mat camera_matrix_ = (cv::Mat1d(3, 3) <<cfg.cam_intrinsic(0,0),cfg.cam_intrinsic(0,1),cfg.cam_intrinsic(0,2),
												cfg.cam_intrinsic(1,0),cfg.cam_intrinsic(1,1),cfg.cam_intrinsic(1,2),
												cfg.cam_intrinsic(2,0),cfg.cam_intrinsic(2,1),cfg.cam_intrinsic(2,2));
	cv::Mat dist_coeffs_ = (cv::Mat1d(4,1)<<0,0,0,0);		

	cv::Vec3d rvec, tvec;
	_marker_debug_img = _rgb_image.clone();
	cv::aruco::estimatePoseBoard(marker_corners_,marker_ids, board, camera_matrix_, dist_coeffs_, rvec, tvec);
	cv::aruco::drawDetectedMarkers(_marker_debug_img, marker_corners_, marker_ids);
	cv::aruco::drawAxis(_marker_debug_img, camera_matrix_, dist_coeffs_, rvec, tvec, 0.1);
	cv::imwrite("/home/pracsys/debug/handbase_marker_debug.png", _marker_debug_img);
	
	cv::Mat rot_mat;
	cv::Rodrigues(rvec,rot_mat);  // double
	Eigen::Matrix3f rot;   // rot from marker frame to cam
	rot<<rot_mat.at<double>(0,0),rot_mat.at<double>(0,1),rot_mat.at<double>(0,2),
		rot_mat.at<double>(1,0),rot_mat.at<double>(1,1),rot_mat.at<double>(1,2),
		rot_mat.at<double>(2,0),rot_mat.at<double>(2,1),rot_mat.at<double>(2,2);

	_handbase2cam.setIdentity();
	_handbase2cam.block(0,0,3,3)=rot;
	_handbase2cam.block(0,3,3,1)=Eigen::Vector3f(tvec[0],tvec[1],tvec[2]);

	std::cout<<std::setprecision(16)<<"Recorded _handbase2cam:\n"<<_handbase2cam<<"\n\n";
	return true;
}

void arucoBoardDetection(const ConfigParser &cfg, tf::TransformBroadcaster &_br)
{
	if (_handbase2cam==Eigen::Matrix4f::Identity())
	{
		ROS_ERROR("hand base pose not recorded! Please do it first!!");
		return;
	}
    _marker_debug_img = _rgb_image.clone();
	cv::Ptr<cv::aruco::DetectorParameters> marker_params = cv::aruco::DetectorParameters::create();	
	marker_params->minDistanceToBorder = cfg.minDistanceToBorder;
	marker_params->adaptiveThreshWinSizeMax = cfg.adaptiveThreshWinSizeMax;
	marker_params->adaptiveThreshWinSizeStep = cfg.adaptiveThreshWinSizeStep;
	marker_params->minCornerDistanceRate = cfg.minCornerDistanceRate;
	marker_params->minMarkerPerimeterRate = cfg.minMarkerPerimeterRate;
	marker_params->minOtsuStdDev = cfg.minOtsuStdDev;
	marker_params->perspectiveRemoveIgnoredMarginPerCell = cfg.perspectiveRemoveIgnoredMarginPerCell;
	marker_params->maxErroneousBitsInBorderRate = cfg.maxErroneousBitsInBorderRate;
	marker_params->errorCorrectionRate = cfg.errorCorrectionRate;
	marker_params->cornerRefinementWinSize = cfg.cornerRefinementWinSize;
	marker_params->cornerRefinementMaxIterations = cfg.cornerRefinementMaxIterations;
	marker_params->cornerRefinementMinAccuracy = cfg.cornerRefinementMinAccuracy;
	marker_params->cornerRefinementMethod = cfg.cornerRefinementMethod; 


	cv::Ptr<cv::aruco::Dictionary> dictionary_ = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
	std::vector<std::vector<cv::Point2f>> marker_corners_, rejected_points;
	std::vector<int> marker_ids;
	cv::aruco::detectMarkers(_rgb_image, dictionary_, marker_corners_, marker_ids, marker_params, rejected_points);
	
	if (marker_ids.size()==0)
	{
		ROS_ERROR("board not detected");
        return;
	}

	int start_id=-1;
	for (auto id:marker_ids)
	{
		if (id==0 || id==2 || id==4 || id==6)
		{
			start_id=id;
			break;
		}
	}

	if (start_id==-1)
	{
		ROS_ERROR("board not detected");
        return;
	}
	_detected_id=start_id;

	cv::Ptr<cv::aruco::GridBoard> board;
	std::cout<<"detected board start id="<<start_id<<"\n";
	board = cv::aruco::GridBoard::create(2,1, 0.025, 0.005, dictionary_, start_id);
	cv::aruco::refineDetectedMarkers(_rgb_image, board, marker_corners_, marker_ids, rejected_points);

	cv::Mat camera_matrix_ = (cv::Mat1d(3, 3) <<cfg.cam_intrinsic(0,0),cfg.cam_intrinsic(0,1),cfg.cam_intrinsic(0,2),
												cfg.cam_intrinsic(1,0),cfg.cam_intrinsic(1,1),cfg.cam_intrinsic(1,2),
												cfg.cam_intrinsic(2,0),cfg.cam_intrinsic(2,1),cfg.cam_intrinsic(2,2));
	cv::Mat dist_coeffs_ = (cv::Mat1d(4,1)<<0,0,0,0);		

	cv::Vec3d rvec, tvec;
	cv::aruco::estimatePoseBoard(marker_corners_,marker_ids, board, camera_matrix_, dist_coeffs_, rvec, tvec);
	cv::aruco::drawDetectedMarkers(_marker_debug_img, marker_corners_, marker_ids);
	cv::aruco::drawAxis(_marker_debug_img, camera_matrix_, dist_coeffs_, rvec, tvec, 0.1);

	cv::Mat rot_mat;
	cv::Rodrigues(rvec,rot_mat);  // double
	Eigen::Matrix3f rot;   // rot from marker frame to cam
	rot<<rot_mat.at<double>(0,0),rot_mat.at<double>(0,1),rot_mat.at<double>(0,2),
		rot_mat.at<double>(1,0),rot_mat.at<double>(1,1),rot_mat.at<double>(1,2),
		rot_mat.at<double>(2,0),rot_mat.at<double>(2,1),rot_mat.at<double>(2,2);

	_marker2cam.setIdentity();
	_marker2cam.block(0,0,3,3)=rot;
	_marker2cam.block(0,3,3,1)=Eigen::Vector3f(tvec[0],tvec[1],tvec[2]);

	_marker2handbase = _handbase2cam.inverse() * _marker2cam;

}

int main(int argc, char **argv)
{
    ros::init(argc,argv,"multiface");
    ConfigParser cfg("/home/pracsys/catkin_ws/src/marker_tf/config.yaml");
    
    std::shared_ptr<ros::NodeHandle> nh(new ros::NodeHandle);
    ros::ServiceServer _record_hand_pose_server = nh->advertiseService<bowen_pose_estimate::recordHandPose::Request,
			bowen_pose_estimate::recordHandPose::Response>("record_hand_pose", boost::bind(&handbaseRecord,cfg,_1,_2));
    ros::Subscriber _rgb_subscriber;
    _rgb_subscriber = nh->subscribe<sensor_msgs::Image>(cfg.rgb_topic,1,boost::bind(&rgbCallback,_1));
    ros::Publisher _marker_publisher = nh->advertise<sensor_msgs::Image>("marker_debug",1);
    ros::Publisher pose_publisher = nh->advertise<bowen_pose_estimate::multifacePose>("marker_pose",1);
    tf::TransformBroadcaster _br;

    ros::Rate r(10);
    while (ros::ok())
	{
		if (_rgb_image.rows>0) 
		{
			arucoBoardDetection(cfg,_br);   //uncomment this for calibration

			cv_bridge::CvImage out_msg;
			out_msg.encoding = sensor_msgs::image_encodings::BGR8; // Or whatever
			out_msg.image    = _marker_debug_img; // Your cv::Mat
			_marker_publisher.publish(out_msg);

			bowen_pose_estimate::multifacePose msg;
			msg.pose.position.x = _marker2handbase(0,3);  // in global frame
			msg.pose.position.y = _marker2handbase(1,3);
			msg.pose.position.z = _marker2handbase(2,3);
			Eigen::Matrix3f R=_marker2handbase.block(0,0,3,3);
			Eigen::Quaternionf q(R);
			msg.pose.orientation.x = q.x();
			msg.pose.orientation.y = q.y();
			msg.pose.orientation.z = q.z();
			msg.pose.orientation.w = q.w();
			msg.id.data = _detected_id;
			pose_publisher.publish(msg);
		}

		ros::spinOnce();
		r.sleep();
	}
}

