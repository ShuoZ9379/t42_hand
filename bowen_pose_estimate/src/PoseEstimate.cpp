#include "PoseEstimate.h"
#include <ctime>
#include <fstream>
#include <opencv2/rgbd.hpp>

PoseEstimate::PoseEstimate()
{
}

PoseEstimate::PoseEstimate(ConfigParser *cfg1) : cfg(cfg1)
{
}

PoseEstimate::PoseEstimate(std::shared_ptr<ros::NodeHandle> nh, ConfigParser *cfg1) : _nh(nh), cfg(cfg1),_is_drop(false)
{
	_finger_marker_ids = {1,2,3,4};
	_finger_marker_sizes = {0.012954,0.012954,0.012954,0.012954};
	_recorded_hand_pose = true;
	image_count = 0;
	_miss_detect_cnt = 0;
	_bad_z = -1;
	_dense_raw_cloud = boost::make_shared<PointCloudRGB>();
	_dense_raw_cloud1 = boost::make_shared<PointCloudRGB>();
	_cam2handmarker.setIdentity();
	_cam2handmarker1.setIdentity();
	_recorded_hand2cam = cfg->recorded_hand2cam;
	_cam2global =  cfg->cam_in_base;
	_cam2global1.setIdentity();
	_cylinder_center.setZero();
	_cylinder_center1.setZero();
	_handmarker_center<<cfg->recorded_hand2cam(0,3),cfg->recorded_hand2cam(1,3),cfg->recorded_hand2cam(2,3),1;
	_handmarker_center = _cam2global * _handmarker_center;
	num_image = 0;
	// _pose_server = _nh->advertiseService("get_pose", &PoseEstimate::poseService, this);
	_record_hand_pose_server = _nh->advertiseService("record_hand_pose", &PoseEstimate::handPoseRecordService, this);
	_cy_publisher = nh->advertise<geometry_msgs::Pose>("cylinder_pose",2);
	_cy_corner_publisher = nh->advertise<geometry_msgs::Pose>("cylinder_corner",2);
	_drop_publisher = nh->advertise<std_msgs::Bool>("cylinder_drop",2);
	_marker_publisher = nh->advertise<sensor_msgs::Image>("marker_debug",2);
	_finger_markers_pub = nh->advertise<geometry_msgs::PoseArray>("finger_markers",2);
	_marker2d_pub = nh->advertise<geometry_msgs::Pose>("marker2D",2);
	
	_rgb_subscriber = nh->subscribe<sensor_msgs::Image>(cfg->rgb_topic,10,boost::bind(&PoseEstimate::rgbCallback,this,_1));
	_depth_subscriber = nh->subscribe<sensor_msgs::Image>(cfg->depth_topic,10,boost::bind(&PoseEstimate::depthCallback,this,_1));
	
	ros::Rate r(10);
	
	while (ros::ok())
	{
		_is_drop=false;
		if (_rgb_image.rows>0) 
		{
			_marker_debug_img = _rgb_image.clone();
			std::clock_t begin = clock();
			bool detected = markerPose();
			std::cout<<"marker pose overall takes "<<double(clock()-begin)/CLOCKS_PER_SEC<<"\n";

			/*
			//bool detected1 = markerPose1(_rgb_image1);
			std::cout<<"detected/detected1: "<<detected<<" "<<detected1<<std::endl;
			if (detected && detected1)
			{
				std::cout<< "cylinder dist between 2 cam: "<<(_cylinder_center-_cylinder_center1).norm()<<std::endl;
				//exit(1);
			}
			*/

			if (detected)  
			{
				if (_cylinder_center(2)-_handmarker_center(2)>-0.01 && fabs(_cylinder_center(2)-_handmarker_center(2)-_bad_z)>1e-6)
					_miss_detect_cnt = 0;
				geometry_msgs::Pose msg;
				msg.position.x = _cylinder_center(0)-_handmarker_center(0);  // in global frame
				msg.position.y = _cylinder_center(1)-_handmarker_center(1);
				msg.position.z = _cylinder_center(2)-_handmarker_center(2);
				msg.orientation.x = 0;
				msg.orientation.y = 0;
				msg.orientation.z = 0;
				msg.orientation.w = 1;
				_cy_publisher.publish(msg);
				_last_msg = msg;

				geometry_msgs::Pose msg1;
				msg1.position.x = _cylinder_corner0(0)-_handmarker_center(0);  // in global frame
				msg1.position.y = _cylinder_corner0(1)-_handmarker_center(1);
				msg1.position.z = _cylinder_corner0(2)-_handmarker_center(2);
				_cy_corner_publisher.publish(msg1);
				_cylinder_corner0_last = _cylinder_corner0;

				geometry_msgs::PoseArray msg2;
				for (int i=0;i<_fingermarkers_in_global.size();i++)
				{
					Eigen::Vector4f marker_pos = _fingermarkers_in_global[i];
					geometry_msgs::Pose tmp;
					if (marker_pos.norm()==4)   // not valid
					{
						marker_pos = _fingermarkers_in_global_last[i];
					}
	
					tmp.position.x = marker_pos(0)-_handmarker_center(0);  // in global frame
					tmp.position.y = marker_pos(1)-_handmarker_center(1);
					tmp.position.z = marker_pos(2)-_handmarker_center(2);
					
					tmp.orientation.x = 0;
					tmp.orientation.y = 0;
					tmp.orientation.z = 0;
					tmp.orientation.w = 1;
					msg2.poses.push_back(tmp);
				}
				_finger_markers_pub.publish(msg2);
			}			
			else
			{
				_miss_detect_cnt++;
				geometry_msgs::Pose msg;
				msg.position.x = _last_msg.position.x;
				msg.position.y = _last_msg.position.y;
				msg.position.z = _last_msg.position.z;
				msg.orientation.x = 0;
				msg.orientation.y = 0;
				msg.orientation.z = 0;
				msg.orientation.w = 1;
				_cy_publisher.publish(msg);

				geometry_msgs::Pose msg1;
				msg1.position.x = _cylinder_corner0_last(0)-_handmarker_center(0);  // in global frame
				msg1.position.y = _cylinder_corner0_last(1)-_handmarker_center(1);
				msg1.position.z = _cylinder_corner0_last(2)-_handmarker_center(2);
				_cy_corner_publisher.publish(msg1);
				
			}
			cv_bridge::CvImage out_msg;
			out_msg.encoding = sensor_msgs::image_encodings::BGR8; // Or whatever
			out_msg.image    = _marker_debug_img; // Your cv::Mat
			_marker_publisher.publish(out_msg);
			std_msgs::Bool is_drop;
			if (detected && _cylinder_center(2)-_handmarker_center(2)<-0.05)
			{
				std::cout<<"detected: "<<detected<<"   _cylinder_center(2)-_handmarker_center(2): "<<_cylinder_center(2)-_handmarker_center(2)<<std::endl;
				_is_drop=true;
				
			}
			else if (_miss_detect_cnt>=3)
			{
				std::cout<<"_miss_detect_cnt: "<<_miss_detect_cnt<<std::endl;
				_is_drop=true;
			}
			is_drop.data = _is_drop;
			_drop_publisher.publish(is_drop);
		}
		// publish motoman base cam to baselink
		tf::Transform transform;
		transform.setOrigin( tf::Vector3(0.524604320526123, -0.02932164259254932, 0.961784303188324) );
		// xyzw
		tf::Quaternion q(-0.6771612763404846, 0.724048912525177, -0.09155406057834625, 0.09393471479415894);
		transform.setRotation(q);
		tf::TransformBroadcaster br;
		br.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "base_link", "camera_color_optical_frame"));
		std::cout<<"sent transform\n";


		ros::spinOnce();
		r.sleep();
	}
}

PoseEstimate::~PoseEstimate()
{
}

void PoseEstimate::rgbCallback(const sensor_msgs::Image::ConstPtr &msg)
{
	cv_bridge::CvImagePtr bridge_ptr = cv_bridge::toCvCopy(msg, "bgr8");
	_rgb_image = bridge_ptr->image; //NOTE overwritting server _rgb_image
}

void PoseEstimate::depthCallback(const sensor_msgs::Image::ConstPtr &msg)
{
	Utils::rawDepthToMeters(msg, _depth_image);
}

void PoseEstimate::subscribeMsg()
{
	sensor_msgs::Image::ConstPtr msg_color;
	sensor_msgs::Image::ConstPtr msg_depth;
	msg_color = ros::topic::waitForMessage<sensor_msgs::Image>(cfg->rgb_topic, *_nh);
	msg_depth = ros::topic::waitForMessage<sensor_msgs::Image>(cfg->depth_topic, *_nh);
	Utils::rawDepthToMeters(msg_depth, _depth_image);
	cv_bridge::CvImagePtr bridge_ptr = cv_bridge::toCvCopy(msg_color, "bgr8"); // BGR 8UC3
	_rgb_image = bridge_ptr->image;
}


// Camera1 subscribe
void PoseEstimate::subscribeMsg1()
{
	sensor_msgs::Image::ConstPtr msg_color;
	sensor_msgs::Image::ConstPtr msg_depth;
	msg_color = ros::topic::waitForMessage<sensor_msgs::Image>(cfg->rgb_topic1, *_nh);
	msg_depth = ros::topic::waitForMessage<sensor_msgs::Image>(cfg->depth_topic1, *_nh);
	Utils::rawDepthToMeters(msg_depth, _depth_image1);
	cv_bridge::CvImagePtr bridge_ptr = cv_bridge::toCvCopy(msg_color, "bgr8"); // BGR 8UC3
	_rgb_image1 = bridge_ptr->image;
}

// Do this offline
bool PoseEstimate::handPoseRecordService(bowen_pose_estimate::recordHandPose::Request &req, bowen_pose_estimate::recordHandPose::Response &res)
{
	
	// _recorded_hand2cam = _cam2handmarker.inverse();   // record hand marker
	// if (_cam2handmarker.isIdentity(1e-15))
	// {
	// 	std::cout<<"[warn] cam2handmarker is Identity, error might occur\n";
	// 	res.info.data = "[warn] cam2handmarker is Identity, error might occur";		
	// 	return false;
	// }
	// else
	// {
	// 	res.info.data = "Recorded hand pose";
	// 	res.pose.position.x = _recorded_hand2cam(0,3);
	// 	res.pose.position.y = _recorded_hand2cam(1,3);
	// 	res.pose.position.z = _recorded_hand2cam(2,3);
	// 	Eigen::Matrix3f R = _recorded_hand2cam.block(0,0,3,3);
	// 	Eigen::Quaternionf q(R);
	// 	res.pose.orientation.w = q.w();
	// 	res.pose.orientation.x = q.x();
	// 	res.pose.orientation.y = q.y();
	// 	res.pose.orientation.z = q.z();

	// 	std::cout<<"_recorded_hand2cam:\n"<<_recorded_hand2cam<<"\n\n";
	// 	exit(1);
	// }


	_recorded_hand_pose = true;

	
	return true;

}


bool PoseEstimate::poseService(bowen_pose_estimate::getpose::Request &req, bowen_pose_estimate::getpose::Response &res)
{
	subscribeMsg();
	// Utils::writeDepthImage(_depth_image, "/home/pracsys/tmp/depth_image" + std::to_string(num_image) + ".png");
	// cv::imwrite("/home/pracsys/tmp/rgb_image" + std::to_string(num_image) + ".png", _rgb_image);
	// num_image++;

	/*
	//CHECK
	_rgb_image = cv::imread("/home/pracsys/debug/rgb_image.png");
	Utils::readDepthImage(_depth_image,"/home/pracsys/debug/depth_image.png");
	PointCloudRGB::Ptr cloud(new PointCloudRGB);
	Utils::convert3dOrganizedRGB(_depth_image,_rgb_image,cfg->cam_intrinsic,cloud);
	pcl::io::savePLYFile("/home/pracsys/debug/cloudrgb.ply",*cloud);
*/

	PointCloudRGB::Ptr cloud(new PointCloudRGB);
	Utils::convert3dOrganizedRGB(_depth_image,_rgb_image,cfg->cam_intrinsic,cloud);
	//pcl::io::savePLYFile("/home/pracsys/debug/raw_cloud.ply", *cloud);
	_dense_raw_cloud->clear();
	pcl::copyPointCloud(*cloud, *_dense_raw_cloud);
	
	getCylinderInBowl(cloud);
	Eigen::Matrix4f cy2world = computePose(_cloud_filtered_sparse);
	
	if (_cam2handmarker.isIdentity(1e-15))
	{
		std::cout<<"[WARN] cam2handmarker is Identity\n";
	}
	// Eigen::Matrix4f cy2world = cfg->endeffector2global * 
	// 	cfg->handmarker2endeffector*_recorded_hand2cam.inverse() * cy2cam;  // cylinder pose relative to the last recorded hand pose, not current
	
	geometry_msgs::Pose msg;
	msg.position.x = cy2world(0,3); 
	msg.position.y = cy2world(1,3);
	msg.position.z = cy2world(2,3);
	Eigen::Matrix3f R = cy2world.block(0,0,3,3);
	// Eigen::Matrix3f R_tmp;
	// R_tmp<<0,-1,0,
	// 	   1,0,0,
	// 	   0,0,1;
	// R = R_tmp * R;
	Eigen::Quaternionf q(R);
	msg.orientation.x = q.x();
	msg.orientation.y = q.y();
	msg.orientation.z = q.z();
	msg.orientation.w = q.w();
	res.pose = msg;

	return true;
}

// compute cylinder pose
Eigen::Matrix4f PoseEstimate::computePose(PointCloudRGB::Ptr cylinder_cloud)
{
	pcl::PointXYZRGB minPt, maxPt;
	
	pcl::getMinMax3D (*cylinder_cloud, minPt, maxPt);
	float center_x = (maxPt.x+minPt.x)/2.0, 
		  center_y = (maxPt.y+minPt.y)/2.0;

	pcl::PCA<pcl::PointXYZRGB> pca = new pcl::PCA<pcl::PointXYZRGB>;
	pca.setInputCloud(cylinder_cloud);
	// Eigen::Vector4f mean = pca.getMean();
	// std::cout << "mean is: \n"
	// 					<< mean << std::endl;

	// Eigen::Vector3f eigvalue = pca.getEigenValues();
	// std::cout<<"eigen value is \n"<<eigvalue<<std::endl;

	Eigen::Matrix3f eigvec = pca.getEigenVectors();
	Eigen::Vector3f direction = eigvec.col(0);
	std::cout << "direction is \n"
						<< direction << std::endl;

	std::vector<float> pose = {center_x, center_y, 0, direction(0), direction(1), direction(2)};
	Eigen::Quaternionf q;
	Utils::directionToQuaternion(pose.data()+3, q);
	std::cout<<"quaternion is: "<<q<<"\n";
	Eigen::Matrix4f cy2world;
	cy2world.setIdentity();
	cy2world.block(0,0,3,3) = q.toRotationMatrix();
	cy2world(0,3) = pose[0];
	cy2world(1,3) = pose[1];
	cy2world(2,3) = pose[2];
	return cy2world;
}

void PoseEstimate::cylinderFittingWithNormal(PointCloudRGB::Ptr cloud_in)
{
	pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGB>());
	pcl::NormalEstimation<pcl::PointXYZRGB, pcl::Normal> ne;
	ne.setSearchMethod(tree);
	ne.setInputCloud(cloud_in);
	ne.setKSearch(cfg->k_search);
	pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>);
	ne.compute(*cloud_normals);

	pcl::SACSegmentationFromNormals<pcl::PointXYZRGB, pcl::Normal> seg;
	seg.setOptimizeCoefficients(true);
	seg.setModelType(pcl::SACMODEL_CYLINDER);
	seg.setMethodType(pcl::SAC_MSAC);
	seg.setNormalDistanceWeight(cfg->normal_distance_weight);
	seg.setMaxIterations(cfg->max_iterations);
	seg.setDistanceThreshold(cfg->cylinder_distance_threshold);
	seg.setRadiusLimits(cfg->radius_limits_min, cfg->radius_limits_max);
	seg.setInputCloud(cloud_in);
	seg.setInputNormals(cloud_normals);
	pcl::PointIndices::Ptr inliers_cylinder(new pcl::PointIndices);
	pcl::ModelCoefficients::Ptr coefficients_cylinder(new pcl::ModelCoefficients);
	seg.segment(*inliers_cylinder, *coefficients_cylinder);

	pcl::ExtractIndices<pcl::PointXYZRGB> extract;
	extract.setInputCloud(cloud_in);
	extract.setIndices(inliers_cylinder);
	extract.setNegative(false);
	PointCloudRGB::Ptr cloud_cylinder(new PointCloudRGB);
	extract.filter(*cloud_cylinder);
	pcl::io::savePLYFile("/home/pracsys/debug/cloud_cylinder.ply", *cloud_cylinder);
}

void PoseEstimate::removeSurface(PointCloudRGB::Ptr cloud)
{

	pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
	pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
	pcl::SACSegmentation<pcl::PointXYZRGB> seg;

	// downsample
	pcl::VoxelGrid<pcl::PointXYZRGB> vox;
	vox.setInputCloud(cloud);
	vox.setLeafSize(cfg->leaf_size, cfg->leaf_size, cfg->leaf_size);
	_cloud_filtered_sparse = boost::make_shared<PointCloudRGB>();
	vox.filter(*_cloud_filtered_sparse);
	pcl::io::savePLYFile("/home/pracsys/debug/downsample_cloud.ply", *_cloud_filtered_sparse);

	pcl::ExtractIndices<pcl::PointXYZRGB> extract;
	
	// get table inlier points
	seg.setOptimizeCoefficients(true);
	seg.setModelType(pcl::SACMODEL_PLANE);
	seg.setMethodType(pcl::SAC_MSAC);
	seg.setDistanceThreshold(cfg->distance_threshold);
	seg.setMaxIterations(cfg->max_iterations);
	seg.setInputCloud(_cloud_filtered_sparse);
	seg.segment(*inliers, *coefficients);
	std::cout<<"Point cloud clustering...\n";
	const float bad_point = 0;
	if (coefficients == NULL || inliers->indices.size() == 0)
	{
		std::cout << "No plane detected\n";
		return;
	}
	if (inliers->indices.size()>1000)
	{
		PointCloudRGB::Ptr cloud_p(new PointCloudRGB);
		
		extract.setInputCloud(_cloud_filtered_sparse);
		extract.setIndices(inliers);
		extract.setNegative(false);
		extract.filter(*cloud_p);
		pcl::io::savePLYFile("/home/pracsys/debug/table.ply", *cloud_p);
		
		for (int v = 0; v < cloud->height; v++)
		{
			for (int u = 0; u < cloud->width; u++)
			{
				auto pt = cloud->at(u, v);
				float dist = pcl::pointToPlaneDistance(pt, coefficients->values[0], coefficients->values[1], coefficients->values[2], coefficients->values[3]);
				if (dist < cfg->distance_to_table)
				{
					cloud->at(u, v).x = bad_point;
					cloud->at(u, v).y = bad_point;
					cloud->at(u, v).z = bad_point;
					_depth_image.at<float>(v, u) = 0;
				}
			}
		}
	
		pcl::io::savePLYFile("/home/pracsys/debug/remove_table_dense.ply", *cloud);

		for (auto &pt : _cloud_filtered_sparse->points)
		{
			float dist = pcl::pointToPlaneDistance(pt, coefficients->values[0], coefficients->values[1], coefficients->values[2], coefficients->values[3]);
			if (dist < cfg->distance_to_table)
			{
				pt.x = bad_point;
				pt.y = bad_point;
				pt.z = bad_point;
			}
		}
		
		pcl::io::savePLYFile("/home/pracsys/debug/remove_table.ply", *_cloud_filtered_sparse);
	}
	

	// remove noise
	PointCloudRGB::Ptr noise_free_cloud(new PointCloudRGB);

	/*
	pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> sor;
	sor.setInputCloud (cloud);
	sor.setMeanK (10);
	sor.setStddevMulThresh (0.01);
	sor.filter (*noise_free_cloud);
	pcl::io::savePLYFile("/home/pracsys/debug/noise_free_cloud.ply",*noise_free_cloud);
	*/

	pcl::RadiusOutlierRemoval<pcl::PointXYZRGB> outrem;
	outrem.setInputCloud(_cloud_filtered_sparse);
	outrem.setRadiusSearch(cfg->radius);
	outrem.setMinNeighborsInRadius(cfg->min_number);
	outrem.filter(*noise_free_cloud);
	pcl::io::savePLYFile("/home/pracsys/debug/noise_free_cloud.ply", *noise_free_cloud);
	if (noise_free_cloud->points.size()==0) 
	{
		std::cout<<"noise_free_cloud size = 0, return!!\n";
		return;
	}
	

	// Segmentation
	pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGB>);
	tree->setInputCloud (noise_free_cloud);
	std::vector<pcl::PointIndices> cluster_indices;
	pcl::EuclideanClusterExtraction<pcl::PointXYZRGB> ec;
	ec.setClusterTolerance (0.02); // 2cm
	ec.setMinClusterSize (100);
	ec.setMaxClusterSize (25000);
	ec.setSearchMethod (tree);
	ec.setInputCloud (noise_free_cloud);
	ec.extract (cluster_indices);
	int min_cluster_size = std::numeric_limits<int>::max();
	pcl::PointIndices cylinder_indices;
	pcl::PCA<pcl::PointXYZRGB> pca = new pcl::PCA<pcl::PointXYZRGB>;
	pcl::PointXYZRGB minPt, maxPt;
	std::cout<<"num of clusters: "<<cluster_indices.size()<<"\n\n";
	bool has_cylinder = false;
	for (int i=0;i<cluster_indices.size();i++)
	{
		PointCloudRGB::Ptr cloud_tmp(new PointCloudRGB);
		pcl::PointIndices indices = cluster_indices[i];
		extract.setInputCloud (noise_free_cloud);
		pcl::PointIndices::Ptr indices_ptr = boost::make_shared<pcl::PointIndices>(indices);
		extract.setIndices (indices_ptr);
		extract.setNegative (false);
		extract.filter (*cloud_tmp);
		pcl::transformPointCloud (*cloud_tmp, *_cloud_filtered_sparse, _cam2global);
		pcl::getMinMax3D (*_cloud_filtered_sparse, minPt, maxPt);
		std::cout<<"_cloud_filtered_sparse size is: "<<_cloud_filtered_sparse->points.size()<<std::endl;
		std::cout<<"min pt, max pt are:\n"<<minPt.x<<"\n"<<maxPt.x<<"\n\n";

		// Remove invalid points and bad points, check coordinates in global frame, endeffector is origin
		if (fabs(maxPt.x-minPt.x)>1e-15 && fabs(maxPt.y-minPt.y)>1e-15 && 
			fabs(maxPt.x-minPt.x)<0.2 && fabs(maxPt.y-minPt.y)<0.2 &&
			maxPt.z<-0.1 && minPt.z<-0.1
			)
		{
			has_cylinder = true;
			break;
		}
		
	}
	if (has_cylinder)  // cylinder on table
	{
		_is_drop = true;
	}
	else
	{
		_is_drop = false;
	}
	PointCloudRGB::Ptr cloud_tmp(new PointCloudRGB);
	pcl::copyPointCloud(*_cloud_filtered_sparse, *cloud_tmp);
	for (auto &pt:_cloud_filtered_sparse->points)
	{
		pt.z = 0;
	}
	pcl::io::savePLYFile("/home/pracsys/debug/cylinder.ply", *cloud_tmp);
	std::cout<<"_cloud_filtered_sparse size is: "<<_cloud_filtered_sparse->points.size()<<"\n";
	

	// noise_free_cloud->swap(*_cloud_filtered_sparse);
}



void PoseEstimate::getCylinderInBowl(PointCloudRGB::Ptr cloud)
{
	_cloud_filtered_sparse = boost::make_shared<PointCloudRGB>();
	pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
	pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
	pcl::SACSegmentation<pcl::PointXYZRGB> seg;

	pcl::ExtractIndices<pcl::PointXYZRGB> extract;
	
	PointCloudRGB::Ptr cloud_global(new PointCloudRGB);	
	pcl::transformPointCloud (*cloud, *cloud_global, _cam2global);
	pcl::io::savePLYFile("/home/pracsys/debug/downsample_cloud_global.ply", *cloud_global);
	pcl::copyPointCloud(*cloud_global, *_cloud_filtered_sparse);

/*
	// downsample
	pcl::VoxelGrid<pcl::PointXYZRGB> vox;
	vox.setInputCloud(_cloud_filtered_sparse);
	vox.setLeafSize(cfg->leaf_size, cfg->leaf_size, cfg->leaf_size);
	vox.filter(*_cloud_filtered_sparse);
	pcl::io::savePLYFile("/home/pracsys/debug/downsample_cloud.ply", *_cloud_filtered_sparse);
*/
	
	PointCloudRGB::Ptr cloud_tmp(new PointCloudRGB);
	for (auto pt:_cloud_filtered_sparse->points)
	{
		if (pt.z<-0.2 && pt.z>-0.22 && pt.y<0.17 && pt.y>0.055 && pt.x>0.005 && pt.x<0.11)
		{
			cloud_tmp->points.push_back(pt);
		}
	}	
	pcl::copyPointCloud(*cloud_tmp,*_cloud_filtered_sparse);
	pcl::io::savePLYFile("/home/pracsys/debug/bowl_bottom.ply", *_cloud_filtered_sparse);

	// Segmentation
	pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGB>);
	tree->setInputCloud (_cloud_filtered_sparse);
	std::vector<pcl::PointIndices> cluster_indices;
	pcl::EuclideanClusterExtraction<pcl::PointXYZRGB> ec;
	ec.setClusterTolerance (0.01); // 1cm
	ec.setMinClusterSize (80);
	ec.setMaxClusterSize (25000);
	ec.setSearchMethod (tree);
	ec.setInputCloud (_cloud_filtered_sparse);
	ec.extract (cluster_indices);
	pcl::PointXYZRGB minPt, maxPt;
	std::cout<<"num of clusters: "<<cluster_indices.size()<<"\n\n";
	for (int i=0;i<cluster_indices.size();i++)
	{
		PointCloudRGB::Ptr cloud_tmp(new PointCloudRGB);
		pcl::PointIndices indices = cluster_indices[i];
		extract.setInputCloud (_cloud_filtered_sparse);
		pcl::PointIndices::Ptr indices_ptr = boost::make_shared<pcl::PointIndices>(indices);
		extract.setIndices (indices_ptr);
		extract.setNegative (false);
		extract.filter (*cloud_tmp);
		pcl::getMinMax3D (*cloud_tmp, minPt, maxPt);
		std::cout<<"_cloud_filtered_sparse size is: "<<cloud_tmp->points.size()<<std::endl;
		std::cout<<"min pt, max pt are:\n"<<minPt.x<<"\n"<<maxPt.x<<"\n\n";

		// Remove invalid points and bad points, check coordinates in global frame, endeffector is origin
		if (fabs(maxPt.x-minPt.x)>1e-15 && fabs(maxPt.y-minPt.y)>1e-15 && 
			fabs(maxPt.x-minPt.x)<0.1 && fabs(maxPt.y-minPt.y)<0.1 
			)
		{
			pcl::copyPointCloud(*cloud_tmp, *_cloud_filtered_sparse);
			break;
		}
		
	}
	
	// remove noise
	pcl::RadiusOutlierRemoval<pcl::PointXYZRGB> outrem;
	outrem.setInputCloud(_cloud_filtered_sparse);
	outrem.setRadiusSearch(cfg->radius);
	outrem.setMinNeighborsInRadius(cfg->min_number);
	outrem.filter(*_cloud_filtered_sparse);
	// pcl::io::savePLYFile("/home/pracsys/debug/noise_free_cloud.ply", *_cloud_filtered_sparse);
	if (_cloud_filtered_sparse->points.size()==0) 
	{
		std::cout<<"noise_free_cloud size = 0, return!!\n";
		return;
	}
	for (auto &pt:_cloud_filtered_sparse->points)
	{
		pt.z = 0;
	}
		

}

bool PoseEstimate::markerPose()
{
	_dense_raw_cloud->clear();
	std::clock_t beg = clock();

	Utils::convert3dOrganizedRGB(_depth_image,_rgb_image,cfg->cam_intrinsic,_dense_raw_cloud);
	// pcl::io::savePLYFile("/home/pracsys/debug/raw_scene.ply", *_dense_raw_cloud);
	// std::cout<<"_dense_raw_cloud size is "<<_dense_raw_cloud->points.size()<<"\n";
	if (_dense_raw_cloud->points.size()==0) 
	{
		ROS_ERROR("[WARN] cloud is empty!");
		return false;
	}

	std::cout<<"convert cloud takes: "<<double(clock()-beg)/CLOCKS_PER_SEC<<"\n";

	cv::Ptr<cv::aruco::DetectorParameters> marker_params = cv::aruco::DetectorParameters::create();
	
	
	marker_params->minDistanceToBorder = cfg->minDistanceToBorder;
	marker_params->adaptiveThreshWinSizeMax = cfg->adaptiveThreshWinSizeMax;
	marker_params->adaptiveThreshWinSizeStep = cfg->adaptiveThreshWinSizeStep;
	marker_params->minCornerDistanceRate = cfg->minCornerDistanceRate;
	marker_params->minMarkerPerimeterRate = cfg->minMarkerPerimeterRate;
	marker_params->minOtsuStdDev = cfg->minOtsuStdDev;
	marker_params->perspectiveRemoveIgnoredMarginPerCell = cfg->perspectiveRemoveIgnoredMarginPerCell;
	marker_params->maxErroneousBitsInBorderRate = cfg->maxErroneousBitsInBorderRate;
	marker_params->errorCorrectionRate = cfg->errorCorrectionRate;
	marker_params->cornerRefinementWinSize = cfg->cornerRefinementWinSize;
	marker_params->cornerRefinementMaxIterations = cfg->cornerRefinementMaxIterations;
	marker_params->cornerRefinementMinAccuracy = cfg->cornerRefinementMinAccuracy;
	marker_params->cornerRefinementMethod = cfg->cornerRefinementMethod; 
	
	
	// marker_params->markerBorderBits = cfg->markerBorderBits;
	// marker_params->aprilTagCriticalRad = cfg->aprilTagCriticalRad;
	// marker_params->aprilTagMinClusterPixels = cfg->aprilTagMinClusterPixels;
	// marker_params->perpectiveRemovePixelPerCell = cfg->perpectiveRemovePixelPerCell;



	cv::Ptr<cv::aruco::Dictionary> dictionary_ = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
	std::vector<std::vector<cv::Point2f>> marker_corners_, rejected_points;
	std::vector<int> marker_ids_;
	std::clock_t begin = clock();
	cv::aruco::detectMarkers(_rgb_image, dictionary_, marker_corners_, marker_ids_, marker_params, rejected_points);
	std::cout<<"detect marker used: "<<double(clock()-begin)/CLOCKS_PER_SEC<<"\n";


	//CHECK
	
	cv::aruco::drawDetectedMarkers(_marker_debug_img, marker_corners_, marker_ids_);
	// cv::imwrite("/home/pracsys/debug/markerdetect.png",_marker_debug_img);
	
	int base_marker_id = -1;
	int cylinder_marker_id = -1;
	std::vector<cv::Point2f> base_marker_corner, cylinder_marker_corner;
	std::vector<int> finger_marker_indices(_finger_marker_ids.size(), -1);
	for (int i=0;i<marker_ids_.size();i++)
	{
		if (marker_ids_[i]==0)
		{
			base_marker_corner = marker_corners_[i];
			base_marker_id = i;
		}
		else if (marker_ids_[i]==5)
		{
			cylinder_marker_id = i;
			cylinder_marker_corner = marker_corners_[i];
		}

		for (int j=0;j<_finger_marker_ids.size();j++)
		{
			if (_finger_marker_ids[j]==marker_ids_[i])
			{
				finger_marker_indices[j]=i;
			}
		}
	}

	//get finger markers
	_fingermarkers_in_global.clear();
	_fingermarkers_in_global.resize(_finger_marker_ids.size(), Eigen::VectorXf::Ones(4));
	for (int i=0;i<finger_marker_indices.size();i++)
	{
		int idx = finger_marker_indices[i];
		if (idx!=-1)
		{
			Eigen::Vector4f pose_tmp;
			pose_tmp<<0,0,0,1;
			processOneFingerMarker(marker_corners_,marker_ids_,idx,_finger_marker_sizes[i],pose_tmp);
			_fingermarkers_in_global[i] = pose_tmp;
			// std::cout<<"marker id: "<<i<<" pos: \n"<<pose_tmp<<"\n\n";
		}
	}

	_fingermarkers_in_global_last = _fingermarkers_in_global;


	if (marker_corners_.size()==0 || base_marker_id==-1 || cylinder_marker_id==-1)
	{
		std::cout<<"[WARNING] Not both marker detected !!\n\n";		
	}
	double base_marker_size_ = 0.012954,
		   cy_marker_size_ = 0.02;
	std::vector<cv::Vec3d> rvecs_, tvecs_;
	cv::Mat camera_matrix_ = (cv::Mat1d(3, 3) <<cfg->cam_intrinsic(0,0),cfg->cam_intrinsic(0,1),cfg->cam_intrinsic(0,2),
												cfg->cam_intrinsic(1,0),cfg->cam_intrinsic(1,1),cfg->cam_intrinsic(1,2),
												cfg->cam_intrinsic(2,0),cfg->cam_intrinsic(2,1),cfg->cam_intrinsic(2,2));
	cv::Mat dist_coeffs_ = (cv::Mat1d(4,1)	<<0,0,0,0);																		
	//std::cout<<"cam_intrinsic mat is \n"<<camera_matrix_<<"\n\n";		
	std::vector<std::vector<cv::Point2f>> tmp_corners;	
	cv::Mat rot_mat;										
	Eigen::Matrix4f hand2cam = Eigen::Matrix4f::Identity(4,4);     // hand marker frame to cam frame
	Eigen::Matrix4f cy2cam = Eigen::Matrix4f::Identity(4,4);      // cylinder frame to cam frame
	pcl::PointXYZRGB cylinder_center3D, handmarker_center3D, cylinder_corner0;
	

	if (base_marker_id!=-1)
	{
		std::clock_t begin = clock();
	
	
		rvecs_.clear();
		tvecs_.clear();
		tmp_corners.clear();
		tmp_corners.push_back(marker_corners_[base_marker_id]);
		handmarker_center3D = getMarkerCenter3D(tmp_corners);
		
		
		// cv::aruco::estimatePoseSingleMarkers(tmp_corners, base_marker_size_, camera_matrix_, dist_coeffs_, rvecs_, tvecs_);
		
		// cv::Vec3d rvec = rvecs_[0], tvec = tvecs_[0]; 
		// cv::Mat rot_mat;	
		// cv::Rodrigues(rvec,rot_mat);  // double

		// Eigen::Matrix3f rot;   // rot from marker frame to cam
		// rot<<rot_mat.at<double>(0,0),rot_mat.at<double>(0,1),rot_mat.at<double>(0,2),
		// 	rot_mat.at<double>(1,0),rot_mat.at<double>(1,1),rot_mat.at<double>(1,2),
		// 	rot_mat.at<double>(2,0),rot_mat.at<double>(2,1),rot_mat.at<double>(2,2);
		// Eigen::Quaternionf q_eigen(rot);
		// tf::Transform transform;
		// transform.setOrigin( tf::Vector3(tvec[0], tvec[1], tvec[2]) );
		// tf::Quaternion q(q_eigen.x(), q_eigen.y(), q_eigen.z(), q_eigen.w());
		// transform.setRotation(q);
		// _hand2cam_br.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "camera_link", "camera_marker"));
		
		// for (int i=0;i<rvecs_.size();i++)
		// {
		// 	cv::aruco::drawAxis(_marker_debug_img, camera_matrix_,dist_coeffs_,rvecs_[i],tvecs_[i],0.01);
		// }
		// cv::imwrite("/home/pracsys/debug/base_marker.png", _marker_debug_img);

		// hand2cam<<rot_mat.at<double>(0,0),rot_mat.at<double>(0,1),rot_mat.at<double>(0,2),tvec[0],    
		// 		rot_mat.at<double>(1,0),rot_mat.at<double>(1,1),rot_mat.at<double>(1,2),tvec[1],
		// 		rot_mat.at<double>(2,0),rot_mat.at<double>(2,1),rot_mat.at<double>(2,2),tvec[2],
		// 		0,0,0,1;
		// _cam2handmarker = hand2cam.inverse(); 
		

	}

	// if (cylinder_marker_id==-1)
	// {
	// 	_is_drop = true;
	// }
	
	if (cylinder_marker_id!=-1)
	{
		rvecs_.clear();
		tvecs_.clear();
		tmp_corners.clear();
		tmp_corners.push_back(marker_corners_[cylinder_marker_id]);
		cylinder_center3D = getMarkerCenter3D(tmp_corners);
		if (cylinder_center3D.z<0.1 || cylinder_center3D.z>2.0)
		{
			ROS_ERROR("get center3D failed");
			return false;
		}

		//Get marker 2D center 
		std::vector<cv::Point2f> cylinder_marker_corners = marker_corners_[cylinder_marker_id];
		if (cylinder_marker_corners.size()!=4)
		{
			ROS_ERROR("cylinder marker corner size is not 4!");
		}
		float u_center=0, v_center=0; 
		for (int i=0;i<4;i++)
		{
			auto corner = cylinder_marker_corners[i];
			u_center+=corner.x;
			v_center+=corner.y;
		}
		u_center/=4.0;
		v_center/=4.0;
		geometry_msgs::Pose msg;
		msg.position.x = u_center; 
		msg.position.y = v_center;
		_marker2d_pub.publish(msg);

		cylinder_corner0 = getPix3D(marker_corners_[cylinder_marker_id][0].x, marker_corners_[cylinder_marker_id][0].y);
		
		_cylinder_corner0<<cylinder_corner0.x,cylinder_corner0.y,cylinder_corner0.z,1;
		// std::cout<<"_cylinder_corner0:\n"<<_cylinder_corner0<<"\n\n";
		_cylinder_corner0 = _cam2global *  _cylinder_corner0;
		
		cv::aruco::estimatePoseSingleMarkers(tmp_corners, cy_marker_size_, camera_matrix_, dist_coeffs_, rvecs_, tvecs_);
		
		cv::Vec3d cyrvec = rvecs_[0], cytvec = tvecs_[0];
		
		cv::Rodrigues(cyrvec,rot_mat);  

		/*
		// Correct rotation by point cloud normal
		if (_dense_raw_cloud->points.size()>0)
		{
			Eigen::Matrix3f rot;   // marker to cam
			rot<<rot_mat.at<double>(0,0),rot_mat.at<double>(0,1),rot_mat.at<double>(0,2)
			,rot_mat.at<double>(1,0),rot_mat.at<double>(1,1),rot_mat.at<double>(1,2)
			,rot_mat.at<double>(2,0),rot_mat.at<double>(2,1),rot_mat.at<double>(2,2);
			Eigen::Vector3f normal = markerNormal(tmp_corners);  // normal in cam frame
			normal = rot.transpose() * normal;  // normal in marker frame
			Eigen::Quaternionf q;
			q.setFromTwoVectors(Eigen::Vector3f(0,0,1),normal);
			q.normalize();
			Eigen::Matrix3f rot_offset = q.toRotationMatrix();
			std::cout<<"In cylinder rot_offset is \n"<<rot_offset<<"\n\n";
			rot =  rot * rot_offset;
			std::cout<<"In cylinder rot is\n"<<rot<<"\n\n";
			Utils::EigenMatTocvMat(rot,rot_mat);
			cv::Rodrigues(rot_mat,cyrvec);  // double
			rvecs_.clear();
			rvecs_.push_back(cyrvec);
		}
		*/

		for (int i=0;i<rvecs_.size();i++)
		{
			cv::aruco::drawAxis(_marker_debug_img, camera_matrix_,dist_coeffs_,rvecs_[i],tvecs_[i],0.01);
		}

		cy2cam<<rot_mat.at<double>(0,0),rot_mat.at<double>(0,1),rot_mat.at<double>(0,2),cytvec[0],    
				rot_mat.at<double>(1,0),rot_mat.at<double>(1,1),rot_mat.at<double>(1,2),cytvec[1],
				rot_mat.at<double>(2,0),rot_mat.at<double>(2,1),rot_mat.at<double>(2,2),cytvec[2],
				0,0,0,1;
		// std::ofstream ff("/home/pracsys/debug/cylinder2cam.txt");
		// ff<<cy2cam<<std::endl;
		// ff.close();
		//std::cout<<"cy2cam is \n"<<cy2cam<<"\n\n";				
	}
	// cv::imwrite("/home/pracsys/debug/markerdetect.png",_marker_debug_img);
	
	if (cylinder_marker_id!=-1)
	{
		// std::cout<<"cylinder_center3D is "<<cylinder_center3D.x<<" "<<cylinder_center3D.y<<" "<<cylinder_center3D.z<<"\n";
		_cylinder_center<<cylinder_center3D.x, cylinder_center3D.y, cylinder_center3D.z, 1;   // In cam frame
		if (_cylinder_center.norm()>1)  // 3D point is valid
		{
			_cylinder_center = _cam2global * _cylinder_center;
		
		}
		else
		{
			
			Eigen::Vector4f tmp = _cam2global * Eigen::Vector4f(0,0,0,1);
			_bad_z  = tmp(2);
			// std::cout<<"BY RGB: _cylinder_center is \n"<<_cylinder_center<<"\n\n";
		}

		Eigen::Vector3f dir1;  // normal in world
		dir1 = _cam2global.block(0,0,3,3) * _recorded_hand2cam.block(0,0,3,3)*Eigen::Vector3f(0,0,1);
		cv::Mat camera_matrix_ = (cv::Mat1d(3, 3) <<cfg->cam_intrinsic(0,0),cfg->cam_intrinsic(0,1),cfg->cam_intrinsic(0,2),
								cfg->cam_intrinsic(1,0),cfg->cam_intrinsic(1,1),cfg->cam_intrinsic(1,2),
								cfg->cam_intrinsic(2,0),cfg->cam_intrinsic(2,1),cfg->cam_intrinsic(2,2));
		cv::Mat Kinv = camera_matrix_.inv();
		
		int H = _rgb_image.rows;
		int W = _rgb_image.cols;
		pcl::VoxelGrid<pcl::PointXYZRGB> sor;
		sor.setInputCloud (_dense_raw_cloud);
		sor.setLeafSize (0.005f, 0.005f, 0.005f);
		PointCloudRGB::Ptr cloud_filtered(new PointCloudRGB);
		sor.filter (*cloud_filtered);

		pcl::PassThrough<pcl::PointXYZRGB> pass;
		pass.setInputCloud (cloud_filtered);
		pass.setFilterFieldName ("z");
		pass.setFilterLimits (0.1, 2.0);
		pass.filter (*cloud_filtered);

		pcl::NormalEstimation<pcl::PointXYZRGB, pcl::PointNormal> ne;
		ne.setInputCloud (cloud_filtered);
		pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGB> ());
		ne.setSearchMethod (tree);
		PointCloudNormal::Ptr cloud_normals(new PointCloudNormal);
		ne.setRadiusSearch (0.01);
		ne.compute (*cloud_normals);
		PointCloudRGBNormal::Ptr scene_normals(new PointCloudRGBNormal);
		pcl::concatenateFields (*cloud_normals, *cloud_filtered, *scene_normals);

		
		dir1.normalize();
		std::vector<cv::Point2f> corners = marker_corners_[cylinder_marker_id];
		int cylinder_center_u = 0;
		int cylinder_center_v = 0;
		for (auto pt:corners)
		{
			cylinder_center_u += pt.y;
			cylinder_center_v += pt.x;
		}
		cylinder_center_u /= 4;
		cylinder_center_v /= 4; 
		
		pcl::search::KdTree<pcl::PointXYZRGBNormal>::Ptr tree1 (new pcl::search::KdTree<pcl::PointXYZRGBNormal> ());
		tree1->setInputCloud(scene_normals);
		std::vector<int> pointIdxNKNSearch(1);
  		std::vector<float> pointNKNSquaredDistance(1);
		pcl::PointXYZRGBNormal pt;
		pt.x = cylinder_center3D.x;
		pt.y = cylinder_center3D.y;
		pt.z = cylinder_center3D.z;

		Eigen::Vector3f dir2;
		PointCloudRGBNormal::Ptr centers(new PointCloudRGBNormal);
		std::vector<float> dots;
		bool isgood=false;
		if ( tree1->radiusSearch (pt, 0.01, pointIdxNKNSearch, pointNKNSquaredDistance) > 0)
		{
			int num_nei=pointIdxNKNSearch.size();
			dots.resize(num_nei);
			for (int i=0;i<num_nei;i++)
			{
				int idx =  pointIdxNKNSearch[i];
				dir2<<scene_normals->points[idx].normal_x,scene_normals->points[idx].normal_y,scene_normals->points[idx].normal_z;
				dir2 = _cam2global.block(0,0,3,3) * dir2;
				dir2.normalize();
				float dot=fabs(dir1.dot(dir2));
				std::cout<<"dot="<<dot<<std::endl;
				if (dot>=std::cos(10*M_PI/180))    // cylinder and handmarker should be kind of parallel,publish meaningless values 
				{
					isgood=true;
					break;
				}

				// pcl::PointXYZRGBNormal pt;
				// pt.x=scene_normals->points[idx].x;
				// pt.y=scene_normals->points[idx].y;
				// pt.z=scene_normals->points[idx].z;
				// pt.r=255;
				// pt.normal[0]=scene_normals->points[idx].normal_x;
				// pt.normal[1]=scene_normals->points[idx].normal_y;
				// pt.normal[2]=scene_normals->points[idx].normal_z;
				// centers->points.push_back(pt);
			}	
			
		}
		// else
		// {
		// 	dir2<<0,0,1;
		// 	ROS_ERROR("dir2 is illed...");
		// }
		if (!isgood) 
		{
			ROS_ERROR("cylinder is not perpendicular to plane. ");
			return false;
		}

		// if (centers->points.size()>0)
		// 	pcl::io::savePLYFile("/home/pracsys/debug/centers.ply", *centers);

		// int du[9]={0,0,0,1,1,1, -1,-1,-1};
		// int dv[9]={0,1,-1,0,1,-1,0,1,-1};
		// Eigen::MatrixXf dir2(3,9);

		
		// for (int i=0;i<9;i++)
		// {
		// 	int u=cylinder_center_u+du[i];
		// 	int v=cylinder_center_v+dv[i];
		// 	dir2(0,i)=scene_normals->at(u,v).normal[0];
		// 	dir2(1,i)=scene_normals->at(u,v).normal[1];
		// 	dir2(2,i)=scene_normals->at(u,v).normal[2];

		// 	pcl::PointXYZRGB pt;
		// 	pt.x=scene_normals->at(u,v).x;
		// 	pt.y=scene_normals->at(u,v).y;
		// 	pt.z=scene_normals->at(u,v).z;
		// 	pt.r=255;
		// 	centers->points.push_back(pt);
		// }

		// dir2 = _cam2global.block(0,0,3,3) * dir2;
		// dir2.normalize();

		// float dot=fabs(dir1.dot(dir2));
		// if (dot<0.97)    // cylinder and handmarker should be kind of parallel,publish meaningless values 
		// {
		// 	ROS_ERROR("cylinder is not perpendicular to plane. dot is %f",dot);
		// 	return false;
		// }

		
		



		// pcl::NormalEstimation<pcl::PointXYZRGB, pcl::Normal> ne;
		// ne.setInputCloud (_dense_raw_cloud);
		// pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGB> ());
		// ne.setSearchMethod (tree);
		// pcl::PointCloud<pcl::Normal>::Ptr cloud_normals (new pcl::PointCloud<pcl::Normal>);
		// ne.setRadiusSearch (0.002);
		// ne.compute (*cloud_normals);
		// dir2<<cloud_normals->at(cylinder_center_u, cylinder_center_v).normal_x,
		// 	  cloud_normals->at(cylinder_center_u, cylinder_center_v).normal_y,
		// 	  cloud_normals->at(cylinder_center_u, cylinder_center_v).normal_z;

		

		// bool isgood=false;
		// for (int i=0;i<9;i++)
		// {
		// 	Eigen::Vector3f dir2_local=dir2.block(0,i,3,1);
		// 	dir2_local.normalize();
		// 	float dot=fabs(dir1.dot(dir2_local));
		// 	std::cout<<"dot="<<dot<<"\n";
		// 	if (dot>=0.97)   // cylinder and handmarker should be kind of parallel,publish meaningless values 
		// 	{
		// 		isgood=true;
		// 		break;
		// 	}
		// }
		// if (!isgood)
		// {
		// 	ROS_ERROR("cylinder is not perpendicular to plane.");
		// 	return false;
		// }
			

		// std::cout<<"dir2 = \n"<<dir2<<"\n\n";
		// std::cout<<"dir1 = \n"<<dir1<<"\n\n";
		// std::cout<<"fabs(dir1.dot(dir2)) = "<<fabs(dir1.dot(dir2))<<std::endl;
		// if (fabs(dir1.dot(dir2))<0.97)   // cylinder and handmarker should be kind of parallel,publish meaningless values 
		// {
		// 	ROS_ERROR("cylinder is not perpendicular to plane. dot product is %f", fabs(dir1.dot(dir2)));
		// 	return false;
		// }
		
		// std::cout<<"_cam2global is \n"<<_cam2global<<"\n\n";
		// if (_handmarker_center.norm()<1e-10 && (handmarker_center3D.x!=0 || handmarker_center3D.y!=0 
		// 	|| handmarker_center3D.z!=0) )  // Not inited
		// {
		// 	_handmarker_center<<handmarker_center3D.x,handmarker_center3D.y,handmarker_center3D.z,1;
			
		// 	_handmarker_center = _cam2global * _handmarker_center;

		// }
		//std::cout<<"_handmarker_center is \n"<<_handmarker_center<<"\n\n";
		
	}

	if (cylinder_marker_id==-1 || _cylinder_center.norm()<1e-10)
	{
		ROS_ERROR("cylinder marker not detected");
		return false;  // publish meaningless values for cy2end_effector
	}

	_dense_raw_cloud->clear();

	/*
	// CHECK
	Eigen::Vector4f X3d;
	X3d<<marker_size_/2.0,marker_size_/2.0,0,1;
	Eigen::Vector4f tmp = cy2end_effector * X3d;
	tmp(0) /= tmp(2);
	tmp(1) /= tmp(2);
	tmp(2) = 1;
	Eigen::Vector3f proj = cfg->cam_intrinsic * tmp.head(3);
	cv::Point2f point(proj(0),proj(1));
	cv::circle(image_copy,point,5,cv::Scalar(255, 165, 0),-1);
	std::cout<<"projected point is\n"<<point<<"\n\n";
	*/

	return true;
}




bool PoseEstimate::markerPose1(const cv::Mat &image_in)
{
	subscribeMsg1();
	_dense_raw_cloud1->clear();
	Utils::convert3dOrganizedRGB(_depth_image1,_rgb_image1,cfg->cam_intrinsic,_dense_raw_cloud1);
	cv::Ptr<cv::aruco::DetectorParameters> marker_params = cv::aruco::DetectorParameters::create();
	
	
	marker_params->minDistanceToBorder = cfg->minDistanceToBorder;
	marker_params->adaptiveThreshWinSizeMax = cfg->adaptiveThreshWinSizeMax;
	marker_params->adaptiveThreshWinSizeStep = cfg->adaptiveThreshWinSizeStep;
	marker_params->minCornerDistanceRate = cfg->minCornerDistanceRate;
	marker_params->minMarkerPerimeterRate = cfg->minMarkerPerimeterRate;
	marker_params->minOtsuStdDev = cfg->minOtsuStdDev;
	marker_params->perspectiveRemoveIgnoredMarginPerCell = cfg->perspectiveRemoveIgnoredMarginPerCell;
	marker_params->maxErroneousBitsInBorderRate = cfg->maxErroneousBitsInBorderRate;
	marker_params->errorCorrectionRate = cfg->errorCorrectionRate;
	marker_params->cornerRefinementWinSize = cfg->cornerRefinementWinSize;
	marker_params->cornerRefinementMaxIterations = cfg->cornerRefinementMaxIterations;
	marker_params->cornerRefinementMinAccuracy = cfg->cornerRefinementMinAccuracy;
	marker_params->cornerRefinementMethod = cfg->cornerRefinementMethod; 

	cv::Ptr<cv::aruco::Dictionary> dictionary_ = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
	std::vector<std::vector<cv::Point2f>> marker_corners_, rejected_points;
	std::vector<int> marker_ids_;
	cv::aruco::detectMarkers(image_in, dictionary_, marker_corners_, marker_ids_, marker_params, rejected_points);
	std::cout<<"marker_ids_ size is :"<<marker_ids_.size()<<std::endl;
	std::cout<<"rejected_points size is :"<<rejected_points.size()<<std::endl;
	//CHECK
	cv::Mat image_copy = image_in;
	cv::aruco::drawDetectedMarkers(image_copy, marker_corners_, marker_ids_);
	cv::imwrite("/home/pracsys/debug/markerdetect1.png",image_copy);

	int base_marker_id = -1;
	int cylinder_marker_id = -1;
	std::vector<cv::Point2f> base_marker_corner, cylinder_marker_corner;
	for (int i=0;i<marker_ids_.size();i++)
	{
		if (marker_ids_[i]==0)
		{
			base_marker_corner = marker_corners_[i];
			base_marker_id = i;
		}
		else if (marker_ids_[i]==5)
		{
			cylinder_marker_id = i;
			cylinder_marker_corner = marker_corners_[i];
		}
	}

	if (marker_corners_.size()==0 || base_marker_id==-1 || cylinder_marker_id==-1)
	{
		std::cout<<"[WARNING] Not both marker detected !!\n\n";		
	}
	double base_marker_size_ = 0.012954,
				 cy_marker_size_ = 0.02;
	std::vector<cv::Vec3d> rvecs_, tvecs_;
	cv::Mat camera_matrix_ = (cv::Mat1d(3, 3) <<cfg->cam_intrinsic(0,0),cfg->cam_intrinsic(0,1),cfg->cam_intrinsic(0,2),
												cfg->cam_intrinsic(1,0),cfg->cam_intrinsic(1,1),cfg->cam_intrinsic(1,2),
												cfg->cam_intrinsic(2,0),cfg->cam_intrinsic(2,1),cfg->cam_intrinsic(2,2));
	cv::Mat dist_coeffs_ = (cv::Mat1d(4,1)	<<0,0,0,0);																		
	std::vector<std::vector<cv::Point2f>> tmp_corners;	
	cv::Mat rot_mat;										
	Eigen::Matrix4f hand2cam = Eigen::Matrix4f::Identity(4,4);     // hand marker frame to cam frame
	Eigen::Matrix4f cy2cam = Eigen::Matrix4f::Identity(4,4);      // cylinder frame to cam frame
	pcl::PointXYZRGB cylinder_center3D, handmarker_center3D;
	
	/*
	if (base_marker_id!=-1)
	{
		rvecs_.clear();
		tvecs_.clear();
		tmp_corners.clear();
		tmp_corners.push_back(marker_corners_[base_marker_id]);
		handmarker_center3D = getMarkerCenter3D1(tmp_corners);
		
		cv::aruco::estimatePoseSingleMarkers(tmp_corners, base_marker_size_, camera_matrix_, dist_coeffs_, rvecs_, tvecs_);
		
		cv::Vec3d rvec = rvecs_[0], tvec = tvecs_[0]; 
				
		cv::Rodrigues(rvec,rot_mat);  // double

		hand2cam<<rot_mat.at<double>(0,0),rot_mat.at<double>(0,1),rot_mat.at<double>(0,2),tvec[0],    
				rot_mat.at<double>(1,0),rot_mat.at<double>(1,1),rot_mat.at<double>(1,2),tvec[1],
				rot_mat.at<double>(2,0),rot_mat.at<double>(2,1),rot_mat.at<double>(2,2),tvec[2],
				0,0,0,1;
		

		for (int i=0;i<rvecs_.size();i++)
		{
			cv::aruco::drawAxis(image_copy, camera_matrix_,dist_coeffs_,rvecs_[i],tvecs_[i],0.01);
		}

		_cam2handmarker1 = hand2cam.inverse(); 

	}
	*/
	
	if (cylinder_marker_id!=-1)
	{
		rvecs_.clear();
		tvecs_.clear();
		tmp_corners.clear();
		tmp_corners.push_back(marker_corners_[cylinder_marker_id]);
		cylinder_center3D = getMarkerCenter3D1(tmp_corners);
		cv::aruco::estimatePoseSingleMarkers(tmp_corners, cy_marker_size_, camera_matrix_, dist_coeffs_, rvecs_, tvecs_);
		
		cv::Vec3d cyrvec = rvecs_[0], cytvec = tvecs_[0];
		
		cv::Rodrigues(cyrvec,rot_mat);  

		for (int i=0;i<rvecs_.size();i++)
		{
			cv::aruco::drawAxis(image_copy, camera_matrix_,dist_coeffs_,rvecs_[i],tvecs_[i],0.01);
		}

		cy2cam<<rot_mat.at<double>(0,0),rot_mat.at<double>(0,1),rot_mat.at<double>(0,2),cytvec[0],    
				rot_mat.at<double>(1,0),rot_mat.at<double>(1,1),rot_mat.at<double>(1,2),cytvec[1],
				rot_mat.at<double>(2,0),rot_mat.at<double>(2,1),rot_mat.at<double>(2,2),cytvec[2],
				0,0,0,1;

		_cy2cam1 = cy2cam;  // cylinder frame to end effector frame
		
	}
	// cv::imwrite("/home/pracsys/debug/markerdetect1.png",image_copy);

	if (cylinder_center3D.x==0 || cylinder_center3D.y==0 || cylinder_center3D.z==0)
	{
		return false;   // depth wrong
	}
	
	if (cylinder_marker_id!=-1 && base_marker_id!=-1)
	{
		

		// std::cout<<"_cy2world is \n"<<_cy2world<<"\n\n";
		// _cylinder_center1<<cylinder_center3D.x, cylinder_center3D.y, cylinder_center3D.z, 1;   // In cam frame
		// std::cout<<"cylinder_center3D1 is "<<cylinder_center3D.x<<" "<<cylinder_center3D.y<<" "<<cylinder_center3D.z<<"\n";
		_cylinder_center1 = _cam2global1 * _cylinder_center1;
		
	}

	_dense_raw_cloud1->clear();

	if (cylinder_marker_id==-1 || base_marker_id==-1)
	{
		return false;  // publish meaningless values for cy2end_effector
	}

	

	return true;
}

Eigen::Vector3f PoseEstimate::markerNormal(const std::vector<std::vector<cv::Point2f>> &all_corners)
{
	std::vector<cv::Point2f> corners = all_corners[0];
	// std::cout<<"_dense_raw_cloud size is "<<_dense_raw_cloud->points.size()<<"\n";
	// std::cout<<corners[0].x<<" "<<corners[0].y<<"\n\n";
	Eigen::Vector3f pt0;
	pt0<< _dense_raw_cloud->at(corners[0].x,corners[0].y).x,
	_dense_raw_cloud->at(corners[0].x,corners[0].y).y,
	_dense_raw_cloud->at(corners[0].x,corners[0].y).z;
	
	Eigen::Vector3f pt1;
	pt1<< _dense_raw_cloud->at(corners[1].x,corners[1].y).x,
	_dense_raw_cloud->at(corners[1].x,corners[1].y).y,
	_dense_raw_cloud->at(corners[1].x,corners[1].y).z;

	Eigen::Vector3f pt2;
	pt2<< _dense_raw_cloud->at(corners[2].x,corners[2].y).x,
	_dense_raw_cloud->at(corners[2].x,corners[2].y).y,
	_dense_raw_cloud->at(corners[2].x,corners[2].y).z;

	Eigen::Vector3f pt3;
	pt3<< _dense_raw_cloud->at(corners[3].x,corners[3].y).x, 
	_dense_raw_cloud->at(corners[3].x,corners[3].y).y,
	_dense_raw_cloud->at(corners[3].x,corners[3].y).z;

	Eigen::Vector3f line02 = pt2-pt0;
	Eigen::Vector3f line13 = pt3-pt0;
	Eigen::Vector3f normal = line02.cross(line13);
	std::cout<<"line02 is \n"<<line02<<"\n\n";
	std::cout<<"line13 is \n"<<line13<<"\n\n";
	normal.normalize();
	std::cout<<"normal is \n"<<normal<<"\n\n";

	return normal;

}



// Get marker center 3D pos in cam frame
pcl::PointXYZRGB PoseEstimate::getMarkerCenter3D(const std::vector<std::vector<cv::Point2f>> &all_corners)
{
	std::clock_t begin = clock();
	// std::cout<<"_dense_raw_cloud size is "<<_dense_raw_cloud->points.size()<<"\n";
	//pcl::io::savePLYFile("/home/pracsys/debug/_dense_raw_cloud.ply",*_dense_raw_cloud);
	float sum_x=0, sum_y=0;
	for (auto pt:all_corners[0])
	{

		sum_x += pt.x; 
		sum_y += pt.y;  
		
	}
	float pix_x = sum_x/4.0; // u
	float pix_y = sum_y/4.0; // v
	int dx[8]={0, 0, 1,1,1, -1,-1,-1};
	int dy[8]={-1,1, 0,1,-1, 0,1,-1};
	pcl::PointXYZRGB center;
	center.x = 0;
	center.y = 0;
	center.z = 0;
	std::queue<std::pair<float,float>> Q;
	std::pair<float,float> pt(pix_x,pix_y);
	Q.push(pt);
	std::map<std::pair<float,float>,bool> hash;
	hash[pt]=true;
	int cnt=0;
	while (!Q.empty())
	{
		std::pair<float,float> tmp=Q.front();
		Q.pop();
		cnt++;
		if (cnt>10) 
		{
			pcl::PointXYZRGB res;
			res.x=0;
			res.y=0;
			res.z=0;
			std::cout<<"get marker 3d takes "<<double(clock()-begin)/CLOCKS_PER_SEC<<"\n";
			return res;
		}
		center = _dense_raw_cloud->at(tmp.first,tmp.second);
		if (center.z>0.1 && center.z<0.3)
		{
			std::cout<<"got center3D\n";
			break;
		}
		
		for (int i=0;i<8;i++)
		{
			std::pair<float,float> tmp_pt(tmp.first+dx[i],tmp.second+dy[i]);
			if (hash[tmp_pt]==false)
			{
				Q.push(tmp_pt);
				hash[tmp_pt]=true;
			}
		}

	}
	std::cout<<"get marker 3d takes "<<double(clock()-begin)/CLOCKS_PER_SEC<<"\n";
	return center;
}


pcl::PointXYZRGB PoseEstimate::getMarkerCenter3D1(const std::vector<std::vector<cv::Point2f>> &all_corners)
{
	float sum_x=0, sum_y=0;
	for (auto pt:all_corners[0])
	{

		sum_x += pt.x; 
		sum_y += pt.y;  
		
	}
	float pix_x = sum_x/4.0; // u
	float pix_y = sum_y/4.0; // v
	int dx[8]={0, 0, 1,1,1, -1,-1,-1};
	int dy[8]={-1,1, 0,1,-1, 0,1,-1};
	pcl::PointXYZRGB center;
	std::queue<cv::Point2f> Q;
	Q.push(cv::Point2f(pix_x,pix_y));
	std::map<cv::Point2f,bool> hash;
	int cnt=0;
	while (!Q.empty())
	{
		cv::Point2f tmp=Q.front();
		Q.pop();
		cnt++;
		if (cnt>10) return center;
		center = _dense_raw_cloud1->at(tmp.x,tmp.y);
		if (center.z!=0)
			break;
		
		for (int i=0;i<8;i++)
		{

			Q.push(cv::Point2f(tmp.x+dx[i],tmp.y+dy[i]));
		}

	}
	 
	return center;
}


pcl::PointXYZRGB PoseEstimate::getPix3D(float u, float v)
{

	float pix_x = u; // u
	float pix_y = v; // v
	int dx[8]={0, 0, 1,1,1, -1,-1,-1};
	int dy[8]={-1,1, 0,1,-1, 0,1,-1};
	pcl::PointXYZRGB pt_3d;
	pt_3d.x = 0;
	pt_3d.y = 0;
	pt_3d.z = 0;
	std::queue<std::pair<float,float>> Q;
	std::pair<float,float> pt(pix_x,pix_y);
	Q.push(pt);
	std::map<std::pair<float,float>,bool> hash;
	hash[pt]=true;
	int cnt=0;
	while (!Q.empty())
	{
		std::pair<float,float> tmp=Q.front();
		Q.pop();
		cnt++;
		if (cnt>10) 
		{
			pcl::PointXYZRGB res;
			res.x=0;
			res.y=0;
			res.z=0;
			return res;
		}
		pt_3d = _dense_raw_cloud->at(tmp.first,tmp.second);
		if (pt_3d.z>0 && pt_3d.z<0.3)
			break;
		
		for (int i=0;i<8;i++)
		{
			std::pair<float,float> tmp_pt(tmp.first+dx[i],tmp.second+dy[i]);
			if (hash[tmp_pt]==false)
			{
				Q.push(tmp_pt);
				hash[tmp_pt]=true;
			}
		}

	}
	return pt_3d;
}

void PoseEstimate::processOneFingerMarker(const std::vector<std::vector<cv::Point2f>> &marker_corners, 
	const std::vector<int> &marker_ids, int query_idx, float marker_size, Eigen::Vector4f &pos_in_basemarker)
{
	std::vector<cv::Vec3d> rvecs_, tvecs_;

	cv::Mat dist_coeffs_ = (cv::Mat1d(4,1)	<<0,0,0,0);			
	cv::Mat camera_matrix_ = (cv::Mat1d(3, 3) <<cfg->cam_intrinsic(0,0),cfg->cam_intrinsic(0,1),cfg->cam_intrinsic(0,2),
							cfg->cam_intrinsic(1,0),cfg->cam_intrinsic(1,1),cfg->cam_intrinsic(1,2),
							cfg->cam_intrinsic(2,0),cfg->cam_intrinsic(2,1),cfg->cam_intrinsic(2,2));

	std::vector<std::vector<cv::Point2f>> tmp_corners;
	tmp_corners.push_back(marker_corners[query_idx]);

	cv::aruco::estimatePoseSingleMarkers(tmp_corners, marker_size, camera_matrix_, dist_coeffs_, rvecs_, tvecs_);
	
	cv::Vec3d cyrvec = rvecs_[0], cytvec = tvecs_[0];
	
	cv::Mat rot_mat;
	cv::Rodrigues(cyrvec,rot_mat);  

	for (int i=0;i<rvecs_.size();i++)
	{
		cv::aruco::drawAxis(_marker_debug_img, camera_matrix_,dist_coeffs_,rvecs_[i],tvecs_[i],marker_size);
	}

	Eigen::Matrix4f cur_in_cam;
	cur_in_cam<<rot_mat.at<double>(0,0),rot_mat.at<double>(0,1),rot_mat.at<double>(0,2),cytvec[0],    
			rot_mat.at<double>(1,0),rot_mat.at<double>(1,1),rot_mat.at<double>(1,2),cytvec[1],
			rot_mat.at<double>(2,0),rot_mat.at<double>(2,1),rot_mat.at<double>(2,2),cytvec[2],
			0,0,0,1;
	
	Eigen::Matrix4f cur_in_basemarker = _cam2global * cur_in_cam;
	pos_in_basemarker<<cur_in_basemarker(0,3),cur_in_basemarker(1,3),cur_in_basemarker(2,3),1;
}