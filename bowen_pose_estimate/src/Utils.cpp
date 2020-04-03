#include <Utils.h>


int numBinsEMD = 20;
namespace enc = sensor_msgs::image_encodings;

namespace Utils
{

/********************************* function: meshgrid **************************************************
	Reference: http://answers.opencv.org/question/11788/is-there-a-meshgrid-function-in-opencv
	*******************************************************************************************************/

static void meshgrid(const cv::Range &xgv, const cv::Range &ygv, cv::Mat1i &X, cv::Mat1i &Y)
{
  std::vector<int> t_x, t_y;
  for (int i = xgv.start; i <= xgv.end; i++)
    t_x.push_back(i);
  for (int i = ygv.start; i <= ygv.end; i++)
    t_y.push_back(i);
  cv::repeat(cv::Mat(t_x).reshape(1, 1), cv::Mat(t_y).total(), 1, X);
  cv::repeat(cv::Mat(t_y).reshape(1, 1).t(), 1, cv::Mat(t_x).total(), Y);
}

/********************************* function: type2str **************************************************
	Reference: https://stackoverflow.com/questions/10167534/how-to-find-out-what-type-of-a-mat-object-is-
	with-mattype-in-opencv
	*******************************************************************************************************/

std::string type2str(int type)
{
  std::string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);
  switch (depth)
  {
  case CV_8U:
    r = "8U";
    break;
  case CV_8S:
    r = "8S";
    break;
  case CV_16U:
    r = "16U";
    break;
  case CV_16S:
    r = "16S";
    break;
  case CV_32S:
    r = "32S";
    break;
  case CV_32F:
    r = "32F";
    break;
  case CV_64F:
    r = "64F";
    break;
  default:
    r = "User";
    break;
  }
  r += "C";
  r += (chans + '0');
  return r;
}

/********************************* function: readDepthImage ********************************************
	*******************************************************************************************************/

void readDepthImage(cv::Mat &depthImg, std::string path)
{
  cv::Mat depthImgRaw = cv::imread(path, CV_16UC1);
  depthImg = cv::Mat::zeros(depthImgRaw.rows, depthImgRaw.cols, CV_32FC1);
  for (int u = 0; u < depthImgRaw.rows; u++)
    for (int v = 0; v < depthImgRaw.cols; v++)
    {
      unsigned short depthShort = depthImgRaw.at<unsigned short>(u, v);  // 16bits
      float depth = (float)depthShort * 1e-3;  

      depthImg.at<float>(u, v) = depth;
    }
}

/********************************* function: readProbImage ********************************************
	*******************************************************************************************************/

void readProbImage(cv::Mat &probImg, std::string path)
{
  cv::Mat depthImgRaw = cv::imread(path, CV_16UC1);
  probImg = cv::Mat::zeros(depthImgRaw.rows, depthImgRaw.cols, CV_32FC1);
  for (int u = 0; u < depthImgRaw.rows; u++)
    for (int v = 0; v < depthImgRaw.cols; v++)
    {
      unsigned short depthShort = depthImgRaw.at<unsigned short>(u, v);

      float depth = (float)depthShort / 10000;
      probImg.at<float>(u, v) = depth;
    }
}

/********************************* function: writeDepthImage *******************************************
	*******************************************************************************************************/
// depthImg: must be CV_32FC1, unit in meters
void writeDepthImage(cv::Mat &depthImg, std::string path)
{
  cv::Mat depthImgRaw = cv::Mat::zeros(depthImg.rows, depthImg.cols, CV_16UC1);
  for (int u = 0; u < depthImg.rows; u++)
    for (int v = 0; v < depthImg.cols; v++)
    {
      float depth = depthImg.at<float>(u, v) / 1000;
      // std::cout<<"writeDepthImage:"<<depth<<"\n";
      unsigned short depthShort = (unsigned short)depth;
      depthImgRaw.at<unsigned short>(u, v) = depthShort;
    }
  cv::imwrite(path, depthImgRaw);
}

/********************************* function: writeClassImage *******************************************
	*******************************************************************************************************/

void writeClassImage(cv::Mat &classImg, cv::Mat colorImage, std::string path)
{
  cv::Mat classImgRaw(classImg.rows, classImg.cols, CV_8UC3, cv::Scalar(0, 0, 0));
  cv::Mat colArray(4, 1, CV_8UC3, cv::Scalar(0, 0, 0));
  colArray.at<cv::Vec3b>(0, 0)[0] = 255;
  colArray.at<cv::Vec3b>(1, 0)[1] = 255;
  colArray.at<cv::Vec3b>(2, 0)[2] = 255;

  for (int u = 0; u < classImg.rows; u++)
    for (int v = 0; v < classImg.cols; v++)
    {
      int classVal = classImg.at<uchar>(u, v);
      if (classVal > 0)
      {
        classImgRaw.at<cv::Vec3b>(u, v)[0] = colArray.at<cv::Vec3b>(classVal - 1, 0)[0];
        classImgRaw.at<cv::Vec3b>(u, v)[1] = colArray.at<cv::Vec3b>(classVal - 1, 0)[1];
        classImgRaw.at<cv::Vec3b>(u, v)[2] = colArray.at<cv::Vec3b>(classVal - 1, 0)[2];
      }
    }

  cv::Mat vizImage(classImg.rows, classImg.cols, CV_8UC3, cv::Scalar(0, 0, 0));
  double alpha = 0.6;
  double beta = (1.0 - alpha);
  addWeighted(colorImage, alpha, classImgRaw, beta, 0.0, vizImage);
  cv::imwrite(path, vizImage);
}

/********************************* function: convert3dOrganized ****************************************
	Description: Convert Depth image to point cloud. TODO: Could it be faster?
	Reference: https://gist.github.com/jacyzon/fa868d0bcb13abe5ade0df084618cf9c
	*******************************************************************************************************/

void convert3dOrganized(cv::Mat &objDepth, Eigen::Matrix3f &camIntrinsic, PointCloud::Ptr objCloud)
{
  int imgWidth = objDepth.cols;
  int imgHeight = objDepth.rows;

  objCloud->height = (uint32_t)imgHeight;
  objCloud->width = (uint32_t)imgWidth;
  objCloud->clear();
  objCloud->is_dense = false;
  objCloud->points.resize(objCloud->width * objCloud->height);

  // Try meshgrid implementation
  // cv::Mat1i X, Y;
  // meshgrid(cv::Range(1,imgWidth), cv::Range(1, imgHeight), X, Y);
  // printf("Matrix: %s %dx%d \n", Utils::type2str( depth_image.type() ).c_str(), depth_image.cols, depth_image.rows );
  // cv::Mat CamX = ((X-camIntrinsic(0,2)).mul(objDepth))/camIntrinsic(0,0);
  // cv::Mat CamY = ((Y-camIntrinsic(1,2)).mul(objDepth))/camIntrinsic(1,1);

  for (int u = 0; u < imgHeight; u++)
    for (int v = 0; v < imgWidth; v++)
    {
      float depth = objDepth.at<float>(u, v);
      if (depth > 0.1 && depth<2.0)
      {
        objCloud->at(v, u).x = (float)((v - camIntrinsic(0, 2)) * depth / camIntrinsic(0, 0));
        objCloud->at(v, u).y = (float)((u - camIntrinsic(1, 2)) * depth / camIntrinsic(1, 1));
        objCloud->at(v, u).z = depth;
      }
    }
}

/********************************* function: convert3dOrganizedRGB *************************************
	Description: Convert Depth image to point cloud. TODO: Could it be faster?
	Reference: https://gist.github.com/jacyzon/fa868d0bcb13abe5ade0df084618cf9c
colImage: 8UC3
objDepth: 16UC1
	*******************************************************************************************************/

void convert3dOrganizedRGB(cv::Mat &objDepth, cv::Mat &colImage, Eigen::Matrix3f &camIntrinsic, PointCloudRGB::Ptr objCloud)
{
  int imgWidth = objDepth.cols;
  int imgHeight = objDepth.rows;

  objCloud->height = (uint32_t)imgHeight;
  objCloud->width = (uint32_t)imgWidth;
  objCloud->is_dense = false;
  objCloud->points.resize(objCloud->width * objCloud->height);

  // Try meshgrid implementation
  // cv::Mat1i X, Y;
  // meshgrid(cv::Range(1,imgWidth), cv::Range(1, imgHeight), X, Y);
  // printf("Matrix: %s %dx%d \n", Utils::type2str( depth_image.type() ).c_str(), depth_image.cols, depth_image.rows );
  // cv::Mat CamX = ((X-camIntrinsic(0,2)).mul(objDepth))/camIntrinsic(0,0);
  // cv::Mat CamY = ((Y-camIntrinsic(1,2)).mul(objDepth))/camIntrinsic(1,1);

  float bad_point = 0;  // this can cause the point cloud visualization problem !!
  for (int u = 0; u < imgHeight; u++)
    for (int v = 0; v < imgWidth; v++)
    {
      float depth = objDepth.at<float>(u, v);
      cv::Vec3b colour = colImage.at<cv::Vec3b>(u, v);  // 3*8 bits
      if (depth > 0.1 && depth<2.0)
      {
        objCloud->at(v, u).x = (float)((v - camIntrinsic(0, 2)) * depth / camIntrinsic(0, 0));
        objCloud->at(v, u).y = (float)((u - camIntrinsic(1, 2)) * depth / camIntrinsic(1, 1));
        objCloud->at(v, u).z = depth;
        objCloud->at(v, u).b = colour[0];
        objCloud->at(v, u).g = colour[1];
        objCloud->at(v, u).r = colour[2];
      }
      else
      {
        objCloud->at(v, u).x = bad_point;
        objCloud->at(v, u).y = bad_point;
        objCloud->at(v, u).z = bad_point;
        objCloud->at(v, u).b = 0;
        objCloud->at(v, u).g = 0;
        objCloud->at(v, u).r = 0;
      }
    }
}

/********************************* function: convert3dUnOrganized **************************************
	*******************************************************************************************************/

void convert3dUnOrganized(cv::Mat &objDepth, Eigen::Matrix3f &camIntrinsic, PointCloud::Ptr objCloud)
{
  int imgWidth = objDepth.cols;
  int imgHeight = objDepth.rows;

  for (int u = 0; u < imgHeight; u++)
    for (int v = 0; v < imgWidth; v++)
    {
      float depth = objDepth.at<float>(u, v);
      if (depth > 0.1 && depth<2.0)
      {
        pcl::PointXYZ pt;
        pt.x = (float)((v - camIntrinsic(0, 2)) * depth / camIntrinsic(0, 0));
        pt.y = (float)((u - camIntrinsic(1, 2)) * depth / camIntrinsic(1, 1));
        pt.z = depth;
        objCloud->points.push_back(pt);
      }
    }
}

/********************************* function: convert3dUnOrganizedRGB ***********************************
	*******************************************************************************************************/

void convert3dUnOrganizedRGB(cv::Mat &objDepth, cv::Mat &colorImage, Eigen::Matrix3f &camIntrinsic, PointCloudRGB::Ptr objCloud)
{
  int imgWidth = objDepth.cols;
  int imgHeight = objDepth.rows;

  for (int u = 0; u < imgHeight; u++)
    for (int v = 0; v < imgWidth; v++)
    {
      float depth = objDepth.at<float>(u, v);
      cv::Vec3b colour = colorImage.at<cv::Vec3b>(u, v);
      if (depth > 0.1 && depth<2.0)
      {
        pcl::PointXYZRGB pt;
        pt.x = (float)((v - camIntrinsic(0, 2)) * depth / camIntrinsic(0, 0));
        pt.y = (float)((u - camIntrinsic(1, 2)) * depth / camIntrinsic(1, 1));
        pt.z = depth;
        uint32_t rgb = ((uint32_t)colour.val[2] << 16 | (uint32_t)colour.val[1] << 8 | (uint32_t)colour.val[0]);
        pt.rgb = *reinterpret_cast<float *>(&rgb);
        objCloud->points.push_back(pt);
      }
    }
}

/********************************* function: convert2d **************************************************
	*******************************************************************************************************/

void convert2d(cv::Mat &objDepth, Eigen::Matrix3f &camIntrinsic, PointCloud::Ptr objCloud)
{
  for (int i = 0; i < objCloud->points.size(); i++)
  {
    Eigen::Vector3f point2D = camIntrinsic *
                              Eigen::Vector3f(objCloud->points[i].x, objCloud->points[i].y, objCloud->points[i].z);
    point2D[0] = point2D[0] / point2D[2];
    point2D[1] = point2D[1] / point2D[2];
    if (point2D[1] > 0 && point2D[1] < objDepth.rows && point2D[0] > 0 && point2D[0] < objDepth.cols)
    {
      if (objDepth.at<float>(point2D[1], point2D[0]) == 0 || point2D[2] < objDepth.at<float>(point2D[1], point2D[0]))
        objDepth.at<float>(point2D[1], point2D[0]) = (float)point2D[2];
    }
  }
}

/********************************* function: simpleVis **************************************************
	Reference: http://pointclouds.org/documentation/tutorials/pcl_visualizer.php
	*******************************************************************************************************/

//
boost::shared_ptr<pcl::visualization::PCLVisualizer> simpleVis(PointCloud::ConstPtr cloud)
{
  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
  viewer->setBackgroundColor(0, 0, 0);
  viewer->addPointCloud<pcl::PointXYZ>(cloud, "sample cloud");
  viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud");
  viewer->initCameraParameters();
  return (viewer);
}

/********************************* function: TransformPolyMesh *****************************************
	*******************************************************************************************************/

void TransformPolyMesh(const pcl::PolygonMesh::Ptr &mesh_in, pcl::PolygonMesh::Ptr &mesh_out, Eigen::Matrix4f transform)
{
  PointCloud::Ptr cloud_in(new PointCloud);
  PointCloud::Ptr cloud_out(new PointCloud);
  pcl::fromPCLPointCloud2(mesh_in->cloud, *cloud_in);
  pcl::transformPointCloud(*cloud_in, *cloud_out, transform);
  *mesh_out = *mesh_in;
  pcl::toPCLPointCloud2(*cloud_out, mesh_out->cloud);
  return;
}

/********************************* function: convertToMatrix *******************************************
	*******************************************************************************************************/

void convertToMatrix(Eigen::Isometry3d &from, Eigen::Matrix4f &to)
{
  for (int ii = 0; ii < 4; ii++)
    for (int jj = 0; jj < 4; jj++)
      to(ii, jj) = from.matrix()(ii, jj);
}

/********************************* function: convertToIsometry3d ***************************************
	*******************************************************************************************************/

void convertToIsometry3d(Eigen::Matrix4f &from, Eigen::Isometry3d &to)
{
  for (int ii = 0; ii < 4; ii++)
    for (int jj = 0; jj < 4; jj++)
      to.matrix()(ii, jj) = from(ii, jj);
}

/********************************* function: convertToWorld ********************************************
	*******************************************************************************************************/

void convertToWorld(Eigen::Matrix4f &transform, Eigen::Matrix4f &cam_pose)
{
  transform = cam_pose * transform.eval();
}

/********************************* function: invertTransformationMatrix *******************************
	*******************************************************************************************************/
void invertTransformationMatrix(Eigen::Matrix4f &tform)
{
  Eigen::Matrix3f rotm;
  Eigen::Vector3f trans;
  for (int ii = 0; ii < 3; ii++)
    for (int jj = 0; jj < 3; jj++)
      rotm(ii, jj) = tform(ii, jj);
  trans[0] = tform(0, 3);
  trans[1] = tform(1, 3);
  trans[2] = tform(2, 3);

  rotm = rotm.inverse().eval();
  trans = rotm * trans;

  for (int ii = 0; ii < 3; ii++)
    for (int jj = 0; jj < 3; jj++)
      tform(ii, jj) = rotm(ii, jj);
  tform(0, 3) = -trans[0];
  tform(1, 3) = -trans[1];
  tform(2, 3) = -trans[2];
}

/********************************* function: convertToCamera *******************************************
	*******************************************************************************************************/

void convertToCamera(Eigen::Matrix4f &tform, Eigen::Matrix4f &cam_pose)
{
  Eigen::Matrix4f invCam(cam_pose);

  invertTransformationMatrix(invCam);
  tform = invCam * tform.eval();
}

/********************************* function: toEulerianAngle *******************************************
	from wikipedia
	*******************************************************************************************************/

static void toEulerianAngle(Eigen::Quaternionf &q, Eigen::Vector3f &eulAngles)
{
  // roll (x-axis rotation)
  double sinr = +2.0 * (q.w() * q.x() + q.y() * q.z());
  double cosr = +1.0 - 2.0 * (q.x() * q.x() + q.y() * q.y());
  eulAngles[0] = atan2(sinr, cosr);

  // pitch (y-axis rotation)
  double sinp = +2.0 * (q.w() * q.y() - q.z() * q.x());
  if (fabs(sinp) >= 1)
    eulAngles[1] = copysign(M_PI / 2, sinp); // use 90 degrees if out of range
  else
    eulAngles[1] = asin(sinp);

  // yaw (z-axis rotation)
  double siny = +2.0 * (q.w() * q.z() + q.x() * q.y());
  double cosy = +1.0 - 2.0 * (q.y() * q.y() + q.z() * q.z());
  eulAngles[2] = atan2(siny, cosy);
}

/********************************* function: toQuaternion **********************************************
	from wikipedia
	*******************************************************************************************************/

void toQuaternion(Eigen::Vector3f &eulAngles, Eigen::Quaternionf &q)
{
  double roll = eulAngles[0];
  double pitch = eulAngles[1];
  double yaw = eulAngles[2];
  double cy = cos(yaw * 0.5);
  double sy = sin(yaw * 0.5);
  double cr = cos(roll * 0.5);
  double sr = sin(roll * 0.5);
  double cp = cos(pitch * 0.5);
  double sp = sin(pitch * 0.5);

  q.w() = cy * cr * cp + sy * sr * sp;
  q.x() = cy * sr * cp - sy * cr * sp;
  q.y() = cy * cr * sp + sy * sr * cp;
  q.z() = sy * cr * cp - cy * sr * sp;
}

/********************************* function: toTransformationMatrix ************************************
	*******************************************************************************************************/

void toTransformationMatrix(Eigen::Matrix4f &camPose, std::vector<double> camPose7D)
{
  camPose(0, 3) = camPose7D[0];
  camPose(1, 3) = camPose7D[1];
  camPose(2, 3) = camPose7D[2];
  camPose(3, 3) = 1;

  Eigen::Quaternionf q;
  q.w() = camPose7D[3];
  q.x() = camPose7D[4];
  q.y() = camPose7D[5];
  q.z() = camPose7D[6];
  Eigen::Matrix3f rotMat;
  rotMat = q.toRotationMatrix();

  for (int ii = 0; ii < 3; ii++)
    for (int jj = 0; jj < 3; jj++)
    {
      camPose(ii, jj) = rotMat(ii, jj);
    }
}

/********************************* function: rotationMatrixToEulerAngles *******************************
	*******************************************************************************************************/
// Calculates rotation matrix to euler angles
// The result is the same as MATLAB except the order
// of the euler angles ( x and z are swapped ).
Eigen::Vector3f rotationMatrixToEulerAngles(Eigen::Matrix3f R)
{
  float sy = sqrt(R(0, 0) * R(0, 0) + R(1, 0) * R(1, 0));
  bool singular = sy < 1e-6;
  float x, y, z;
  if (!singular)
  {
    x = atan2(R(2, 1), R(2, 2));
    y = atan2(-R(2, 0), sy);
    z = atan2(R(1, 0), R(0, 0));
  }
  else
  {
    x = atan2(-R(1, 2), R(1, 1));
    y = atan2(-R(2, 0), sy);
    z = 0;
  }
  Eigen::Vector3f rot;
  rot << x, y, z;
  return rot;
}

/********************************* function: getEMDError ***********************************************
	*******************************************************************************************************/

void getEMDError(Eigen::Matrix4f testPose, Eigen::Matrix4f gtPose, PointCloud::Ptr objModel, float &error,
                 std::pair<float, float> &xrange, std::pair<float, float> &yrange, std::pair<float, float> &zrange)
{

  PointCloud::Ptr pcl_1(new PointCloud);
  PointCloud::Ptr pcl_2(new PointCloud);
  pcl::transformPointCloud(*objModel, *pcl_1, testPose);
  pcl::transformPointCloud(*objModel, *pcl_2, gtPose);

  int num_rows = pcl_1->points.size();
  cv::Mat xyzPts_1(num_rows, 1, CV_32FC3);
  cv::Mat xyzPts_2(num_rows, 1, CV_32FC3);

  for (int ii = 0; ii < num_rows; ii++)
  {
    xyzPts_1.at<cv::Vec3f>(ii, 0)[0] = pcl_1->points[ii].x;
    xyzPts_1.at<cv::Vec3f>(ii, 0)[1] = pcl_1->points[ii].y;
    xyzPts_1.at<cv::Vec3f>(ii, 0)[2] = pcl_1->points[ii].z;

    xyzPts_2.at<cv::Vec3f>(ii, 0)[0] = pcl_2->points[ii].x;
    xyzPts_2.at<cv::Vec3f>(ii, 0)[1] = pcl_2->points[ii].y;
    xyzPts_2.at<cv::Vec3f>(ii, 0)[2] = pcl_2->points[ii].z;
  }
  cv::MatND hist_1, hist_2;
  int xbins = numBinsEMD, ybins = numBinsEMD, zbins = numBinsEMD;
  int histSize[] = {xbins, ybins, zbins};
  float xranges[] = {xrange.first, xrange.second};
  float yranges[] = {yrange.first, yrange.second};
  float zranges[] = {zrange.first, zrange.second};
  int channels[] = {0, 1, 2};
  const float *ranges[] = {xranges, yranges, zranges};

  cv::calcHist(&xyzPts_1, 1, channels, cv::Mat(), hist_1, 3, histSize, ranges, true, false);
  cv::calcHist(&xyzPts_2, 1, channels, cv::Mat(), hist_2, 3, histSize, ranges, true, false);

  // std::cout << xrange.first << " " << yrange.first << " " << zrange.first << " " << xrange.second << " " << yrange.second << " " << zrange.second << std::endl;
  int sigSize = xbins * ybins * zbins;
  cv::Mat sig1(sigSize, 4, CV_32FC1);
  cv::Mat sig2(sigSize, 4, CV_32FC1);

  //fill value into signature
  for (int x = 0; x < xbins; x++)
  {
    for (int y = 0; y < ybins; y++)
    {
      for (int z = 0; z < zbins; z++)
      {
        float binval = hist_1.at<float>(x, y, z);
        sig1.at<float>(x * ybins * zbins + y * zbins + z, 0) = binval;
        sig1.at<float>(x * ybins * zbins + y * zbins + z, 1) = x;
        sig1.at<float>(x * ybins * zbins + y * zbins + z, 2) = y;
        sig1.at<float>(x * ybins * zbins + y * zbins + z, 3) = z;

        binval = hist_2.at<float>(x, y, z);
        sig2.at<float>(x * ybins * zbins + y * zbins + z, 0) = binval;
        sig2.at<float>(x * ybins * zbins + y * zbins + z, 1) = x;
        sig2.at<float>(x * ybins * zbins + y * zbins + z, 2) = y;
        sig2.at<float>(x * ybins * zbins + y * zbins + z, 3) = z;
      }
    }
  }

  error = cv::EMD(sig1, sig2, CV_DIST_L2); //emd 0 is best matching.
}

/********************************* function: c_dist_pose **********************************************
	*******************************************************************************************************/

// float c_dist_pose(Eigen::Matrix4f pose_1, Eigen::Matrix4f pose_2, PointCloud::Ptr objModel) {
//   size_t number_of_points = objModel->points.size();

//   float max_distance = 0;
//   for(int ii=0; ii<number_of_points; ii++){
//     float min_distance = FLT_MAX;

//     Eigen::Matrix<Scalar, 3, 1> p = (allTransforms[index_1]*hull_Q_3D[ii].pos().homogeneous()).head<3>();
//     for(int jj=0; jj<number_of_points; jj++){
//       Eigen::Matrix<Scalar, 3, 1> q = (allTransforms[index_2]*hull_Q_3D[jj].pos().homogeneous()).head<3>();
//       float dist = (p - q).norm();
//       if(dist < min_distance)
//         min_distance = dist;
//     }

//     if(min_distance > max_distance)
//       max_distance = min_distance;
//   }

//   return max_distance;
// }

/********************************* function: getPoseError **********************************************
	*******************************************************************************************************/

void getPoseError(Eigen::Matrix4f testPose, Eigen::Matrix4f gtPose, Eigen::Vector3f symInfo,
                  float &meanrotErr, float &transErr)
{
  Eigen::Matrix3f testRot, gtRot, rotdiff;
  for (int ii = 0; ii < 3; ii++)
    for (int jj = 0; jj < 3; jj++)
    {
      testRot(ii, jj) = testPose(ii, jj);
      gtRot(ii, jj) = gtPose(ii, jj);
    }

  testRot = testRot.inverse().eval();
  rotdiff = testRot * gtRot;
  Eigen::Quaternionf rotdiffQ(rotdiff);
  Eigen::Vector3f rotErrXYZ;
  toEulerianAngle(rotdiffQ, rotErrXYZ);
  rotErrXYZ = rotErrXYZ * 180.0 / M_PI;

  for (int dim = 0; dim < 3; dim++)
  {
    rotErrXYZ(dim) = fabs(rotErrXYZ(dim));
    if (symInfo(dim) == 90)
    {
      rotErrXYZ(dim) = abs(rotErrXYZ(dim) - 90);
      rotErrXYZ(dim) = std::min(rotErrXYZ(dim), 90 - rotErrXYZ(dim));
    }
    else if (symInfo(dim) == 180)
    {
      rotErrXYZ(dim) = std::min(rotErrXYZ(dim), 180 - rotErrXYZ(dim));
    }
    else if (symInfo(dim) == 360)
    {
      rotErrXYZ(dim) = 0;
    }
  }

  meanrotErr = (rotErrXYZ(0) + rotErrXYZ(1) + rotErrXYZ(2)) / 3;
  transErr = sqrt(pow(gtPose(0, 3) - testPose(0, 3), 2) +
                  pow(gtPose(1, 3) - testPose(1, 3), 2) +
                  pow(gtPose(2, 3) - testPose(2, 3), 2));
}

/******************************** function: getRotDistance *********************************************
	*******************************************************************************************************/

float getRotDistance(Eigen::Matrix3f rotMat1, Eigen::Matrix3f rotMat2, Eigen::Vector3f symInfo)
{
  Eigen::Matrix3f rotdiff;

  rotdiff = rotMat1 * rotMat2;
  Eigen::Vector3f rotErrXYZ;
  rotErrXYZ = rotationMatrixToEulerAngles(rotdiff);
  rotErrXYZ = rotErrXYZ * 180.0 / M_PI;

  for (int dim = 0; dim < 3; dim++)
  {
    rotErrXYZ(dim) = fabs(rotErrXYZ(dim));
    if (symInfo(dim) == 90)
    {
      rotErrXYZ(dim) = abs(rotErrXYZ(dim) - 90);
      rotErrXYZ(dim) = std::min(rotErrXYZ(dim), 90 - rotErrXYZ(dim));
    }
    else if (symInfo(dim) == 180)
    {
      rotErrXYZ(dim) = std::min(rotErrXYZ(dim), 180 - rotErrXYZ(dim));
    }
    else if (symInfo(dim) == 360)
    {
      rotErrXYZ(dim) = 0;
    }
  }

  float meanrotErr = (rotErrXYZ(0) + rotErrXYZ(1) + rotErrXYZ(2)) / 3;
  return meanrotErr;
}

/********************************* function: convertToCVMat ********************************************
	*******************************************************************************************************/

void convertToCVMat(Eigen::Matrix4f &pose, cv::Mat &cvPose)
{
  cvPose.at<float>(0, 0) = pose(0, 3);
  cvPose.at<float>(0, 1) = pose(1, 3);
  cvPose.at<float>(0, 2) = pose(2, 3);

  Eigen::Matrix3f rotMat;
  for (int ii = 0; ii < 3; ii++)
    for (int jj = 0; jj < 3; jj++)
    {
      rotMat(ii, jj) = pose(ii, jj);
    }
  Eigen::Quaternionf rotMatQ(rotMat);
  Eigen::Vector3f rotErrXYZ;
  toEulerianAngle(rotMatQ, rotErrXYZ);
  rotErrXYZ = rotErrXYZ * 180.0 / M_PI;

  cvPose.at<float>(0, 3) = rotErrXYZ(0);
  cvPose.at<float>(0, 4) = rotErrXYZ(1);
  cvPose.at<float>(0, 5) = rotErrXYZ(2);
}

/********************************* function: convert6DToMatrix *****************************************
	*******************************************************************************************************/

void convert6DToMatrix(Eigen::Matrix4f &pose, cv::Mat &points, int index)
{
  pose.setIdentity();
  pose(0, 3) = points.at<float>(index, 0);
  pose(1, 3) = points.at<float>(index, 1);
  pose(2, 3) = points.at<float>(index, 2);

  Eigen::Matrix3f rotMat;
  Eigen::Quaternionf q;
  Eigen::Vector3f rotXYZ;
  rotXYZ << points.at<float>(index, 3) * M_PI / 180.0,
      points.at<float>(index, 4) * M_PI / 180.0,
      points.at<float>(index, 5) * M_PI / 180.0;
  toQuaternion(rotXYZ, q);
  rotMat = q.toRotationMatrix();

  for (int ii = 0; ii < 3; ii++)
    for (int jj = 0; jj < 3; jj++)
    {
      pose(ii, jj) = rotMat(ii, jj);
    }
}

/********************************* function: writePoseToFile *******************************************
	*******************************************************************************************************/

void writePoseToFile(Eigen::Matrix4f pose, std::string objName, std::string scenePath, std::string filename)
{
  ofstream pFile;
  pFile.open((scenePath + filename + "_" + objName + ".txt").c_str(), std::ofstream::out | std::ofstream::app);
  pFile << pose(0, 0) << " " << pose(0, 1) << " " << pose(0, 2) << " " << pose(0, 3)
        << " " << pose(1, 0) << " " << pose(1, 1) << " " << pose(1, 2) << " " << pose(1, 3)
        << " " << pose(2, 0) << " " << pose(2, 1) << " " << pose(2, 2) << " " << pose(2, 3) << std::endl;
  pFile.close();
}

/********************************* function: writeScoreToFile ******************************************
	*******************************************************************************************************/

void writeScoreToFile(float score, std::string objName, std::string scenePath, std::string filename)
{
  ofstream pFile;
  pFile.open((scenePath + filename + "_" + objName + ".txt").c_str(), std::ofstream::out | std::ofstream::app);
  pFile << score << std::endl;
  pFile.close();
}

/********************************* function: performTrICP **********************************************
	*******************************************************************************************************/

void performTrICP(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr pclSegment,
                  pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr pclModel,
                  Eigen::Isometry3d &currTransform,
                  Eigen::Isometry3d &finalTransform,
                  float trimPercentage)
{
  PointCloud::Ptr modelCloud(new PointCloud);
  PointCloud::Ptr segmentCloud(new PointCloud);

  Eigen::Matrix4f tform;

  copyPointCloud(*pclModel, *modelCloud);
  copyPointCloud(*pclSegment, *segmentCloud);

  // initialize trimmed ICP
  pcl::console::setVerbosityLevel(pcl::console::L_ALWAYS);
  pcl::recognition::TrimmedICP<pcl::PointXYZ, float> tricp;
  tricp.init(modelCloud);
  tricp.setNewToOldEnergyRatio(1.f);

  float numPoints = trimPercentage * segmentCloud->points.size();

  // get current object transform
  Utils::convertToMatrix(currTransform, tform);
  tform = tform.inverse().eval();

  tricp.align(*segmentCloud, abs(numPoints), tform);
  tform = tform.inverse().eval();

  Utils::convertToIsometry3d(tform, finalTransform);
}

/********************************* function: performICP **********************************************
	*******************************************************************************************************/

void performICP(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr pclSegment,
                pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr pclModel,
                Eigen::Matrix4f &offsetTransform)
{
  PointCloud::Ptr modelCloud(new PointCloud);
  PointCloud::Ptr segmentCloud(new PointCloud);

  Eigen::Matrix4f tform;

  copyPointCloud(*pclModel, *modelCloud);
  copyPointCloud(*pclSegment, *segmentCloud);

  // initialize trimmed ICP
  pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
  icp.setMaximumIterations(100);
  icp.setInputCloud(segmentCloud);
  icp.setInputTarget(modelCloud);
  pcl::PointCloud<pcl::PointXYZ> Final;
  icp.align(Final);
  offsetTransform = icp.getFinalTransformation();
}

/********************************* function: pointToPlaneICP *******************************************
	*******************************************************************************************************/

void pointToPlaneICP(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr pclSegment,
                     pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr pclModel,
                     Eigen::Matrix4f &offsetTransform)
{

  PointCloudNormal::Ptr modelCloud(new PointCloudNormal);
  PointCloudNormal::Ptr segmentCloud(new PointCloudNormal);
  PointCloudNormal segCloudTrans;
  copyPointCloud(*pclModel, *modelCloud);
  copyPointCloud(*pclSegment, *segmentCloud);

  std::vector<int> indices;
  pcl::removeNaNNormalsFromPointCloud(*modelCloud, *modelCloud, indices);
  pcl::removeNaNNormalsFromPointCloud(*segmentCloud, *segmentCloud, indices);

  pcl::IterativeClosestPointWithNormals<pcl::PointNormal, pcl::PointNormal>::Ptr icp(new pcl::IterativeClosestPointWithNormals<pcl::PointNormal, pcl::PointNormal>());
  icp->setMaximumIterations(100);
  icp->setInputSource(segmentCloud); // not cloud_source, but cloud_source_trans!
  icp->setInputTarget(modelCloud);
  icp->align(segCloudTrans);
  if (icp->hasConverged())
  {
    offsetTransform = icp->getFinalTransformation();
    std::cout << "ICP score: " << icp->getFitnessScore() << std::endl;
  }
  else
  {
    std::cout << "ICP did not converge." << std::endl;
    offsetTransform << 1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1;
  }
}

void watershedSegmentation(cv::Mat &image_in, cv::Mat &image_out)
{
  using namespace cv;
  using namespace std;

  imshow("Black Background Image", image_in);
  // Create a kernel that we will use to sharpen our image
  Mat kernel = (Mat_<float>(3,3) <<
                1,  1, 1,
                1, -8, 1,
                1,  1, 1); // an approximation of second derivative, a quite strong kernel
  // do the laplacian filtering as it is
  // well, we need to convert everything in something more deeper then CV_8U
  // because the kernel has some negative values,
  // and we can expect in general to have a Laplacian image with negative values
  // BUT a 8bits unsigned int (the one we are working with) can contain values from 0 to 255
  // so the possible negative number will be truncated
  Mat imgLaplacian;
  filter2D(image_in, imgLaplacian, CV_32F, kernel);
  Mat sharp;
  image_in.convertTo(sharp, CV_32F);
  Mat imgResult = sharp - imgLaplacian;
  // convert back to 8bits gray scale
  imgResult.convertTo(imgResult, CV_8UC3);
  imgLaplacian.convertTo(imgLaplacian, CV_8UC3);
  // imshow( "Laplace Filtered Image", imgLaplacian );
  imshow( "New Sharped Image", imgResult );
  // Create binary image from source image
  Mat bw;
  cvtColor(imgResult, bw, COLOR_BGR2GRAY);
  threshold(bw, bw, 40, 255, THRESH_BINARY | THRESH_OTSU);
  imshow("Binary Image", bw);
  // Perform the distance transform algorithm
  Mat dist;
  distanceTransform(bw, dist, DIST_L2, 3);
  // Normalize the distance image for range = {0.0, 1.0}
  // so we can visualize and threshold it
  normalize(dist, dist, 0, 1.0, NORM_MINMAX);
  imshow("Distance Transform Image", dist);
  // Threshold to obtain the peaks
  // This will be the markers for the foreground objects
  threshold(dist, dist, 0.4, 1.0, THRESH_BINARY);
  // Dilate a bit the dist image
  Mat kernel1 = Mat::ones(3, 3, CV_8U);
  dilate(dist, dist, kernel1);
  imshow("Peaks", dist);
  // Create the CV_8U version of the distance image
  // It is needed for findContours()
  Mat dist_8u;
  dist.convertTo(dist_8u, CV_8U);
  // Find total markers
  vector<vector<Point> > contours;
  findContours(dist_8u, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
  // Create the marker image for the watershed algorithm
  Mat markers = Mat::zeros(dist.size(), CV_32S);
  // Draw the foreground markers
  for (size_t i = 0; i < contours.size(); i++)
  {
      drawContours(markers, contours, static_cast<int>(i), Scalar(static_cast<int>(i)+1), -1);
  }
  // Draw the background marker
  circle(markers, Point(5,5), 3, Scalar(255), -1);
  imshow("Markers", markers*10000);
  // Perform the watershed algorithm
  watershed(imgResult, markers);
  Mat mark;
  markers.convertTo(mark, CV_8U);
  bitwise_not(mark, mark);
  //    imshow("Markers_v2", mark); // uncomment this if you want to see how the mark
  // image looks like at that point
  // Generate random colors
  vector<Vec3b> colors;
  for (size_t i = 0; i < contours.size(); i++)
  {
      int b = theRNG().uniform(0, 256);
      int g = theRNG().uniform(0, 256);
      int r = theRNG().uniform(0, 256);
      colors.push_back(Vec3b((uchar)b, (uchar)g, (uchar)r));
  }
  // Create the result image
  image_out = Mat::zeros(markers.size(), CV_8UC3);
  // Fill labeled objects with random colors
  for (int i = 0; i < markers.rows; i++)
  {
      for (int j = 0; j < markers.cols; j++)
      {
          int index = markers.at<int>(i,j);
          if (index > 0 && index <= static_cast<int>(contours.size()))
          {
              image_out.at<Vec3b>(i,j) = colors[index-1];
          }
      }
  }
  cv::imshow("Final result",image_out);

}

void visualizePointXYZRGB(PointCloudRGB::ConstPtr cloud)
{
  // --------------------------------------------
  // -----Open 3D viewer and add point cloud-----
  // --------------------------------------------
  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
  viewer->setBackgroundColor (0, 0, 0);
  pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
  viewer->addPointCloud<pcl::PointXYZRGB> (cloud, rgb, "sample cloud");
  viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud");
  viewer->addCoordinateSystem (1.0);
  viewer->initCameraParameters ();

  while (!viewer->wasStopped ())
  {
    viewer->spinOnce (100);
    boost::this_thread::sleep(boost::posix_time::microseconds (100000));
  }
  
}


boost::shared_ptr<pcl::visualization::PCLVisualizer> PointXYZRGBToViewer(PointCloudRGB::ConstPtr cloud)
{
  // --------------------------------------------
  // -----Open 3D viewer and add point cloud-----
  // --------------------------------------------
  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
  viewer->setBackgroundColor (0, 0, 0);
  pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
  viewer->addPointCloud<pcl::PointXYZRGB> (cloud, rgb, "sample cloud");
  viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud");
  viewer->addCoordinateSystem (1.0);
  viewer->initCameraParameters ();
  return viewer;  
}


void visualizePointXYZ(pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud, std::string path="")
{
  // --------------------------------------------
  // -----Open 3D viewer and add point cloud-----
  // --------------------------------------------
  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
  viewer->setBackgroundColor (0, 0, 0);
  viewer->addPointCloud<pcl::PointXYZ> (cloud, "sample cloud");
  viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud");
  viewer->addCoordinateSystem (1.0);
  viewer->initCameraParameters ();
  if (path!="")
  {
    pcl::io::savePLYFile(path, *cloud);
  }

  while (!viewer->wasStopped ())
  {
    viewer->spinOnce (100);
    boost::this_thread::sleep(boost::posix_time::microseconds (100000));
  }
}


void removeRGBBackgroundByDepth(cv::Mat &rgb_image, const cv::Mat &depth_image)
{
  for (int u=0;u<depth_image.cols;u++)
  {
    for (int v=0;v<depth_image.rows;v++)
    {
      float depth = depth_image.at<float>(v,u);
      std::cout<<depth<<std::endl;
      if (depth>2)
      {
        rgb_image.at<cv::Vec3b>(v,u)[0] = 0;
        rgb_image.at<cv::Vec3b>(v,u)[1] = 0;
        rgb_image.at<cv::Vec3b>(v,u)[2] = 0;
      }
    }
  }
}

/*
  http://wiki.ros.org/depth_image_proc
  https://github.com/ros-perception/image_pipeline/blob/indigo/depth_image_proc/src/nodelets/convert_metric.cpp 
  function "ConvertMetricNodelet::depthCb"
*/
bool rawDepthToMeters(const sensor_msgs::ImageConstPtr& raw_msg, cv::Mat& depth_out)
{
  // Allocate new Image message
  sensor_msgs::ImagePtr depth_msg( new sensor_msgs::Image );
  depth_msg->header   = raw_msg->header;
  depth_msg->height   = raw_msg->height;
  depth_msg->width    = raw_msg->width;

  // Set data, encoding and step after converting the metric.
  if (raw_msg->encoding == enc::TYPE_16UC1)
  {
    depth_msg->encoding = enc::TYPE_32FC1;
    depth_msg->step     = raw_msg->width * (enc::bitDepth(depth_msg->encoding) / 8);
    depth_msg->data.resize(depth_msg->height * depth_msg->step);
    // Fill in the depth image data, converting mm to m
    float bad_point = 0;
    const uint16_t* raw_data = reinterpret_cast<const uint16_t*>(&raw_msg->data[0]);
    float* depth_data = reinterpret_cast<float*>(&depth_msg->data[0]);
    for (unsigned index = 0; index < depth_msg->height * depth_msg->width; ++index)
    {
      uint16_t raw = raw_data[index];
      depth_data[index] = (raw == 0) ? bad_point : (float)raw * SR300_DEPTH_UNIT;  // convert to [m]
    }
  }
  else
  {
    ROS_ERROR("Unsupported image conversion from %s. only support from 16UC1", raw_msg->encoding.c_str());
    return false;
  }

  cv_bridge::CvImagePtr bridge_ptr = cv_bridge::toCvCopy(depth_msg,sensor_msgs::image_encodings::TYPE_32FC1);   // 32FC1
  depth_out = bridge_ptr->image;
  return true;

}



PointCloudRGB::Ptr rosCloud2ToPCLCloudRGB(const sensor_msgs::PointCloud2::ConstPtr& input)
{
  pcl::PCLPointCloud2 pcl_pc2;
  pcl_conversions::toPCL(*input,pcl_pc2);
  PointCloudRGB::Ptr temp_cloud(new PointCloudRGB);
  pcl::fromPCLPointCloud2(pcl_pc2,*temp_cloud);
  return temp_cloud;
}


// rotation from Z axis to "direction"
void directionToQuaternion(const float *direction, Eigen::Quaternionf &q)
{
  Eigen::Vector3f dir;
  dir<<direction[0],direction[1],direction[2];
  q.setFromTwoVectors(Eigen::Vector3f(1,0,0), dir);
  q.normalize();
}


void cvRotMatToEigenMat(const cv::Mat& mat, Eigen::Matrix3f &out)
{
  int height = mat.rows, width = mat.cols;
  assert(height==3);
  assert(width==3);
  out.setZero();
  for (int i=0;i<height*width;i++)
  {
    int row = i/width;
    int col = i%width;
    out<<mat.at<double>(row,col);
  }
  
}

void EigenMatTocvMat(const Eigen::Matrix3f &mat, cv::Mat& out)
{
  for (int i=0;i<9;i++)
  {
    int row = i/3;
    int col = i%3;
    out.at<double>(row,col) = mat(row,col);
  }
  
}


} // namespace Utils

ostream& operator<< (ostream& os, const Eigen::Quaternionf &q)
{
  os<<q.w()<<" "<<q.x()<<" "<<q.y()<<" "<<q.z()<<"\n";
  return os;
}