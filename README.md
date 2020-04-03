# t42_hand
This includes all source files for both T42 real hand and simulation using ros-kinetic.
Put all folders under your `~/home/catkin_ws/src/`
### Basic Experiment Setups
*You need to change some paths if some predefined paths do not exist*
- Build all ros packages including the realsense camera (see [realsense](https://github.com/ShuoZ9379/t42_hand/tree/master/realsense-ros))
-launch camera `roslaunch realsense2_camera rs_rgbd.launch`
-launch marker tracker `roslaunch marker_tracker zs_marker_tracker_v2.launch`
-launch hand control `roslaunch hand_control zs_blue_run_v2.launch`
-launch collect data `roslaunch collect_t42 zs_collect_v2.launch`

