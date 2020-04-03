# Gazebo underactuated hand simulator

The simulator offers three hands: Reflex, Model-O and Model-T42 (Yale OpenHand project - https://www.eng.yale.edu/grablab/openhand/). 

Install:

    1. sudo apt-get install ros-kinetic-gazebo-ros-pkgs ros-kinetic-gazebo-ros-control
    
    2. sudo apt-get install ros-kinetic-joint-state-controller
    
To run the simulation:

    1. In gripper.yaml, set 'gripper_type' name ('reflex','model_O','model_T42').
    
    2. For reflex hand:
       roslaunch hand_simulator hand.launch gripper:=hand_reflex fingers3:=true
       
    3. For Model_O hand:
       roslaunch hand_simulator hand.launch gripper:=hand_O fingers3:=true
       
    4. For Model_T42 hand:
       roslaunch hand_simulator hand.launch gripper:=hand_T42 fingers3:=false
        
    5. rosrun hand_simulator keyboard_control.py (Loads keyboard control - tested only on the Model-T42).
    


Youtube video:

[![Demo with Model T42](https://img.youtube.com/vi/Mz0lp8VCFuk/0.jpg)](https://youtu.be/Mz0lp8VCFuk)
