#!/usr/bin/env python

import rospy
import numpy as np
from hand_control.msg import plan_for_goal_request, plan_for_goal_response


class planCall():

    planning_done = False

    path = '/home/pracsys/catkin_ws/src/t42_control/hand_control/plans/'

    goal_counter = 0
    publishToPlanner = True
    goals = np.array([[12,103],[2.8,110],[57,108],[77,95],[-25, 96],[95, 76],[-38, 82],[58, 81]])
    # goals = np.array([[-25, 96],[95, 76],[-38, 82],[58, 81]])
    planning_algorithm = 'naive'

    def __init__(self):
        rospy.init_node('plan_call', anonymous=True)

        self.planning_request_pub = rospy.Publisher('/planning/request', plan_for_goal_request, queue_size=1)
        rospy.Subscriber('/planning/response', plan_for_goal_response, self.planningResponseCallback)

        rate = rospy.Rate(10)
        rospy.sleep(2)
        while not rospy.is_shutdown():
            rate.sleep()
            if (self.publishToPlanner):
                print('[plan_call] Calling planner with goal ' + str(self.goals[self.goal_counter]) + '...')
                self.plan_for_goal(self.goals[self.goal_counter])
            rate.sleep()

            
    def plan_for_goal(self, goal):
        
        msg = plan_for_goal_request()

        msg.goal_state = goal
        msg.goal_radius = 10.0
        msg.time_limit = 2*600 #seconds
        msg.probability_success_threshold = 0.5 #affects SVM validity check 
        msg.planning_algorithm = self.planning_algorithm

        print('Requesting goal %d with algorithm %s...'%(self.goal_counter, self.planning_algorithm))
        self.planning_request_pub.publish(msg)

        self.planning_done = False
        self.publishToPlanner = False


    def planningResponseCallback(self, msg):
        self.planning_actions = msg.planned_actions
        self.planned_path = msg.planned_path
        self.planning_succeeded = msg.reached_goal

        self.planning_done = True

        self.planning_actions = np.array(self.planning_actions).reshape((-1,2))
        self.planned_path = np.array(self.planned_path).reshape((-1,4))

        File = self.planning_algorithm + '_goal' + str(self.goal_counter) + '_plan.txt'
        np.savetxt(self.path + File, self.planning_actions, delimiter = ', ')
        File =  self.planning_algorithm + '_goal' + str(self.goal_counter) + '_traj.txt'
        np.savetxt(self.path + File, self.planned_path, delimiter = ', ')

        print('[plan_call] Planned received and saved in ' + File)
        if self.planning_algorithm == 'naive':
            self.planning_algorithm = 'robust'
        else:
            self.goal_counter += 1
            self.planning_algorithm = 'naive'
        self.publishToPlanner = True
        

if __name__ == '__main__':
    
    try:
        planCall()
    except rospy.ROSInterruptException:
        pass