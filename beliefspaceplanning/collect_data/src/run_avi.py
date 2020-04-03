#!/usr/bin/env python

import rospy
from std_srvs.srv import Empty, EmptyResponse
import numpy as np
import matplotlib.pyplot as plt

save_srv = rospy.ServiceProxy('/collect/save_data', Empty)
rand_epi_srv = rospy.ServiceProxy('/collect/random_episode', Empty)


for i in range(1,2):

    print "Running random episode..."
    rand_epi_srv()
    print "Collection For One Episode is Done!"
    save_srv()
