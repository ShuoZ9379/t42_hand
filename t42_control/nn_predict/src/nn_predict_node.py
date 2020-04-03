#!/usr/bin/env python

import rospy
from nn_predict.srv import StateAction2State
import numpy as np

from predict_nn import predict_nn

class Spin_predict(predict_nn):

    def __init__(self):
        predict_nn.__init__(self)

        rospy.Service('/nn/predict', StateAction2State, self.callbackPredictService)

        rospy.init_node('predict', anonymous=True)

        print('[nn_predict_node] Ready to predict...')
        rospy.spin()


    def callbackPredictService(self, req):
        s = np.array(req.state)
        a = np.array(req.action)

        print('-----------------------------------------------------------------------------------------------')
        print('state: ' + str(s) + ', action: ' + str(a))

        sa = np.concatenate((s, a), axis=0)

        sa = sa.reshape((1,self.state_action_dim))

        s_next = self.predict(sa).reshape(1, self.state_dim)

        print('Predicted next state: ' + str(s_next))
     
        return {'next_state': s_next[0]}

   

if __name__ == '__main__':
    
    try:
        SP = Spin_predict()
    except rospy.ROSInterruptException:
        pass