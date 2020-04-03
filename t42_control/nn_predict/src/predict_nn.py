""" 
Author: Avishai Sintov
"""
from __future__ import division, print_function, absolute_import

from nn_functions import * # My utility functions

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import pickle

simORreal = 't42_cyl35'

class predict_nn:

    # def __init__(self):
    #     param_file = '/home/pracsys/catkin_ws/src/t42_control/nn_predict/models/' + simORreal + '_param.obj'

    #     with open(param_file, 'rb') as f: 
    #         x_mean, x_std, DropOut, Regularization, self.hidden_layers, self.activation, self.state_action_dim, self.state_dim, self.num_output, self.model_file = pickle.load(f)

    #     # Network Parameters
    #     self.num_input = self.state_action_dim

    #     self.x_mu = x_mean
    #     self.x_sigma = x_std

    #     # tf Graph input 
    #     self.X = tf.placeholder("float", [None, self.num_input])
    #     self.Y = tf.placeholder("float", [None, self.num_output])

    #     # Store layers weight & bias
    #     self.weights, self.biases = wNb(self.num_input, self.hidden_layers, self.num_output)

    #     self.prediction = neural_net(self.X, self.weights, self.biases, self.activation)

    #     self.sess = tf.Session()

    #     # Restore variables from disk.
    #     self.saver = tf.train.Saver()
    #     self.saver.restore(self.sess, self.model_file)


    # def predict(self, sa):
    #     s = np.copy(sa[0][:self.num_output])
        
    #     sa = normzG(sa.reshape(1,-1), self.x_mu[:self.num_input], self.x_sigma[:self.num_input])
    #     ds = self.sess.run(self.prediction, {self.X: sa.reshape(1,self.num_input)})
    #     ds = denormzG(ds, self.x_mu[self.num_input:self.num_input + self.num_output], self.x_sigma[self.num_input:self.num_input + self.num_output])
        
    #     s_next = s + ds
    #     return s_next

    def __init__(self):
        param_file = '/home/pracsys/catkin_ws/src/t42_control/nn_predict/models/' + simORreal + '_param.obj'

        with open(param_file, 'rb') as f: 
            x_mean, x_std, DropOut, Regularization, self.hidden_layers, self.activation, self.state_action_dim, self.state_dim, self.num_output_pos, self.num_output_load, self.model_file = pickle.load(f)

        # Network Parameters
        self.num_input = self.state_action_dim

        self.x_mu = x_mean
        self.x_sigma = x_std

        # tf Graph input 
        self.X = tf.placeholder("float", [None, self.num_input])
        self.Y_pos = tf.placeholder("float", [None, self.num_output_pos])
        self.Y_load = tf.placeholder("float", [None, self.num_output_load])

        # Store layers weight & bias
        self.weights_pos, self.biases_pos = wNb(self.num_input, self.hidden_layers, self.num_output_pos)
        self.weights_load, self.biases_load = wNb(self.num_input, self.hidden_layers, self.num_output_load)

        self.prediction_pos = neural_net(self.X, self.weights_pos, self.biases_pos, self.activation)
        self.prediction_load = neural_net(self.X, self.weights_load, self.biases_load, self.activation)

        self.sess = tf.Session()

        # Restore variables from disk.
        self.saver = tf.train.Saver()
        self.saver.restore(self.sess, self.model_file)

    def predict(self, sa):
        s = np.copy(sa[0][:self.num_output_pos])
        
        sa = normzG(sa.reshape(1,-1), self.x_mu[:self.num_input], self.x_sigma[:self.num_input])
        ds_pos, ds_load = self.sess.run([self.prediction_pos, self.prediction_pos], {self.X: sa.reshape(1,self.num_input)})
        ds = np.concatenate((ds_pos, ds_load), axis = 0)
        ds = denormzG(ds, self.x_mu[self.num_input:], self.x_sigma[self.num_input:])
        
        s_next = s + ds
        return s_next

    

        
# if __name__ == "__main__":
#     NN = predict_nn()

#     sa = np.array([67.7478, -109.5858 ,   80.,    20., 1.0,   -1.0])
#     s_next = NN.predict(sa)
#     print(s_next)





