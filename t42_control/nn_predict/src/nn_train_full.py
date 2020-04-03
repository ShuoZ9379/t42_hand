""" 
Author: Avishai Sintov
"""
from __future__ import division, print_function, absolute_import

from nn_functions import * # My utility functions

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from scipy.io import loadmat
import time
import pickle

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-r", help="Retrain existing model", action="store_true")
parser.add_argument("-p", help="Plot trained models", action="store_true")
args = parser.parse_args()
if args.r and args.p:
    training = True
    retrain = True
if args.r:
    training = True
    retrain = True
elif args.p:
    training = False
    retrain = False
else:
    training = True
    retrain = False

DropOut = True
Regularization = False

print('Loading training data...')

simORreal = 't42_cyl35'
discreteORcont = 'discrete'

dim_ = 5 if np.any(simORreal == np.array(['t42_sqr30','t42_poly10','t42_poly6','t42_elp40'])) else 4

postfix = '_v' + str(0) + '_d' + str(dim_) + '_m' + str(1)
prefix =  simORreal + '_'
file = simORreal + '_data_' + discreteORcont + postfix + '.obj'
path = '/home/pracsys/catkin_ws/src/t42_control/gpup_gp_node/data/dataset_processed/'

print('[data_load] Loading data from "' + file + '"...' )
with open(path + file, 'rb') as f: 
    X, state_dim, action_dim, _, _ = pickle.load(f)
state_action_dim = state_dim + action_dim
num_output = 4

states = X[:,:state_dim]
next_states = X[:,state_action_dim:state_action_dim+num_output]
actions = X[:, state_dim:state_dim+action_dim]
X = np.concatenate((states, actions, next_states-states[:,:num_output]), axis=1)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)
x_mean = scaler.mean_
x_std = scaler.scale_
# X = X*x_std + x_mean # Denormalize or use scaler.inverse_transform(X)

X, Y = X[:,:state_action_dim], X[:,state_action_dim:state_action_dim+num_output]

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.05, random_state= 42)

# Network Parameters
hidden_layers = [200]*2
activation = 2

# Training Parameters
learning_rate =  0.001
num_steps = int(1e4)
batch_size = 150
display_step = 100

# tf Graph input 
X = tf.placeholder("float", [None, state_action_dim])
Y = tf.placeholder("float", [None, num_output])

# Store layers weight & bias
weights, biases = wNb(state_action_dim, hidden_layers, num_output)

# Construct model
keep_prob_input = tf.placeholder(tf.float32)
keep_prob = tf.placeholder(tf.float32)
if not DropOut:
    prediction = neural_net(X, weights, biases, activation)
else:
    X_drop = tf.nn.dropout(X, keep_prob=keep_prob_input)
    prediction = neural_net_dropout(X, weights, biases, keep_prob, activation)

# Define loss 
cost = tf.reduce_mean(0.5*tf.pow(prediction - Y, 2))#/(2*n)
# cost = tf.reduce_mean(np.absolute(y_true - y_pred))
# cost = tf.reduce_sum(tf.square(prediction - Y))

# L2 Regularization
if Regularization:
    beta = 0.01
    regularizer = computeReg(weights)
    cost = cost + beta * regularizer

# Define optimizer
# optimizer = tf.train.AdamOptimizer(learning_rate)
# optimizer = tf.train.GradientDescentOptimizer(learning_rate)
optimizer = tf.train.AdagradOptimizer(learning_rate)
train_op = optimizer.minimize(cost)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

model_file = '/home/pracsys/catkin_ws/src/t42_control/nn_predict/models/' + simORreal + '.ckpt'

with open('../models/' + simORreal + '_param.obj', 'wb') as f: 
    pickle.dump([x_mean, x_std, DropOut, Regularization, hidden_layers, activation, state_action_dim, state_dim, num_output, model_file], f)

# Start Training
# Start a new TF session
COSTS = []	# for plotting
STEPS = []	# for plotting
start = time.time()
with tf.Session() as sess:

    if training:
    
        if  not retrain:
            # Run the initializer
            sess.run(init)
        else:
            # Restore variables from disk.
            saver.restore(sess, model_file)                
            # print("Loaded saved model: %s" % "./models/" + load_from)

        # Training
        for i in range(1, num_steps+1):
            # Get the next batch 
            batch_x, batch_y = next_batch(batch_size, X_train, Y_train)

            # Run optimization op (backprop) and cost op (to get loss value)
            if not DropOut:
                _, c = sess.run([train_op, cost], feed_dict={X: batch_x, Y: batch_y})
            else:
                _, c = sess.run([train_op, cost], feed_dict={X: batch_x, Y: batch_y, keep_prob_input: 0.5, keep_prob: 0.5})
            # Display logs per step
            if i % display_step == 0 or i == 1:
                print('Step %i: Minibatch Loss: %f' % (i, c))
                save_path = saver.save(sess, model_file)
                COSTS.append(c)
                STEPS.append(i)

        print("Optimization Finished!")

        # Save the variables to disk.
        save_path = saver.save(sess, model_file)
        print("Model saved in path: %s" % save_path)

        # Plot cost convergence
        plt.figure(4)
        plt.semilogy(STEPS, COSTS, 'k-')
        plt.xlabel('Step')
        plt.ylabel('Cost')
        plt.ylim([0, np.max(COSTS)])
        plt.grid(True)
    else:
        # Restore variables from disk.
        saver.restore(sess, model_file)

    # Testing
    # Calculate cost for training data
    Y_train_pred = sess.run(prediction, {X: X_train, keep_prob_input: 1.0, keep_prob: 1.0})
    training_cost = sess.run(cost, feed_dict={X: X_train, Y: Y_train, keep_prob_input: 1.0, keep_prob: 1.0})
    print("Training cost:", training_cost)

    Y_test_pred = sess.run(prediction, {X: X_test, keep_prob_input: 1.0, keep_prob: 1.0})
    testing_cost = sess.run(cost, feed_dict={X: X_test, Y: Y_test, keep_prob_input: 1.0, keep_prob: 1.0})
    print("Testing cost=", testing_cost)

    j = 100#np.random.random_integers(1, X_train.shape[0]) %np.array([-0.1177 ,   0.0641  ,  0.9617  ,  0.9387])#
    x_test = X_test[j, :].reshape(1, state_action_dim)
    y = sess.run(prediction, {X: x_test, keep_prob_input: 1.0, keep_prob: 1.0})
    print("Testing point: ", x_test)
    print("Point ", j, ": ", y, Y_test[j, :])

    x_test = x_test.reshape(state_action_dim, 1)
    xo = denormzG(x_test[0:2], x_mean, x_std)
    yo = denormzG(y, x_mean[state_action_dim:state_action_dim+num_output], x_std[state_action_dim:state_action_dim+num_output])
    yr = denormzG(Y_test[j,:], x_mean[state_action_dim:state_action_dim+num_output], x_std[state_action_dim:state_action_dim+num_output])
    # print(x_mean, x_std)
    print("y ", yo, yr)
    print("State: ", xo.reshape(1,2))
    print("Predicted next state: ", xo.reshape(1,2) + yo[:,0:2])
    print("Real next step: ", xo.reshape(1,2) + yr[0:2])

    for y in Y_test_pred:
        y = denormzG(y, x_mean[state_action_dim:state_action_dim+num_output], x_std[state_action_dim:state_action_dim+num_output])
    for y in Y_test:
        y = denormzG(y, x_mean[state_action_dim:state_action_dim+num_output], x_std[state_action_dim:state_action_dim+num_output])
    for x in X_test:
        x = denormzG(x, x_mean, x_std)

    X_next_pred = X_test[:,:num_output] + Y_test_pred
    X_next = X_test[:,:num_output] + Y_test

    plt.plot(X_test[:,0],X_test[:,1], '.r')
    plt.plot(X_next[:,0],X_next[:,1], '.b')
    plt.plot(X_next_pred[:,0],X_next_pred[:,1], 'xk')
    plt.plot([X_test[:,0], X_next_pred[:,0]],[X_test[:,1], X_next_pred[:,1]], '-k')


# plt.show()


