import tensorflow as tf
import numpy as np

import logger

def make_placeholder(num_input, num_output):
    inputs = tf.placeholder(tf.float32, (None, num_input))
    targets = tf.placeholder(tf.float32, (None, num_output))
    return inputs, targets

class NN(object):
    def __init__(self, num_input, num_output, make_model, classify=False, lr=1e-3, optimizer=tf.train.AdamOptimizer, **kwargs):        
        logger.log('NN', locals())
        self.classify=classify
        self.inputs_ph, self.targets_ph = make_placeholder(num_input, num_output)
        phi = make_model(self.inputs_ph)
        self.pred = self._make_pred(phi, num_output)
        self.classify_pred=tf.cast(self.pred>=0.5,tf.float32)
        
        self.loss = tf.losses.mean_squared_error(labels=self.targets_ph, predictions=self.pred)
        self.clipped_pred=tf.clip_by_value(self.pred,1e-7,1-1e-7)
        self.classify_loss= tf.reduce_mean(-self.targets_ph*tf.log(self.clipped_pred)-(1-self.targets_ph)*tf.log(1-self.clipped_pred))
        self.train_op = optimizer(learning_rate=lr).minimize(self.loss) 
       
    def _make_pred(self, phi, num_output):
        pred = tf.contrib.layers.fully_connected(phi, num_outputs=num_output, activation_fn=tf.identity)
        pred = tf.nn.sigmoid(pred)
        return pred

    def train(self, inputs, targets):
        sess = tf.get_default_session()
        if not self.classify:
            _, loss = sess.run([self.train_op, self.loss], feed_dict={self.inputs_ph: inputs, self.targets_ph: targets})
        else:
            _, loss = sess.run([self.train_op, self.classify_loss], feed_dict={self.inputs_ph: inputs, self.targets_ph: targets})
        return loss

    def predict(self, inputs):      
        sess = tf.get_default_session()
        if not self.classify:
            pred = sess.run(self.pred, feed_dict={self.inputs_ph: inputs})
        else:
            pred = sess.run(self.classify_pred, feed_dict={self.inputs_ph: inputs})
        return pred

    def eval(self, inputs, targets):
        sess = tf.get_default_session()
        if not self.classify:
            loss = sess.run(self.loss, feed_dict={self.inputs_ph: inputs, self.targets_ph: targets})
        else:
            loss = sess.run(self.classify_loss, feed_dict={self.inputs_ph: inputs, self.targets_ph: targets})
        return loss



        
    
