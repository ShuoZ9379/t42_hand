import numpy as np
import tensorflow as tf
import copy 
import gym
import scipy.stats as stats

import logger
from common import explained_variance, zipsame
import common.tf_util as U

from r_diff.dynamics import ForwardDynamic
from r_diff.model_config import make_mlp
from r_diff.reward_func import get_reward_done_func

def get_variables(scope):
    return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope)

class R_diff_model(object):
    def __init__(self, env, env_id, make_model=make_mlp, 
            #num_warm_start=int(1e4),            
            #init_epochs=20, 
            update_epochs=1, 
            batch_size=512, 
            r_diff_classify=False,
            forward_dynamic=None,
            **kwargs):
        logger.log('r_diff args', locals())
       
        self.ob_space = env.observation_space
        self.ac_space = env.action_space
        self.started = False
        self.r_diff_classify=r_diff_classify
        
        self.use_self_forward_dynamic = False
        if forward_dynamic is None:
            self.use_self_forward_dynamic = True
            self.forward_dynamic = ForwardDynamic(ob_space=self.ob_space, ac_space=self.ac_space, make_model=make_model, scope='new_forward_dynamic', classify=self.r_diff_classify, **kwargs)
            self.old_forward_dynamic = ForwardDynamic(ob_space=self.ob_space, ac_space=self.ac_space, make_model=make_model, scope='old_forward_dynamic', classify=self.r_diff_classify, **kwargs)

            self.update_model = U.function([],[], updates=[tf.assign(oldv, newv)
                for (oldv, newv) in zipsame(get_variables("old_forward_dynamic"), get_variables("new_forward_dynamic"))])
            self.restore_model = U.function([],[], updates=[tf.assign(newv, oldv)
                for (oldv, newv) in zipsame(get_variables("old_forward_dynamic"), get_variables("new_forward_dynamic"))])
        else:
            self.forward_dynamic = forward_dynamic
              
        #self.num_warm_start = num_warm_start 
        #self.init_epochs = init_epochs
        self.update_epochs = update_epochs
        self.batch_size = batch_size
        self.warm_start_done = False
        self.prev_loss_val = None
        
    def predict(self,obs,action):
        # Forward simulation
        predicted_r_diff = self.forward_dynamic.predict(ob=obs, ac=action) # N x 1
        return predicted_r_diff
           
    def _add_data(self, ob, ac, r_diff_label):
        assert self.use_self_forward_dynamic, 'It is invalid to update the external forward dynamics model'
        self.forward_dynamic.append(copy.copy(ob), copy.copy(ac), copy.copy(r_diff_label))
   
    def update_forward_dynamic(self, require_update, ob_val=None, ac_val=None, r_diff_label_val=None):
        '''
        Update the forward dynamic model
        '''
        assert self.use_self_forward_dynamic, 'It is invalid to update the external forward dynamics model'
        #if not self.is_warm_start_done():
        #    logger.log('Warm start progress', (self.forward_dynamic.memory.nb_entries / self.num_warm_start))

        # Check if need to update train
        #if require_update and self.is_warm_start_done():
        if require_update:
            logger.log('Update train')
            self.forward_dynamic.train(self.batch_size, self.update_epochs)
            self.started = True
            '''
        # Check if enough for init train
        if self.forward_dynamic.memory.nb_entries >= self.num_warm_start and not self.warm_start_done:
            logger.log('Init train')
            self.forward_dynamic.train(self.batch_size, self.init_epochs)
            self.warm_start_done = True 
            self.started = True
        '''
            # Check if need to validate
            if self.started:
                if ob_val is not None and ac_val is not None and r_diff_label_val is not None:               
                    logger.log('Validating...')
                    loss_val = self.eval_forward_dynamic(ob_val, ac_val, r_diff_label_val)
                    logger.log('Validation loss: {}'.format(loss_val))
                    if self.prev_loss_val is not None:                    
                        if self.prev_loss_val < loss_val:
                            logger.log('New model is worse or equal, restore')
                            self.restore_model()
                        else:
                            logger.log('New model is better, update')
                            self.update_model()
                    self.prev_loss_val = loss_val                
                else:
                    logger.log('Update without validation')

 
    def add_data_batch(self, obs, acs, r_diff_labels):
        assert self.use_self_forward_dynamic, 'It is invalid to update the external forward dynamics model'
        for ob, ac, r_diff_label in zip(obs, acs, r_diff_labels):
            self._add_data(ob, ac, r_diff_label)

    def eval_forward_dynamic(self, obs, acs, r_diff_label):
        return self.forward_dynamic.eval(obs, acs, r_diff_label)
        
    def is_warm_start_done(self):
        return self.warm_start_done