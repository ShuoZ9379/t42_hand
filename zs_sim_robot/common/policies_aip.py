import tensorflow as tf
import numpy as np
from common import tf_util
from a2c.utils import fc
from common.distributions_aip import make_pdtype
from common.input import observation_placeholder, observation_placeholder_free, encode_observation
from common.tf_util import adjust_shape
from common.mpi_running_mean_std import RunningMeanStd
from common.models import get_network_builder
import functools
from common.tf_util import get_session, save_variables, load_variables
import gym
from common.cmd_util import make_vec_env

def build_env_ref(env_id,seed=0,goal_height=1.0,ho=0,ah_goal_loc_idx=8,ctrl_rwd=1,env_type='corl',with_obs=0,obs_idx=20,ah_with_goal_loc=0,ah_with_reach_goal=1,ctrl_rwd_coef=1):
    config = tf.ConfigProto(allow_soft_placement=True,intra_op_parallelism_threads=1,inter_op_parallelism_threads=1)
    config.gpu_options.allow_growth = True
    get_session(config=config)
    env = make_vec_env(env_id, env_type, 2000, with_obs, 1, obs_idx, 1e6, 0, ah_with_goal_loc, ah_goal_loc_idx, ah_with_reach_goal, ctrl_rwd, 0, ctrl_rwd_coef, ho, goal_height, 1, seed, reward_scale=1.0, flatten_dict_observations=True)
    return env

def reacher_adjust_obs(obs):
    if len(obs.shape)==1:
        return np.concatenate((obs[:4],obs[6:8],obs[8:10]+obs[4:6],obs[4:6])).reshape((1,-1))
    else:
        return np.concatenate((obs[:,:4],obs[:,6:8],obs[:,8:10]+obs[:,4:6],obs[:,4:6]),axis=1)

def get_ppo_action_ref(obs,policy_ref,env_name='Reacher-v2'):
    if env_name=='Reacher-v2':
        adjusted_obs=reacher_adjust_obs(obs)
        if len(adjusted_obs.shape)==1:
            adjusted_obs=adjusted_obs.reshape((1,-1))
        actions, values, _, neglogpacs=policy_ref.step(adjusted_obs[0,:])
        actions_ref=actions.reshape((1,-1))
        for i in range(1,adjusted_obs.shape[0]):
            actions, values, _, neglogpacs=policy_ref.step(adjusted_obs[i,:])
            actions_ref=np.concatenate((actions_ref,actions),axis=0)
    else:
        raise NotImplementedError
    return actions_ref

class PolicyWithValue(object):
    """
    Encapsulates fields and methods for RL policy and value function estimation with shared parameters
    """

    def __init__(self, ablation, ref_type, policy_ref, r_diff_model, alpha_func, env, env_type,env_id, observations, reference_actions,trust_values,latent, estimate_q=False, vf_latent=None, sess=None, **tensors):
        """
        Parameters:
        ----------
        env             RL environment

        observations    tensorflow placeholder in which the observations will be fed

        latent          latent state from which policy distribution parameters should be inferred

        vf_latent       latent state from which value function should be inferred (if None, then latent is used)

        sess            tensorflow session to run calculations in (if None, default session is used)

        **tensors       tensorflow tensors for additional attributes such as state or mask

        """

        self.X = observations
        self.state = tf.constant([])
        self.initial_state = None
        self.__dict__.update(tensors)

        self.REF_A=reference_actions
        self.ALPHA=trust_values
        self.policy_ref=policy_ref
        self.r_diff_model=r_diff_model
        self.ablation=ablation
        #print(self.ablation)
        #raise
        self.ref_type=ref_type
        self.alpha_func=alpha_func
        self.env_name=env_id

        vf_latent = vf_latent if vf_latent is not None else latent

        vf_latent = tf.layers.flatten(vf_latent)
        latent = tf.layers.flatten(latent)

        # Based on the action space, will select what probability distribution type
        self.pdtype = make_pdtype(env_type, env.action_space)

        self.pd, self.pi = self.pdtype.pdfromlatent(latent,reference_actions,trust_values,init_scale=0.01)
        

        # Take an action
        self.action_sto = self.pd.sample()
        self.action_det = self.pd.mode()

        # Calculate the neg log of our probability
        self.neglogp_sto = self.pd.neglogp(self.action_sto)
        self.neglogp_det = self.pd.neglogp(self.action_det)
        self.sess = sess or tf.get_default_session()

        if estimate_q:
            assert isinstance(env.action_space, gym.spaces.Discrete)
            self.q = fc(vf_latent, 'q', env.action_space.n)
            self.vf = self.q
        else:
            self.vf = fc(vf_latent, 'vf', 1)
            self.vf = self.vf[:,0]

    def _evaluate(self, variables, observation, action_ref, alpha,**extra_feed):
        sess = self.sess
        feed_dict = {self.X: adjust_shape(self.X, observation),self.REF_A: adjust_shape(self.REF_A, action_ref),self.ALPHA: adjust_shape(self.ALPHA, alpha)}

        for inpt_name, data in extra_feed.items():
            if inpt_name in self.__dict__.keys():
                raise
                inpt = self.__dict__[inpt_name]
                if isinstance(inpt, tf.Tensor) and inpt._op.type == 'Placeholder':
                    feed_dict[inpt] = adjust_shape(inpt, data)

        return sess.run(variables, feed_dict)

    def update_r_diff_model(self,r_diff_model):
        self.r_diff_model=r_diff_model

    def step(self, observation, stochastic,**extra_feed):
        
        if self.ref_type=='ppo':
            action_ref=get_ppo_action_ref(observation,self.policy_ref,env_name=self.env_name)
        else:
            raise NotImplementedError
        if self.ablation=='auto':
            if not self.r_diff_model.started:
                alpha=np.array([[1.0]])
            else:
                r_diff=self.r_diff_model.predict(observation,action_ref)
                if self.alpha_func=='squared':
                    alpha=1-np.sqrt(r_diff)
                    #print(alpha)
                else:
                    raise NotImplementedError
        else:
            alpha=np.array([[float(self.ablation)]])
        
        if stochastic:
            a, v, state, neglogp = self._evaluate([self.action_sto, self.vf, self.state, self.neglogp_sto], observation,action_ref, alpha, **extra_feed)
        else:
            a, v, state, neglogp = self._evaluate([self.action_det, self.vf, self.state,self.neglogp_det], observation, action_ref, alpha, **extra_feed)
        if state.size == 0:
            state = None
        return a, v, state, neglogp, action_ref[0], alpha[0]

    def value(self, ob, *args, **kwargs):
        """
        Compute value estimate(s) given the observation(s)

        Parameters:
        ----------

        observation     observation data (either single or a batch)

        **extra_feed    additional data such as state or mask (names of the arguments should match the ones in constructor, see __init__)

        Returns:
        -------
        value estimate
        """
        return self._evaluate(self.vf, ob, *args, **kwargs)

    def save(self, save_path):
        #tf_util.save_state(save_path, sess=self.sess)
        functools.partial(save_variables, sess=get_session())(save_path)
    def load(self, load_path):
        #tf_util.load_state(load_path, sess=self.sess)
        functools.partial(load_variables, sess=get_session())(load_path)

def build_policy(env, policy_network, value_network=None,  normalize_observations=False, estimate_q=False, **policy_kwargs):
    if isinstance(policy_network, str):
        network_type = policy_network
        policy_network = get_network_builder(network_type)(**policy_kwargs)

    def policy_fn(ablation,env_type,env_id,ref_type,policy_ref,r_diff_model,alpha_func,nbatch=None, sess=None, observ_placeholder=None,ac_placeholder=None,alpha_placeholder=None):
        ob_space = env.observation_space
        env_space = env.action_space

        if nbatch==np.inf:
            X = observ_placeholder if observ_placeholder is not None else observation_placeholder_free(ob_space)
            REF_A = ac_placeholder if ac_placeholder is not None else observation_placeholder_free(ac_space,name='Ref_ac')
        else:
            X = observ_placeholder if observ_placeholder is not None else observation_placeholder(ob_space, batch_size=nbatch)
            REF_A = ac_placeholder if ac_placeholder is not None else observation_placeholder(ac_space,batch_size=nbatch,name='Ref_ac')
        ALPHA = alpha_placeholder

        extra_tensors = {}
        if normalize_observations and (X.dtype == tf.float32 or X.dtype == tf.float64):
            #print('Normalizing Observations!!!!!!!!!')
            #raise
            encoded_x, rms = _normalize_clip_observation(X)
            extra_tensors['rms'] = rms
        else:
            encoded_x = X
        encoded_x = encode_observation(ob_space, env_type, encoded_x)

        with tf.variable_scope('pi', reuse=tf.AUTO_REUSE):
            policy_latent = policy_network(encoded_x)

        _v_net = value_network

        if _v_net is None or _v_net == 'shared':
            vf_latent = policy_latent
        else:
            if _v_net == 'copy':
                _v_net = policy_network
            else:
                assert callable(_v_net)
            with tf.variable_scope('vf', reuse=tf.AUTO_REUSE):
                vf_latent = _v_net(encoded_x)

        policy = PolicyWithValue(
            ablation=ablation,
            ref_type=ref_type,
            policy_ref=policy_ref,
            r_diff_model=r_diff_model,
            alpha_func=alpha_func,
            env=env,
            env_type=env_type,
            env_id=env_id,
            observations=X,
            reference_actions=REF_A,
            trust_values=ALPHA,
            latent=policy_latent,
            vf_latent=vf_latent,
            sess=sess,
            estimate_q=estimate_q,
            **extra_tensors
        )
        return policy

    return policy_fn


def _normalize_clip_observation(x, clip_range=[-5.0, 5.0]):
    rms = RunningMeanStd(shape=x.shape[1:])
    norm_x = tf.clip_by_value((x - rms.mean) / rms.std, min(clip_range), max(clip_range))
    return norm_x, rms

