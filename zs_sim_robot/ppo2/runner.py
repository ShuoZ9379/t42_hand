import numpy as np
from common.runners import AbstractEnvRunner
import gym

class Runner(AbstractEnvRunner):
    """
    We use this object to make a mini batch of experiences
    __init__:
    - Initialize the runner

    run():
    - Make a mini batch
    """
    def __init__(self, env, model, nsteps, gamma, lam, **kwargs):
        super().__init__(env=env, model=model, nsteps=nsteps, **kwargs)
        # Lambda used in GAE (General Advantage Estimation)
        self.lam = lam
        # Discount rate
        self.gamma = gamma
        # Possibly arguments for reset() (Deprecated!!!)
        self.kwargs=kwargs

    def run(self,do_eval=False,num_eval_eps=1,compare=False,compare_ah_idx=8,reacher_sd=1,acrobot_sd=1,eval_steps=2048):
        #compare_ah_idx : [0,2,7,8,15];  reacher_sd : [1,2,5]
        # Here, we init the lists that will contain the mb of experiences
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs = [],[],[],[],[],[]
        mb_final_obs=[]
        mb_states = self.states
        epinfos = []

        if not do_eval:
            # For n in range number of steps
            for _ in range(self.nsteps):
                # Given observations, get action value and neglopacs
                # We already have self.obs because Runner superclass run self.obs[:] = env.reset() on init
                actions, values, self.states, neglogpacs = self.model.step(self.obs, S=self.states, M=self.dones)
                mb_obs.append(self.obs.copy())
                mb_actions.append(actions)
                mb_values.append(values)
                mb_neglogpacs.append(neglogpacs)
                mb_dones.append(self.dones)

                # Take actions in env and look the results
                # Infos contains a ton of useful informations
                self.obs[:], rewards, self.dones, infos = self.env.step(actions)
                if type(rewards)!=np.ndarray and type(self.dones)!=np.ndarray and type(infos)!=list:
                    no_dummy=True
                    rewards,self.dones,infos=np.array([rewards]),np.array([self.dones]),[infos]
                else:
                    no_dummy=False
                for info in infos:
                    maybeepinfo = info.get('episode')
                    if maybeepinfo: 
                        epinfos.append(maybeepinfo)
                        if no_dummy:
                            self.obs[:] = self.env.reset(**self.kwargs)
                mb_rewards.append(rewards)
        else:
            mb_four_obs=np.ones((0,self.env.observation_space.shape[0]))
            self.nsteps=eval_steps
            self.obs = np.zeros((self.nenv,) + self.env.observation_space.shape, dtype=self.env.observation_space.dtype.name)
            self.obs[:] = self.env.reset(**self.kwargs)
            
            if compare:
                assert num_eval_eps==1
                if self.env.env_name=='ah':
                    self.obs[0,:4]=np.array([0.03238881511894898396,118.18717713766936583397,16.0,16.0])
                    if not (self.env.with_obs and self.env.obs_idx==14):
                        self.env.goal_loc=self.env.goals[compare_ah_idx]
                        if self.env.state_with_goal_loc:
                            self.obs[0,4:6]=self.env.goal_loc
                    self.env.cur_state=self.obs[0,:]
                    self.env.compare_reset(self.env.goal_loc.copy(),self.obs[0,:].copy())

                elif self.env.env_name=='corl_Reacher-v2':
                    environ=gym.make('Reacher-v2')
                    environ.seed(1000000+reacher_sd)
                    compare_init=environ.reset()
                    self.obs[0,:]=np.concatenate((compare_init[:4],compare_init[6:8],compare_init[8:10]+compare_init[4:6],compare_init[4:6]))
                    self.env.goal_loc=self.obs[0,-2:]
                    self.env.cur_state=self.obs[0,:]
                    self.env.compare_reset(self.obs[0,-2:].copy(),self.obs[0,:].copy())

                elif self.env.env_name=='corl_Acrobot-v1':
                    environ=gym.make('Acrobot-v1')
                    environ.seed(10000000+acrobot_sd)
                    compare_init=environ.reset()
                    self.obs[0,:]=compare_init
                    self.env.cur_state=self.obs[0,:]
                    self.env.compare_reset(self.env.cur_state.copy())

                elif self.env.env_name=='real_ah':
                    self.obs[0,:4]=self.env.init_mu
                    if not (self.env.with_obs and self.env.obs_idx==14):
                        self.env.goal_loc=self.env.goals[compare_ah_idx]
                        if self.env.state_with_goal_loc:
                            self.obs[0,4:6]=self.env.goal_loc
                    self.env.cur_state=self.obs[0,:]
                    self.env.compare_reset(self.env.goal_loc.copy(),self.obs[0,:].copy())
               
            self.states = self.model.initial_state
            self.dones = [False for _ in range(self.nenv)]
            done_ct=1
            #while done_ct<=num_eval_eps:
            for _ in range(self.nsteps):
                actions, values, self.states, neglogpacs = self.model.step(self.obs, S=self.states, M=self.dones)
                mb_obs.append(self.obs.copy())
                mb_actions.append(actions)
                mb_values.append(values)
                mb_neglogpacs.append(neglogpacs)
                mb_dones.append(self.dones)

                if self.env.env_name=='real_ah':
                    self.obs[:], rewards, self.dones, infos, four_obs = self.env.step(actions,True)
                    mb_four_obs=np.concatenate((mb_four_obs,four_obs))
                else:
                    self.obs[:], rewards, self.dones, infos = self.env.step(actions)
                if type(rewards)!=np.ndarray and type(self.dones)!=np.ndarray and type(infos)!=list:
                    no_dummy=True
                    rewards,self.dones,infos=np.array([rewards]),np.array([self.dones]),[infos]
                else:
                    no_dummy=False
                for info in infos:
                    maybeepinfo = info.get('episode')
                    if maybeepinfo: 
                        epinfos.append(maybeepinfo)
                        mb_final_obs.append(self.obs.copy())
                        if no_dummy:
                            self.obs[:] = self.env.reset(**self.kwargs)
                            if compare:
                                assert num_eval_eps==1
                                if self.env.env_name=='ah':
                                    self.obs[0,:4]=np.array([0.03238881511894898396,118.18717713766936583397,16.0,16.0])
                                    if not (self.env.with_obs and self.env.obs_idx==14):
                                        self.env.goal_loc=self.env.goals[compare_ah_idx]
                                        if self.env.state_with_goal_loc:
                                            self.obs[0,4:6]=self.env.goal_loc
                                    self.env.cur_state=self.obs[0,:]
                                    self.env.compare_reset(self.env.goal_loc.copy(),self.obs[0,:].copy())

                                elif self.env.env_name=='corl_Reacher-v2':
                                    environ=gym.make('Reacher-v2')
                                    environ.seed(1000000+reacher_sd)
                                    compare_init=environ.reset()
                                    self.obs[0,:]=np.concatenate((compare_init[:4],compare_init[6:8],compare_init[8:10]+compare_init[4:6],compare_init[4:6]))
                                    self.env.goal_loc=self.obs[0,-2:]
                                    self.env.cur_state=self.obs[0,:]
                                    self.env.compare_reset(self.obs[0,-2:].copy(),self.obs[0,:].copy())

                                elif self.env.env_name=='corl_Acrobot-v1':
                                    environ=gym.make('Acrobot-v1')
                                    environ.seed(10000000+acrobot_sd)
                                    compare_init=environ.reset()
                                    self.obs[0,:]=compare_init
                                    self.env.cur_state=self.obs[0,:]
                                    self.env.compare_reset(self.env.cur_state.copy())
                        done_ct+=1
                mb_rewards.append(rewards)
            mb_final_obs = np.asarray(mb_final_obs, dtype=self.obs.dtype)

        #batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        last_values = self.model.value(self.obs, S=self.states, M=self.dones)

        # discount/bootstrap off value fn
        mb_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0
        for t in reversed(range(len(mb_rewards))):
            if t == self.nsteps - 1:
                nextnonterminal = 1.0 - self.dones
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[t+1]
                nextvalues = mb_values[t+1]
            delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_values[t]
            mb_advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
        mb_returns = mb_advs + mb_values
        if self.env.env_name=='real_ah' and do_eval:
            return (*map(sf01, (mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs)),
                mb_states, epinfos, mb_final_obs,mb_four_obs)
        else:
            return (*map(sf01, (mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs)),
                mb_states, epinfos, mb_final_obs)

def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])


