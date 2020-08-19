import os
import time
import numpy as np
import os.path as osp
import logger
from collections import deque
from common import explained_variance, set_global_seeds
from common.policies import build_policy
from common.ah_env import in_hull
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from copy import copy
try:
    from mpi4py import MPI
except ImportError:
    MPI = None
from ppo2.runner import Runner
if not os.path.exists('./ppo2_results/eval/'):
    os.makedirs('./ppo2_results/eval/')
if not os.path.exists('./ppo2_results/single_loss/'):
    os.makedirs('./ppo2_results/single_loss/')

def constfn(val):
    def f(_):
        return val
    return f

# Avoid division error when calculate the mean (in our case if epinfo is empty returns np.nan, not return an error)
def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)

def learn(*, network, env, env_type, total_timesteps, save_path, eval_env = None, need_eval=False, num_eval_eps=1, compare=False,compare_ah_idx=8,reacher_sd=1,acrobot_sd=1,ho=0,find_best=0,
            seed=None, nsteps=2048, ent_coef=0.0, lr=3e-4,lr_factor=3,
            vf_coef=0.5,  max_grad_norm=0.5, gamma=0.99, lam=0.95,
            log_interval=1, nminibatches=4, noptepochs=4, cliprange=0.2,
            save_interval=0, load_path=None, model_fn=None, update_fn=None, init_fn=None, mpi_rank_weight=1, comm=None, plot_single_loss=False, single_loss_suf='',**network_kwargs):
    '''
    Learn policy using PPO algorithm (https://arxiv.org/abs/1707.06347)

    Parameters:
    ----------

    network:                          policy network architecture. Either string (mlp, lstm, lnlstm, cnn_lstm, cnn, cnn_small, conv_only - see baselines.common/models.py for full list)
                                      specifying the standard network architecture, or a function that takes tensorflow tensor as input and returns
                                      tuple (output_tensor, extra_feed) where output tensor is the last network layer output, extra_feed is None for feed-forward
                                      neural nets, and extra_feed is a dictionary describing how to feed state into the network for recurrent neural nets.
                                      See common/models.py/lstm for more details on using recurrent nets in policies

    env: baselines.common.vec_env.VecEnv     environment. Needs to be vectorized for parallel environment simulation.
                                      The environments produced by gym.make can be wrapped using baselines.common.vec_env.DummyVecEnv class.


    nsteps: int                       number of steps of the vectorized environment per update (i.e. batch size is nsteps * nenv where
                                      nenv is number of environment copies simulated in parallel)

    total_timesteps: int              number of timesteps (i.e. number of actions taken in the environment)

    ent_coef: float                   policy entropy coefficient in the optimization objective

    lr: float or function             learning rate, constant or a schedule function [0,1] -> R+ where 1 is beginning of the
                                      training and 0 is the end of the training.

    vf_coef: float                    value function loss coefficient in the optimization objective

    max_grad_norm: float or None      gradient norm clipping coefficient

    gamma: float                      discounting factor

    lam: float                        advantage estimation discounting factor (lambda in the paper)

    log_interval: int                 number of timesteps between logging events

    nminibatches: int                 number of training minibatches per update. For recurrent policies,
                                      should be smaller or equal than number of environments run in parallel.

    noptepochs: int                   number of training epochs per update

    cliprange: float or function      clipping range, constant or schedule function [0,1] -> R+ where 1 is beginning of the training
                                      and 0 is the end of the training

    save_interval: int                number of timesteps between saving events

    load_path: str                    path to load the model from

    **network_kwargs:                 keyword arguments to the policy / network builder. See baselines.common/policies.py/build_policy and arguments to a particular type of network
                                      For instance, 'mlp' network architecture has arguments num_hidden and num_layers.



    '''
    #set_global_seeds(seed)
    if env_type=='corl' and env.env_name=='ah':
        if not os.path.exists('./ppo2_results/test_ah_single_loss'+single_loss_suf+'/'):
            os.makedirs('./ppo2_results/test_ah_single_loss'+single_loss_suf+'/')
    if env_type=='corl' and env.env_name=='real_ah':
        if not os.path.exists('./ppo2_results/test_real_ah_single_loss'+single_loss_suf+'/'):
            os.makedirs('./ppo2_results/test_real_ah_single_loss'+single_loss_suf+'/')

    if env_type=='corl':
        lr_final=lr_factor*1e-4
        lr=lambda f: lr_final * f

    if isinstance(lr, float): lr = constfn(lr)
    else: assert callable(lr)
    if isinstance(cliprange, float): cliprange = constfn(cliprange)
    else: assert callable(cliprange)
    total_timesteps = int(total_timesteps)
    print(network_kwargs)
    policy = build_policy(env, network, **network_kwargs)

    best_model_ls,best_ret_ls=[],[]
    # Get the nb of env
    #nenvs = env.num_envs
    nenvs=1

    # Get state_space and action_space
    ob_space = env.observation_space
    ac_space = env.action_space

    # Calculate the batch_size
    nbatch = nenvs * nsteps
    nbatch_train = nbatch // nminibatches
    is_mpi_root = (MPI is None or MPI.COMM_WORLD.Get_rank() == 0)

    # Instantiate the model object (that creates act_model and train_model)
    if model_fn is None:
        from ppo2.model import Model
        model_fn = Model

    model = model_fn(policy=policy, env_type=env_type, ob_space=ob_space, ac_space=ac_space, nbatch_act=nenvs, nbatch_train=nbatch_train,
                    ent_coef=ent_coef, vf_coef=vf_coef,
                    max_grad_norm=max_grad_norm, comm=comm, mpi_rank_weight=mpi_rank_weight)

    if load_path is not None:
        model.load(load_path)
    # Instantiate the runner object
    runner = Runner(env=env, model=model, nsteps=nsteps, gamma=gamma, lam=lam)
    if eval_env is not None:
        eval_runner = Runner(env = eval_env, model = model, nsteps = nsteps, gamma = gamma, lam= lam)

    epinfobuf = deque(maxlen=100)
    if eval_env is not None:
        eval_epinfobuf = deque(maxlen=100)

    if init_fn is not None:
        init_fn()

    # Start total timer
    tfirststart = time.perf_counter()
    models_ls,eprewmean_ls,best_eps_ret_ls,best_eps_len_ls,best_eps_update_ls=[],[],[],[],[]

    if not (env_type=='corl' and need_eval==True and eval_env is None):
        nupdates = total_timesteps//nbatch
        for update in range(1, nupdates+1):
            assert nbatch % nminibatches == 0
            # Start timer
            tstart = time.perf_counter()
            frac = 1.0 - (update - 1.0) / nupdates
            # Calculate the learning rate
            lrnow = lr(frac)
            # Calculate the cliprange
            cliprangenow = cliprange(frac)
                
            if update % log_interval == 0 and is_mpi_root: logger.info('Stepping environment...')

            # Get minibatch
            obs, returns, masks, actions, values, neglogpacs, states, epinfos, final_obs, succ, best_eps_ret, best_eps_len = runner.run() #pylint: disable=E0632
            if eval_env is not None:
                eval_obs, eval_returns, eval_masks, eval_actions, eval_values, eval_neglogpacs, eval_states, eval_epinfos, eval_final_obs, eval_succ, eval_best_eps_ret, eval_best_eps_len= eval_runner.run() #pylint: disable=E0632
            
            if update % log_interval == 0 and is_mpi_root: logger.info('Done.')

            epinfobuf.extend(epinfos)
            if eval_env is not None:
                eval_epinfobuf.extend(eval_epinfos)


            if env_type=='corl' and find_best and (env.env_name=='ah' or env.env_name=='real_ah') and succ:
                logger.logkv('saving a success model update',update-1)
                model.save(save_path+'_find_best_update_'+str(update-1))
                best_eps_ret_ls.append(best_eps_ret)
                best_eps_len_ls.append(best_eps_len)
                best_eps_update_ls.append(update-1)
                logger.logkv('success episode return',best_eps_ret)
                logger.logkv('success episode length',best_eps_len)
                logger.dumpkvs()


            # Here what we're going to do is for each minibatch calculate the loss and append it.
            mblossvals = []
            if states is None: # nonrecurrent version
                # Index of each element of batch_size
                # Create the indices array
                inds = np.arange(nbatch)
                for _ in range(noptepochs):
                    # Randomize the indexes
                    np.random.shuffle(inds)
                    # 0 to batch_size with batch_train_size step
                    for start in range(0, nbatch, nbatch_train):
                        end = start + nbatch_train
                        mbinds = inds[start:end]
                        slices = (arr[mbinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                        mblossvals.append(model.train(lrnow, cliprangenow, *slices))

            # Feedforward --> get losses --> update
            lossvals = np.mean(mblossvals, axis=0)
            # End timer
            tnow = time.perf_counter()
            # Calculate the fps (frame per second)
            fps = int(nbatch / (tnow - tstart))

            if update_fn is not None:
                update_fn(update)

            if update % log_interval == 0 or update == 1:
                # Calculates if value function is a good predicator of the returns (ev > 1)
                # or if it's just worse than predicting nothing (ev =< 0)
                ev = explained_variance(values, returns)
                logger.logkv("misc/serial_timesteps", update*nsteps)
                logger.logkv("misc/nupdates", update)
                logger.logkv("misc/total_timesteps", update*nbatch)
                logger.logkv("fps", fps)
                logger.logkv("misc/explained_variance", float(ev))
                logger.logkv('eprewmean', safemean([epinfo['r'] for epinfo in epinfobuf]))
                logger.logkv('eplenmean', safemean([epinfo['l'] for epinfo in epinfobuf]))
                if eval_env is not None:
                    logger.logkv('eval_eprewmean', safemean([epinfo['r'] for epinfo in eval_epinfobuf]) )
                    logger.logkv('eval_eplenmean', safemean([epinfo['l'] for epinfo in eval_epinfobuf]) )
                logger.logkv('misc/time_elapsed', tnow - tfirststart)
                for (lossval, lossname) in zip(lossvals, model.loss_names):
                    logger.logkv('loss/' + lossname, lossval)

                logger.dumpkvs()

            if env_type=='corl':
                #models_ls.append(copy(model))
                eprewmean_ls.append(safemean([epinfo['r'] for epinfo in epinfobuf]))
                if env.env_name=='ah' and save_interval and (update % save_interval == 0 or update == 1) and logger.get_dir() and is_mpi_root:
                    checkdir = osp.join(logger.get_dir(), 'checkpoints')
                    os.makedirs(checkdir, exist_ok=True)
                    savepath = osp.join(checkdir, '%.5i'%update)
                    #print('Saving to', savepath)
                    model.save(savepath)

        if plot_single_loss and env_type=='corl':
            plt.plot(eprewmean_ls)
            plt.ylabel('Average Return Over 100 Episodes')
            plt.xlabel('PPO Updates')
            if env_type=='corl' and (env.env_name=='ah' or env.env_name=='real_ah'):
                if env.with_obs:
                    suf='_obs_idx_'+str(env.obs_idx)
                    o_i=env.obs_idx
                else:
                    suf='_obs_idx_20'
                    o_i=20
                if o_i==20 and not env.state_with_goal_loc:
                    suf+='_no_goal_loc_'+str(env.goal_loc_idx)
                else:
                    if o_i==14:
                        suf+='_no_goal_loc'
                    else:
                        suf+='_with_goal_loc'
                if env.with_obs:
                    suf+='_withobs'
                else:
                    suf+='_noobs'
            elif env_type=='corl' and env.env_name=='corl_Acrobot-v1':
                suf='_goal_height_'+str(env.goal_height)
            else:
                suf=''
            suf+=single_loss_suf
            #plt.savefig('./ppo2_results/single_loss/'+env.env_name+'_single_seed_'+str(seed)+suf+'_loss.png')
            #plt.savefig('./ppo2_results/test_ah_single_loss'+single_loss_suf+'/'+env.env_name +'_lr_'+str(lr_factor)+'_total_timesteps_'+str(total_timesteps)+'_single_seed_'+str(seed)+suf+'_loss.png')
            if env.env_name=='corl_Reacher-v2' or env.env_name=='corl_Acrobot-v1':
                if not os.path.exists('./ppo2_results/single_loss_bkup/'):
                    os.makedirs('./ppo2_results/single_loss_bkup/')
                #plt.savefig('./ppo2_results/single_loss/'+env.env_name+'_single_seed_'+str(seed)+suf+'_loss.png')
                plt.savefig('./ppo2_results/single_loss_bkup/'+env.env_name+'_single_seed_'+str(seed)+suf+'_loss.png')
            else:
                #plt.savefig('./ppo2_results/single_loss/'+env.env_name+'_single_seed_'+str(seed)+suf+'_loss.png')
                plt.savefig('./ppo2_results/test_'+env.env_name+'_single_loss'+single_loss_suf+'/'+env.env_name +'_lr_'+str(lr_factor)+'_total_timesteps_'+str(total_timesteps)+'_single_seed_'+str(seed)+suf+'_loss.png')
            
        if env_type=='corl':
            if env.env_name=='ah' or env.env_name=='real_ah':
                best_update=np.argmax(eprewmean_ls)
                best_update_a=best_update-1
                #best_model=models_ls[best_update]
                #best_model_a=models_ls[best_update_a]
                #best_model_path = osp.join(logger.get_dir(), 'best_model_'+'%.5i'%best_update)
                #best_model_path_a = osp.join(logger.get_dir(), 'best_model2_'+'%.5i'%best_update_a)
                #print('Saving best model to', best_model_path)
                #print('Saving best model2 to', best_model_path_a)
                logger.logkv('best update 1', best_update)
                logger.logkv('best update 2', best_update_a)
                #best_model.save(best_model_path)
                #best_model_a.save(best_model_path_a)
                if find_best and len(best_eps_ret_ls)!=0:
                    logger.logkv('best eps return update: update',best_eps_update_ls[np.argmax(best_eps_ret_ls)])
                    logger.logkv('best eps return:',np.max(best_eps_ret_ls))
                    logger.logkv('best eps length update: update',best_eps_update_ls[np.argmin(best_eps_len_ls)])
                    logger.logkv('best eps length:',np.min(best_eps_len_ls))
                logger.dumpkvs()
            #else:
            #    best_model=model
            #    best_model_a=model


    else:
        if compare:
            num_eval_eps=1
        for eval_eps in range(num_eval_eps):
            tstart = time.perf_counter()
            lrnow = lr(1.0)
            cliprangenow = cliprange(1.0)
            logger.info('Evaluation: Stepping environment...')
            obs, returns, masks, actions, values, neglogpacs, states, epinfos, final_obs, succ, best_eps_ret, best_eps_len = runner.run(do_eval=True,num_eval_eps=num_eval_eps,compare=compare,compare_ah_idx=compare_ah_idx,reacher_sd=reacher_sd,acrobot_sd=acrobot_sd) #pylint: disable=E0632
            logger.info('Evaluation: Done.')
            do_eval_epinfobuf=deque(maxlen=100)
            do_eval_epinfobuf.extend(epinfos)
            mblossvals = []
            if states is None:
                inds = np.arange(obs.shape[0])
                for _ in range(noptepochs):
                    np.random.shuffle(inds)
                    for start in range(0, obs.shape[0], nbatch_train):
                        end = start + nbatch_train
                        mbinds = inds[start:end]
                        slices = (arr[mbinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                        mblossvals.append(model.train(lrnow, cliprangenow, *slices, do_eval=True))
            lossvals = np.mean(mblossvals, axis=0)
            tnow = time.perf_counter()
            fps = int(obs.shape[0] / (tnow - tstart))
            ev = explained_variance(values, returns)
            logger.logkv('misc/time_elapsed', tnow - tstart)
            logger.logkv("misc/total_timesteps", obs.shape[0])
            logger.logkv("misc/nupdates", 0)
            logger.logkv("misc/serial_timesteps", obs.shape[0])
            logger.logkv("fps", fps)
            logger.logkv("misc/explained_variance", float(ev))
            logger.logkv('eprewmean', safemean([epinfo['r'] for epinfo in do_eval_epinfobuf]))
            logger.logkv('eplenmean', safemean([epinfo['l'] for epinfo in do_eval_epinfobuf]))
            for (lossval, lossname) in zip(lossvals, model.loss_names):
                logger.logkv('loss/' + lossname, lossval)
            logger.dumpkvs()
            start_index=0
            observ=obs[start_index:start_index+do_eval_epinfobuf[0]['l'],:]
            observ=np.concatenate([observ,final_obs[0]],axis=0)
            if not compare:
                plot_eval_eps(seed,observ,env,eval_eps,compare)
            else:
                if env.env_name=='ah' or env.env_name=='real_ah':
                    plot_eval_eps(ho,seed,observ,env,compare_ah_idx,compare,pre_suf=single_loss_suf)
                elif env.env_name=='corl_Reacher-v2':
                    plot_eval_eps(ho,seed,observ,env,reacher_sd,compare)
                elif env.env_name=='corl_Acrobot-v1':
                    plot_eval_eps(ho,seed,observ,env,acrobot_sd,compare)
    return model

def plot_eval_eps(ho,seed,observ,env,idx,compare,pre_suf=''):
    if env.env_name=='ah':
        if env.state_with_goal_loc:
            goal_loc=observ[0,4:6]
            if env.state_with_goal_radius:
                big_goal_radius=goal_loc[0,6]/0.6875
            else:
                big_goal_radius=env.goal_radius/0.6875
        else:
            goal_loc=env.goal_loc
            big_goal_radius=env.goal_radius/0.6875
        fig, ax = plt.subplots(figsize=(8,3.5))
        H = np.concatenate((np.array(env.H1)[:,:], env.H2), axis=0)
        pgon = plt.Polygon(H, color='y', alpha=1, zorder=0)
        ax.add_patch(pgon)
        goal_plan = plt.Circle((goal_loc[0], goal_loc[1]), big_goal_radius*0.6875, color='m')
        #goal_plan = plt.Circle((goal_loc[0], goal_loc[1]), big_goal_radius, color='m')
        ax.add_artist(goal_plan)
        if env.with_obs:
            for o in env.Obs:
                if in_hull(np.array(o[:2]),env.H1D,env.H2D):
                    obs = plt.Circle(o[:2], env.obs_dist, color=[0.4,0.4,0.4])
                    ax.add_artist(obs)
        plt.plot(0, 119, 'o', markersize=16, color ='r')
        plt.plot(observ[:,0],observ[:,1],'-k')
        plt.ylim([60, 140])
        plt.xlabel('x')
        plt.ylabel('y')
        if not compare:
            suffix='_not'
        else:
            suffix=''
        if env.with_obs:
            if env.obs_idx==20 and env.state_with_goal_loc:
                goal_loc_suffix='_withobs_with_goal_loc'
            else:
                goal_loc_suffix='_withobs_without_goal_loc'
        else:
            if env.state_with_goal_loc:
                goal_loc_suffix='_noobs_with_goal_loc'
            else:
                goal_loc_suffix='_noobs_without_goal_loc'
        if env.with_obs:
            plt.savefig('./ppo2_results/eval/Eval_model_seed_'+str(seed)+'_lr_'+str(lr_factor)+'_total_timesteps_'+str(total_timesteps)+'_obs_idx_'+str(env.obs_idx)+goal_loc_suffix+'_'+env.env_name+'_'+str(idx)+pre_suf+suffix+'_compare.png',dpi=200)
        else:
            plt.savefig('./ppo2_results/eval/Eval_model_seed_'+str(seed)+'_lr_'+str(lr_factor)+'_total_timesteps_'+str(total_timesteps)+'_obs_idx_20'+goal_loc_suffix+'_'+env.env_name+'_'+str(idx)+pre_suf+suffix+'_compare.png',dpi=200)
    
    elif env.env_name=='corl_Reacher-v2':
        goal_loc=observ[0,-2:]
        big_goal_radius=0.02
        fig, ax = plt.subplots(figsize=(8,8))
        goal_plan = plt.Circle((goal_loc[0], goal_loc[1]), big_goal_radius, color='m')
        ax.add_artist(goal_plan)
        plt.scatter(observ[0,6], observ[0,7], s=150, marker='*', color ='k',label='start')
        plt.plot(observ[:,6],observ[:,7],'-k',label='Trajectory')
        plt.xlim([-0.22, 0.22])
        plt.ylim([-0.22, 0.22])
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        if ho!=0:
            pre_suf='_ho'+str(ho)
        if not compare:
            plt.savefig('./ppo2_results/eval/Eval_model_seed_'+str(seed)+'_lr_'+str(lr_factor)+'_total_timesteps_'+str(total_timesteps)+'_'+env.env_name+'_'+str(idx)+pre_suf+'_not_compare.png',dpi=200)
        else:
            plt.savefig('./ppo2_results/eval/Eval_model_seed_'+str(seed)+'_lr_'+str(lr_factor)+'_total_timesteps_'+str(total_timesteps)+'_'+env.env_name+'_'+str(idx)+pre_suf+'_compare.png',dpi=200)

    elif env.env_name=='corl_Acrobot-v1':
        fig, ax = plt.subplots(figsize=(8,8))
        initial_state=observ[0,:]
        plt.scatter(0, -initial_state[0]-(initial_state[0]*initial_state[2]-initial_state[1]*initial_state[3]), s=150, marker='*', color ='k',label='start')
        plt.plot(-observ[:,0]-(observ[:,0]*observ[:,2]-observ[:,1]*observ[:,3]),'-k',label='Trajectory of Y position')
        plt.ylim([-2.2, 2.2])
        plt.xlabel('Steps')
        plt.ylabel('Y position')
        plt.legend()
        if not compare:
            plt.savefig('./ppo2_results/eval/Eval_model_seed_'+str(seed)+'_lr_'+str(lr_factor)+'_total_timesteps_'+str(total_timesteps)+'_goal_height_'+str(env.goal_height)+'_'+env.env_name+'_'+str(idx)+pre_suf+'_not_compare.png',dpi=200)
        else:
            plt.savefig('./ppo2_results/eval/Eval_model_seed_'+str(seed)+'_lr_'+str(lr_factor)+'_total_timesteps_'+str(total_timesteps)+'_goal_height_'+str(env.goal_height)+'_'+env.env_name+'_'+str(idx)+pre_suf+'_compare.png',dpi=200)
    
    elif env.env_name=='real_ah':
        if env.state_with_goal_loc:
            goal_loc=observ[0,4:6]
            if env.state_with_goal_radius:
                big_goal_radius=goal_loc[0,6]/0.6875
            else:
                big_goal_radius=env.goal_radius/0.6875
        else:
            goal_loc=env.goal_loc
            big_goal_radius=env.goal_radius/0.6875
        fig, ax = plt.subplots(figsize=(10,3.5))
        goal_plan = plt.Circle((goal_loc[0], goal_loc[1]), big_goal_radius*0.6875, color='m')
        #goal_plan = plt.Circle((goal_loc[0], goal_loc[1]), big_goal_radius, color='m')
        ax.add_artist(goal_plan)
        if env.with_obs:
            for o in env.Obs:
                obs = plt.Circle(o[:2], env.obs_dist, color=[0.4,0.4,0.4])
                ax.add_artist(obs)
        plt.plot(round(env.init_mu[0]), round(env.init_mu[1]), 'o', markersize=16, color ='r')
        plt.plot(observ[:,0],observ[:,1],'-k')
        plt.xlim([-50, 90])
        #plt.xlim([-60, 120])
        plt.ylim([70, 120])
        #plt.ylim([50, 120])
        plt.xlabel('x')
        plt.ylabel('y')
        if not compare:
            suffix='_not'
        else:
            suffix=''
        if env.with_obs:
            if env.obs_idx==20 and env.state_with_goal_loc:
                goal_loc_suffix='_withobs_with_goal_loc'
            else:
                goal_loc_suffix='_withobs_without_goal_loc'
        else:
            if env.state_with_goal_loc:
                goal_loc_suffix='_noobs_with_goal_loc'
            else:
                goal_loc_suffix='_noobs_without_goal_loc'
        if env.with_obs:
            plt.savefig('./ppo2_results/eval/Eval_model_seed_'+str(seed)+'_lr_'+str(lr_factor)+'_total_timesteps_'+str(total_timesteps)+'_obs_idx_'+str(env.obs_idx)+goal_loc_suffix+'_'+env.env_name+'_'+str(idx)+pre_suf+suffix+'_compare.png',dpi=200)
        else:
            plt.savefig('./ppo2_results/eval/Eval_model_seed_'+str(seed)+'_lr_'+str(lr_factor)+'_total_timesteps_'+str(total_timesteps)+'_obs_idx_20'+goal_loc_suffix+'_'+env.env_name+'_'+str(idx)+pre_suf+suffix+'_compare.png',dpi=200)
    
    else:
        raise NotImplementedError


