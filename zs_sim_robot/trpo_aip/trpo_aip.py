import sys
sys.stdout.flush()
from common.math_util import explained_variance
from common.misc_util import zipsame
from common import dataset
import logger
import common.tf_util as U
import tensorflow as tf, numpy as np
import time
from common.console_util import colorize
from collections import deque
from common.misc_util import set_global_seeds
from common.mpi_adam import MpiAdam
from common.cg import cg
from common.input import observation_placeholder
from common.policies import build_policy as build_policy_ref
from common.policies_aip import build_policy,build_env_ref,reacher_adjust_obs
from contextlib import contextmanager
import os
import os.path as osp
import matplotlib.pyplot as plt
try:
    from mpi4py import MPI
except ImportError:
    MPI = None
from ppo2.model import Model
from r_diff.r_diff import R_diff_model
from r_diff.exp_util import eval_policy, Policy
from r_diff.util.util import load_extracted_val_data as load_val_data
from r_diff.util.util import to_onehot
from r_diff.model_config import get_make_mlp_model
import torch, pickle

if not os.path.exists('./trpo_aip_results/eval/'):
    os.makedirs('./trpo_aip_results/eval/')
if not os.path.exists('./trpo_aip_results/single_loss/'):
    os.makedirs('./trpo_aip_results/single_loss/')

def normalize(data,x_std_arr,x_mean_arr):
    return (data - x_mean_arr[:data.shape[-1]]) / x_std_arr[:data.shape[-1]]

def denormalize(data,y_std_arr,y_mean_arr):
    return data * y_std_arr[:data.shape[-1]] + y_mean_arr[:data.shape[-1]]

def traj_segment_generator(pi, env, horizon, stochastic,ref_stochastic=False):
    # Initialize state variables
    t = 0
    ac = env.action_space.sample()
    new = True
    rew = 0.0
    ob = env.reset()

    cur_ep_ret = 0
    cur_ep_len = 0
    ep_rets = []
    ep_lens = []
    final_obs=[]

    # Initialize history arrays
    obs = np.array([ob for _ in range(horizon)])
    rews = np.zeros(horizon, 'float32')
    vpreds = np.zeros(horizon, 'float32')
    news = np.zeros(horizon, 'int32')
    acs = np.array([ac for _ in range(horizon)])
    ac_refs = np.array([ac for _ in range(horizon)])
    alphas = np.zeros(horizon, 'float32').reshape((-1,1))
    prevacs = acs.copy()

    while True:
        prevac = ac
        ac, vpred, _, _, ac_ref, alpha = pi.step(ob,stochastic=stochastic,ref_stochastic=ref_stochastic)
        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        if t > 0 and t % horizon == 0:
            yield {"ob" : obs, "ac_ref": ac_refs, "alpha": alphas, "rew" : rews, "vpred" : vpreds, "new" : news,
                    "ac" : acs, "prevac" : prevacs, "nextvpred": vpred * (1 - new),
                    "ep_rets" : ep_rets, "ep_lens" : ep_lens, "final_obs": np.array(final_obs)}
            _, vpred, _, _, _, _ = pi.step(ob, stochastic=stochastic,ref_stochastic=ref_stochastic)
            #_, vpred, _, _, ac_ref, alpha = pi.step(ob, stochastic=stochastic,ref_stochastic=ref_stochastic)
            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy
            ep_rets = []
            ep_lens = []
        i = t % horizon
        obs[i] = ob
        vpreds[i] = vpred
        news[i] = new
        acs[i] = ac
        ac_refs[i] = ac_ref
        alphas[i] = alpha
        prevacs[i] = prevac

        ob, rew, new, _ = env.step(ac)
        rews[i] = rew

        cur_ep_ret += rew
        cur_ep_len += 1
        if new:
            final_ob=ob.copy()
            final_obs.append(final_ob)
            ep_rets.append(cur_ep_ret)
            ep_lens.append(cur_ep_len)
            cur_ep_ret = 0
            cur_ep_len = 0
            ob = env.reset()
        t += 1

def add_vtarg_and_adv(seg, gamma, lam):
    new = np.append(seg["new"], 0) # last element is only used for last vtarg, but we already zeroed it if last new = 1
    vpred = np.append(seg["vpred"], seg["nextvpred"])
    T = len(seg["rew"])
    seg["adv"] = gaelam = np.empty(T, 'float32')
    rew = seg["rew"]
    lastgaelam = 0
    for t in reversed(range(T)):
        nonterminal = 1-new[t+1]
        delta = rew[t] + gamma * vpred[t+1] * nonterminal - vpred[t]
        gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
    seg["tdlamret"] = seg["adv"] + seg["vpred"]

def learn(*,
        network,
        env,
        policy_ref,
        env_id,
        env_type,
        total_timesteps,
        timesteps_per_batch=1024, # what to train on
        max_kl=0.001,
        cg_iters=10,
        gamma=0.99,
        lam=1.0, # advantage estimation
        seed=None,
        ent_coef=0.0,
        cg_damping=1e-2,
        vf_stepsize=3e-4,
        vf_iters =3,
        max_episodes=0, max_iters=0,  # time constraint
        callback=None,
        load_path=None,
        log_interval=1,
        need_eval=False, num_eval_eps=1, compare=False,compare_ah_idx=8,reacher_sd=1,acrobot_sd=1,
        plot_single_loss=False, single_loss_suf='',
        save_path=None,
        ho=None,
        lr_factor=None,
        find_best=None,

        ref_type='ppo',
        alpha_func='squared',
        ablation='auto',
        accurate=False,
        ref_stochastic=False,

        #For r_diff
        r_diff_train_freq=5,
        update_epochs=5, 
        batch_size=512,
        # For get_model
        num_fc=2,
        num_fwd_hidden=500,
        use_layer_norm=False, 
        **network_kwargs
        ):
    '''
    learn a policy function with TRPO algorithm

    Parameters:
    ----------

    network                 neural network to learn. Can be either string ('mlp', 'cnn', 'lstm', 'lnlstm' for basic types)
                            or function that takes input placeholder and returns tuple (output, None) for feedforward nets
                            or (output, (state_placeholder, state_output, mask_placeholder)) for recurrent nets

    env                     environment (one of the gym environments or wrapped via baselines.common.vec_env.VecEnv-type class

    timesteps_per_batch     timesteps per gradient estimation batch

    max_kl                  max KL divergence between old policy and new policy ( KL(pi_old || pi) )

    ent_coef                coefficient of policy entropy term in the optimization objective

    cg_iters                number of iterations of conjugate gradient algorithm

    cg_damping              conjugate gradient damping

    vf_stepsize             learning rate for adam optimizer used to optimie value function loss

    vf_iters                number of iterations of value function optimization iterations per each policy optimization step

    total_timesteps           max number of timesteps

    max_episodes            max number of episodes

    max_iters               maximum number of policy optimization iterations

    callback                function to be called with (locals(), globals()) each policy optimization step

    load_path               str, path to load the model from (default: None, i.e. no model is loaded)

    **network_kwargs        keyword arguments to the policy / network builder. See baselines.common/policies.py/build_policy and arguments to a particular type of network

    Returns:
    -------

    learnt model

    '''
    if env_id=='Reacher-v2':
        env_name='corl_Reacher-v2'
    if MPI is not None:
        nworkers = MPI.COMM_WORLD.Get_size()
        rank = MPI.COMM_WORLD.Get_rank()
    else:
        nworkers = 1
        rank = 0

    cpus_per_worker = 1
    U.get_session(config=tf.ConfigProto(
            allow_soft_placement=True,
            inter_op_parallelism_threads=cpus_per_worker,
            intra_op_parallelism_threads=cpus_per_worker
    ))

    policy = build_policy(env, network, value_network='copy', **network_kwargs)
    set_global_seeds(seed)

    np.set_printoptions(precision=3)
    # Setup losses and stuff
    # ----------------------------------------
    ob_space = env.observation_space
    ac_space = env.action_space

    if ref_type!='ppo':
        raise NotImplementedError

    make_model = get_make_mlp_model(num_fc=num_fc, num_fwd_hidden=num_fwd_hidden, layer_norm=use_layer_norm)
    r_diff_model = R_diff_model(env=env, env_id=env_id, make_model=make_model,
            #num_warm_start=num_warm_start,            
            #init_epochs=init_epochs, 
            update_epochs=update_epochs, 
            batch_size=batch_size, 
            **network_kwargs)

    ob = observation_placeholder(ob_space,name='Ob')
    ref_ac = observation_placeholder(ac_space,name='Ref_ac')
    alpha_holder=tf.placeholder(shape=(None,1), dtype=np.float32, name='Alpha')
    with tf.variable_scope("pi_aip"):
        pi = policy(ablation=str(ablation),env_type=env_type,env_id=env_id,ref_type=ref_type,policy_ref=policy_ref,r_diff_model=r_diff_model,alpha_func=alpha_func,observ_placeholder=ob,ac_placeholder=ref_ac,alpha_placeholder=alpha_holder)
    with tf.variable_scope("oldpi_aip"):
        oldpi = policy(ablation=str(ablation),env_type=env_type,env_id=env_id,ref_type=ref_type,policy_ref=policy_ref,r_diff_model=r_diff_model,alpha_func=alpha_func,observ_placeholder=ob,ac_placeholder=ref_ac,alpha_placeholder=alpha_holder)


    val_dataset = {'ob': None, 'ac': None, 'r_diff_label': None}
    atarg = tf.placeholder(dtype=tf.float32, shape=[None]) # Target advantage function (if applicable)
    ret = tf.placeholder(dtype=tf.float32, shape=[None]) # Empirical return
    ac = pi.pdtype.sample_placeholder([None])

    kloldnew = oldpi.pd.kl(pi.pd)
    ent = pi.pd.entropy()
    meankl = tf.reduce_mean(kloldnew)
    meanent = tf.reduce_mean(ent)
    entbonus = ent_coef * meanent

    vferr = tf.reduce_mean(tf.square(pi.vf - ret))

    ratio = tf.exp(pi.pd.logp(ac) - oldpi.pd.logp(ac)) # advantage * pnew / pold
    surrgain = tf.reduce_mean(ratio * atarg)

    optimgain = surrgain + entbonus
    losses = [optimgain, meankl, entbonus, surrgain, meanent]
    loss_names = ["optimgain", "meankl", "entloss", "surrgain", "entropy"]

    dist = meankl

    all_var_list = get_trainable_variables("pi_aip")
    # var_list = [v for v in all_var_list if v.name.split("/")[1].startswith("pol")]
    # vf_var_list = [v for v in all_var_list if v.name.split("/")[1].startswith("vf")]
    var_list = get_pi_trainable_variables("pi_aip")
    vf_var_list = get_vf_trainable_variables("pi_aip")

    vfadam = MpiAdam(vf_var_list)

    get_flat = U.GetFlat(var_list)
    set_from_flat = U.SetFromFlat(var_list)
    klgrads = tf.gradients(dist, var_list)
    flat_tangent = tf.placeholder(dtype=tf.float32, shape=[None], name="flat_tan")
    shapes = [var.get_shape().as_list() for var in var_list]
    start = 0
    tangents = []
    for shape in shapes:
        sz = U.intprod(shape)
        tangents.append(tf.reshape(flat_tangent[start:start+sz], shape))
        start += sz
    gvp = tf.add_n([tf.reduce_sum(g*tangent) for (g, tangent) in zipsame(klgrads, tangents)]) #pylint: disable=E1111
    fvp = U.flatgrad(gvp, var_list)

    assign_old_eq_new = U.function([],[], updates=[tf.assign(oldv, newv)
        for (oldv, newv) in zipsame(get_variables("oldpi_aip"), get_variables("pi_aip"))])

    compute_losses = U.function([ob, ac, atarg,ref_ac,alpha_holder], losses)
    compute_lossandgrad = U.function([ob, ac, atarg,ref_ac,alpha_holder], losses + [U.flatgrad(optimgain, var_list)])
    compute_fvp = U.function([flat_tangent, ob, ac, atarg,ref_ac,alpha_holder], fvp)
    compute_vflossandgrad = U.function([ob, ret], U.flatgrad(vferr, vf_var_list))

    @contextmanager
    def timed(msg):
        if rank == 0:
            print(colorize(msg, color='magenta'))
            tstart = time.time()
            yield
            print(colorize("done in %.3f seconds"%(time.time() - tstart), color='magenta'))
        else:
            yield

    def allmean(x):
        assert isinstance(x, np.ndarray)
        if MPI is not None:
            out = np.empty_like(x)
            MPI.COMM_WORLD.Allreduce(x, out, op=MPI.SUM)
            out /= nworkers
        else:
            out = np.copy(x)

        return out

    U.initialize()
    if load_path is not None:
        pi.load(load_path)

    th_init = get_flat()
    if MPI is not None:
        MPI.COMM_WORLD.Bcast(th_init, root=0)

    set_from_flat(th_init)
    vfadam.sync()
    print("Init param sum", th_init.sum(), flush=True)

    # Prepare for rollouts
    # ----------------------------------------
    seg_gen = traj_segment_generator(pi, env, timesteps_per_batch, stochastic=True,ref_stochastic=ref_stochastic)

    episodes_so_far = 0
    timesteps_so_far = 0
    iters_so_far = 0
    eprewmean_ls=[]
    tstart = time.time()
    lenbuffer = deque(maxlen=100) # rolling buffer for episode lengths
    rewbuffer = deque(maxlen=100) # rolling buffer for episode rewards

    #if sum([max_iters>0, total_timesteps>0, max_episodes>0])==0:
        # noththing to be done
    #    return pi

    assert sum([max_iters>0, total_timesteps>0, max_episodes>0]) < 2, \
        'out of max_iters, total_timesteps, and max_episodes only one should be specified'
    if not (need_eval and env_type=='corl'):
        while True:
            if callback: callback(locals(), globals())
            if total_timesteps and timesteps_so_far >= total_timesteps:
                break
            elif max_episodes and episodes_so_far >= max_episodes:
                break
            elif max_iters and iters_so_far >= max_iters:
                break
            logger.log("********** Iteration %i ************"%iters_so_far)

            with timed("sampling"):
                if sys.version[0]=='3':
                    seg = seg_gen.__next__()
                elif sys.version[0]=='2':
                    seg = seg_gen.next()
                else:
                    raise NotImplementedError
            add_vtarg_and_adv(seg, gamma, lam)
            
            # ob, ac, atarg, ret, td1ret = map(np.concatenate, (obs, acs, atargs, rets, td1rets))
            ob, ac, atarg, tdlamret, ref_ac, alpha_holder = seg["ob"], seg["ac"], seg["adv"], seg["tdlamret"], seg["ac_ref"], seg["alpha"]

            imrwd = seg['rew']  

            vpredbefore = seg["vpred"] # predicted value function before udpate
            atarg = (atarg - atarg.mean()) / atarg.std() # standardized advantage function estimate

            if hasattr(pi, "ret_rms"): pi.ret_rms.update(tdlamret)
            if hasattr(pi, "rms"): pi.rms.update(ob) # update running mean/std for policy
            
            args = seg["ob"], seg["ac"], atarg, ref_ac, alpha_holder
            fvpargs = [arr[::5] for arr in args]
            def fisher_vector_product(p):
                return allmean(compute_fvp(p, *fvpargs)) + cg_damping * p

            assign_old_eq_new() # set old parameter values to new parameter values
            with timed("computegrad"):
                *lossbefore, g = compute_lossandgrad(*args)
            lossbefore = allmean(np.array(lossbefore))
            g = allmean(g)
            if np.allclose(g, 0):
                logger.log("Got zero gradient. not updating")
            else:
                with timed("cg"):
                    stepdir = cg(fisher_vector_product, g, cg_iters=cg_iters, verbose=rank==0)
                assert np.isfinite(stepdir).all()
                shs = .5*stepdir.dot(fisher_vector_product(stepdir))
                lm = np.sqrt(shs / max_kl)
                # logger.log("lagrange multiplier:", lm, "gnorm:", np.linalg.norm(g))
                fullstep = stepdir / lm
                expectedimprove = g.dot(fullstep)
                surrbefore = lossbefore[0]
                stepsize = 1.0
                thbefore = get_flat()
                for _ in range(10):
                    thnew = thbefore + fullstep * stepsize
                    set_from_flat(thnew)
                    meanlosses = surr, kl, *_ = allmean(np.array(compute_losses(*args)))
                    improve = surr - surrbefore
                    logger.log("Expected: %.3f Actual: %.3f"%(expectedimprove, improve))
                    if not np.isfinite(meanlosses).all():
                        logger.log("Got non-finite value of losses -- bad!")
                    elif kl > max_kl * 1.5:
                        logger.log("violated KL constraint. shrinking step.")
                    elif improve < 0:
                        logger.log("surrogate didn't improve. shrinking step.")
                    else:
                        logger.log("Stepsize OK!")
                        break
                    stepsize *= .5
                else:
                    logger.log("couldn't compute a good step")
                    set_from_flat(thbefore)
                if nworkers > 1 and iters_so_far % 20 == 0:
                    paramsums = MPI.COMM_WORLD.allgather((thnew.sum(), vfadam.getflat().sum())) # list of tuples
                    assert all(np.allclose(ps, paramsums[0]) for ps in paramsums[1:])
            for (lossname, lossval) in zip(loss_names, meanlosses):
                logger.record_tabular(lossname, lossval)

            with timed("vf"):
                for _ in range(vf_iters):
                    for (mbob, mbret) in dataset.iterbatches((seg["ob"], seg["tdlamret"]),
                    include_final_partial_batch=False, batch_size=64):
                        g = allmean(compute_vflossandgrad(mbob, mbret))
                        vfadam.update(g, vf_stepsize)
            
            with timed('r_diff_model'):         
                if env_id=='Reacher-v2':
                    model_path='./trans_model_data/Reacher-v2_model/Reacher-v2_model_lr0.0001_nodes512_seed0_ho0.999_epochs_100'
                    with open(model_path, 'rb') as pickle_file:
                        reacher_model = torch.load(pickle_file, map_location='cpu')
                    norm_path='./trans_model_data/Reacher-v2_normalization/normalization_arr_ho0.999'
                    with open(norm_path, 'rb') as pickle_file:
                        x_norm_arr, y_norm_arr = pickle.load(pickle_file)
                        x_mean_arr, x_std_arr = x_norm_arr[0], x_norm_arr[1]
                        y_mean_arr, y_std_arr = y_norm_arr[0], y_norm_arr[1]

                    clipped_ac = np.clip(ac, np.array([-1.,-1.]), np.array([-1.,-1.]))
                    goal_loc=ob[:,4:6]
                    sa=np.concatenate(([reacher_adjust_obs(ob)[:,:8],clipped_ac]),axis=1) # ref_ac dim??
                    inpt = normalize(sa,x_std_arr,x_mean_arr)
                    inpt = torch.tensor(inpt, dtype=torch.float)
                    state_delta = reacher_model(inpt)    
                    state_delta = state_delta.detach().numpy()
                    state_delta = denormalize(state_delta,y_std_arr,y_mean_arr)
                    next_state = sa[:,:8] + state_delta
                    
                    dm_imrwd=-np.linalg.norm(goal_loc-next_state[:,6:8],axis=1)-np.sum(np.square(ac),axis=1)
                    if accurate:
                        r_diff_label=np.clip(np.abs((dm_imrwd-imrwd)/np.abs(dm_imrwd)),a_min=None, a_max=1)
                    else:
                        r_diff_label=np.clip((dm_imrwd-imrwd)/np.abs(dm_imrwd),0,1)

                    #r_diff_label=np.clip(100*(dm_imrwd-imrwd),0,1)
                    #print(r_diff_label)

                    r_diff_model.add_data_batch(ob, ac, r_diff_label)
                    if (iters_so_far+1) % r_diff_train_freq==0:
                        r_diff_model.update_forward_dynamic(require_update=True, 
                            ob_val=val_dataset['ob'], ac_val=val_dataset['ac'], r_diff_label_val=val_dataset['r_diff_label'])
                        pi.update_r_diff_model(r_diff_model)
                        oldpi.update_r_diff_model(r_diff_model)
                else:
                    raise NotImplementedError

            logger.record_tabular("ev_tdlam_before", explained_variance(vpredbefore, tdlamret))

            lrlocal = (seg["ep_lens"], seg["ep_rets"]) # local values
            if MPI is not None:
                listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal) # list of tuples
            else:
                listoflrpairs = [lrlocal]

            lens, rews = map(flatten_lists, zip(*listoflrpairs))
            lenbuffer.extend(lens)
            rewbuffer.extend(rews)
            logger.record_tabular("EpLenMean", np.mean(lenbuffer))
            logger.record_tabular("EpRewMean", np.mean(rewbuffer))
            logger.record_tabular("EpThisIter", len(lens))
            episodes_so_far += len(lens)
            timesteps_so_far += sum(lens)
            iters_so_far += 1

            logger.record_tabular("EpisodesSoFar", episodes_so_far)
            logger.record_tabular("TimestepsSoFar", timesteps_so_far)
            logger.record_tabular("TimeElapsed", time.time() - tstart)

            if rank==0:
                logger.dump_tabular()

            if env_type=='corl':
                eprewmean_ls.append(np.mean(rewbuffer))
                if env.env_name=='ah' and logger.get_dir():
                    checkdir = osp.join(logger.get_dir(), 'checkpoints')
                    os.makedirs(checkdir, exist_ok=True)
                    savepath = osp.join(checkdir, '%.5i'%iters_so_far)
                    pi.save(savepath)
        '''
        if plot_single_loss and env_type=='corl':
            plt.plot(eprewmean_ls)
            plt.ylabel('Average Return Over 100 Episodes')
            plt.xlabel('TRPO Updates') 
            if env_type=='corl' and env.env_name=='ah':
                if env.with_obs:
                    suf='_obs_idx_'+str(env.obs_idx)
                    o_i=env.obs_idx
                else:
                    suf='_obs_idx_20'
                    o_i=20
                if o_i==20 and not env.state_with_goal_loc:
                    suf+='_no_goal_loc_'+str(env.goal_loc_idx)
                else:
                    suf+=''
                if env.with_obs:
                    suf+='_withobs'
                else:
                    suf+='_noobs'
            elif env_type=='corl' and env.env_name=='corl_Acrobot-v1':
                suf='_goal_height_'+str(env.goal_height)
            else:
                suf=''
            suf+=single_loss_suf
            #plt.savefig('./trpo_results/single_loss/'+env.env_name+'_single_seed_'+str(seed)+suf+'_loss.png')
            plt.savefig('./trpo_results/test_ah_single_loss'+single_loss_suf+'/'+env.env_name+'_single_seed_'+str(seed)+suf+'_loss.png')

        if env_type=='corl':
            if env.env_name=='ah':
                best_update=np.argmax(eprewmean_ls)
                best_update_a=best_update-1
                logger.logkv('best update 1', best_update)
                logger.logkv('best update 2', best_update_a)
                logger.dumpkvs()

    else:
        if compare:
            num_eval_eps=1
        for eval_eps in range(num_eval_eps):
            lenbuffer = deque(maxlen=100) # rolling buffer for episode lengths
            rewbuffer = deque(maxlen=100) # rolling buffer for episode rewards
            logger.log("********** Evaluation %i ************"%eval_eps+1)

            with timed("sampling"):
                seg = seg_gen.__next__()
            add_vtarg_and_adv(seg, gamma, lam)

            # ob, ac, atarg, ret, td1ret = map(np.concatenate, (obs, acs, atargs, rets, td1rets))
            ob, ac, atarg, tdlamret, final_obs = seg["ob"], seg["ac"], seg["adv"], seg["tdlamret"], seg["final_obs"]
            vpredbefore = seg["vpred"] # predicted value function before udpate
            atarg = (atarg - atarg.mean()) / atarg.std() # standardized advantage function estimate

            args = seg["ob"], seg["ac"], atarg
            fvpargs = [arr[::5] for arr in args]
            #def fisher_vector_product(p):
            #    return allmean(compute_fvp(p, *fvpargs)) + cg_damping * p

            assign_old_eq_new() # set old parameter values to new parameter values

            meanlosses = surr, kl, *_ = allmean(np.array(compute_losses(*args)))

            for (lossname, lossval) in zip(loss_names, meanlosses):
                logger.record_tabular(lossname, lossval)

            logger.record_tabular("ev_tdlam_before", explained_variance(vpredbefore, tdlamret))

            lrlocal = (seg["ep_lens"], seg["ep_rets"]) # local values
            if MPI is not None:
                listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal) # list of tuples
            else:
                listoflrpairs = [lrlocal]

            lens, rews = map(flatten_lists, zip(*listoflrpairs))
            lenbuffer.extend(lens)
            rewbuffer.extend(rews)
            logger.record_tabular("EpLenMean", np.mean(lenbuffer))
            logger.record_tabular("EpRewMean", np.mean(rewbuffer))
            logger.record_tabular("EpThisIter", len(lens))
            episodes_so_far += len(lens)
            timesteps_so_far += sum(lens)
            iters_so_far += 1

            logger.record_tabular("EpisodesSoFar", episodes_so_far)
            logger.record_tabular("TimestepsSoFar", timesteps_so_far)
            logger.record_tabular("TimeElapsed", time.time() - tstart)

            if rank==0:
                logger.dump_tabular()

            start_index=0
            observ=ob[start_index:start_index+lenbuffer[0],:]
            observ=np.concatenate([observ,final_obs[0]],axis=0)
            if not compare:
                plot_eval_eps(seed,observ,env,eval_eps,compare)
            else:
                if env.env_name=='ah':
                    plot_eval_eps(seed,observ,env,compare_ah_idx,compare,pre_suf=single_loss_suf)
                elif env.env_name=='corl_Reacher-v2':
                    plot_eval_eps(seed,observ,env,reacher_sd,compare)
                elif env.env_name=='corl_Acrobot-v1':
                    plot_eval_eps(seed,observ,env,acrobot_sd,compare)
'''
    return pi
'''
def plot_eval_eps(seed,observ,env,idx,compare,pre_suf=''):
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
            plt.savefig('./trpo_results/eval/Eval_model_seed_'+str(seed)+'_obs_idx_'+str(env.obs_idx)+goal_loc_suffix+'_'+env.env_name+'_'+str(idx)+pre_suf+suffix+'_compare.png',dpi=200)
        else:
            plt.savefig('./trpo_results/eval/Eval_model_seed_'+str(seed)+'_obs_idx_20'+goal_loc_suffix+'_'+env.env_name+'_'+str(idx)+pre_suf+suffix+'_compare.png',dpi=200)
    
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
        if not compare:
            plt.savefig('./trpo_results/eval/Eval_model_seed_'+str(seed)+'_'+env.env_name+'_'+str(idx)+pre_suf+'_not_compare.png',dpi=200)
        else:
            plt.savefig('./trpo_results/eval/Eval_model_seed_'+str(seed)+'_'+env.env_name+'_'+str(idx)+pre_suf+'_compare.png',dpi=200)

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
            plt.savefig('./trpo_results/eval/Eval_model_seed_'+str(seed)+'_goal_height_'+str(env.goal_height)+'_'+env.env_name+'_'+str(idx)+pre_suf+'_not_compare.png',dpi=200)
        else:
            plt.savefig('./trpo_results/eval/Eval_model_seed_'+str(seed)+'_goal_height_'+str(env.goal_height)+'_'+env.env_name+'_'+str(idx)+pre_suf+'_compare.png',dpi=200)
    else:
        raise NotImplementedError
'''

def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]

def get_variables(scope):
    return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope)

def get_trainable_variables(scope):
    return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)

def get_vf_trainable_variables(scope):
    return [v for v in get_trainable_variables(scope) if 'vf' in v.name[len(scope):].split('/')]

def get_pi_trainable_variables(scope):
    return [v for v in get_trainable_variables(scope) if 'pi' in v.name[len(scope):].split('/')]

