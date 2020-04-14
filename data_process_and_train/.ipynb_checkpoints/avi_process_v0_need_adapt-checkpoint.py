from avi_process_params import *
from sys import argv
#argv idx1 idx2 idx3 idx4 idx5 ... nm ne suffxi(avi_,avi_v1_,...)
mix=False
dm='nm'
train_mode='ne'
suffix='avi_'
if len(argv)>1:
    mix=True
    if len(argv)<=3:
        if argv[0][-10:]=='process.py':
            mix_idx_ls=[int(argv[i+1]) for i in range(len(argv)-1)]
        else:
            mix=False
    elif argv[-3][-1]!='m':
        mix_idx_ls=[int(argv[i+1]) for i in range(len(argv)-1)]
    else:
        dm=argv[-3]
        train_mode=argv[-2]
        suffix=argv[-1]
        mix_idx_ls=[int(argv[i+1]) for i in range(len(argv)-4)]
        if mix_idx_ls==[]:
            mix=False
data_mode=data_mode[:-2]+dm
import glob,pickle,os,copy
import numpy as np

def medfilter(x, W):
    w = int(W/2)
    x_new = np.copy(x)
    for i in range(1, x.shape[0]-1):
        if i < w:
            x_new[i] = np.mean(x[:i+w])
        elif i > x.shape[0]-w:
            x_new[i] = np.mean(x[i-w:])
        else:
            x_new[i] = np.mean(x[i-w:i+w])
    return x_new
def get_test_ground_truth(test_ds,test_state_dim,test_action_dim):
    test_states,test_actions,test_next_states=np.split(test_ds,[test_state_dim,test_state_dim+test_action_dim],axis=1)
    test_traj=[test_states[0,:]]
    for i in range(test_states.shape[0]-1):
        if (test_next_states[i,:]==test_states[i+1,:]).all():
            test_traj.append(test_next_states[i,:])
        else:
            test_traj.append(test_next_states[i,:])
            test_traj.append(test_states[i+1,:])
    test_traj.append(test_next_states[-1,:])
    return test_traj

def f_valid(states):
    ss=np.concatenate((states,np.roll(states,-1,axis=0)),axis=1)
    prev=ss[:,:2]
    nxt=ss[:,states.shape[1]:states.shape[1]+2]
    conti=np.linalg.norm(prev-nxt,axis=1)<=1.2
    return conti


with_finger,with_angle,with_start_state,with_med_filter=False,False,True,False
if 'wf' in data_mode:
    with_finger=True
if 'wa' in data_mode:
    with_angle=True
if 'ns' in data_mode:
    with_start_state=False
if 'wm' in data_mode:
    with_med_filter=True
obj_dir=base_path+color+'_data/'+obj
if not os.path.exists(base_path+color+'_data/'+obj):
    os.makedirs(obj_dir)
test_dir=obj_dir+'/test'
if not os.path.exists(test_dir):
    os.makedirs(test_dir)
    
train_path=base_path+color+'_data/'+'raw_t42_'+obj+'_'+data_type+'.obj'
with open(train_path,'rb') as filehandler:
    memory=pickle.load(filehandler,encoding='latin1')

states_ls,actions_ls,next_states_ls=[],[],[]
sas_ls,sasc_ls=[],[]
for eps in memory:
    states=eps['states']
    actions=eps['actions']
    next_states=eps['next_states']
    
    if with_angle:
        if with_finger:
            states = states[:,[0,1,11,12,2,3,4,5,6,7,8,9,10]]
            next_states = next_states[:,[0,1,11,12,2,3,4,5,6,7,8,9,10]]
        else:
            states = states[:,[0,1,11,12,2]]  
            next_states = next_states[:,[0,1,11,12,2]] 
    else:
        if with_finger:
            states = states[:,[0,1,11,12,3,4,5,6,7,8,9,10]]
            next_states = next_states[:,[0,1,11,12,3,4,5,6,7,8,9,10]]
        else:
            states = states[:,[0, 1, 11, 12]]
            next_states = next_states[:,[0, 1, 11, 12]]
            
    if with_med_filter:
        h = [40, 40, 100, 100]
        for i in range(states.shape[1]):
            if i == 4:
                continue
            try:
                states[:,i] = medfilter(states[:,i], h[i])
            except:
                states[:,i] = medfilter(states[:,i], 40)
            next_states[:-1,i]=states[1:,i]
            w=int(h[i]/2)
            next_states[-1,i]=np.mean(next_states[next_states.shape[0]-1-w:,i])
    else:
        states=states   
        
    if with_start_state:
        start_state=states[0,:]
        grasp_states=np.tile(start_state, (states.shape[0], 1))
        actions=np.concatenate((actions,grasp_states),axis=1)
    else:
        actions=actions
    states_ls.append(states)
    actions_ls.append(actions[:states.shape[0]])
    next_states_ls.append(next_states)
    sas=np.concatenate([states,actions[:states.shape[0]],next_states],axis=1)
    checks=f_valid(states)
    sasc=np.concatenate([sas,checks],axis=1)
    #sas_ls.append(sas)
    sas_ls.append(sas[checks])
    sasc_ls.append(sasc)

train_state_dim=states_ls[0].shape[1]
train_action_dim=actions_ls[0].shape[1]
#train_ds_path=obj_dir+'/avi_train_separate_'+data_type+data_mode
train_ds_path=obj_dir+'/train_separate_'+data_type+data_mode+'_'+suffix+'f'
with open(train_ds_path,'wb') as f:
    #pickle.dump([sas_ls,sas_ls,train_state_dim,train_action_dim,[get_test_ground_truth(sas,train_state_dim,train_action_dim) for sas in sas_ls],actions_ls],f)
    train_traj_gt_ls=[get_test_ground_truth(sas,train_state_dim,train_action_dim) for sas in sas_ls]
    pickle.dump([sas_ls,sasc_ls,train_state_dim,train_action_dim,train_traj_gt_ls,actions_ls],f)
    
test_path=base_path+color+'_data/'+'testpaths_'+obj+'_'+data_type+'_v1.pkl'
with open(test_path,'rb') as filehandler:
    trajectory=pickle.load(filehandler,encoding='latin1')
test_paths = trajectory[1]
action_seq = trajectory[0]
states_ls,test_actions_ls,next_states_ls=[],[],[]
test_sas_ls,test_sasc_ls=[],[]
#for path_inx in range(0,len(test_paths)):
for path_inx in range(3):###only first 3 trajs for cyl30 have almost all valid states, thus can be used to calculate test loss approximately.
    states = test_paths[path_inx]
    actions = action_seq[path_inx]
    if with_angle:
        if with_finger:
            states = states[:,[0,1,11,12,2,3,4,5,6,7,8,9,10]]
        else:
            states = states[:,[0,1,11,12,2]]  
    else:
        if with_finger:
            states = states[:,[0,1,11,12,3,4,5,6,7,8,9,10]]
        else:
            states = states[:,[0, 1, 11, 12]]
    next_states = np.roll(states,-1,axis=0)[:-1,:]
    states = states[:-1,:]
    
            
    if with_med_filter:
        h = [40, 40, 100, 100]
        for i in range(states.shape[1]):
            if i == 4:
                continue
            try:
                states[:,i] = medfilter(states[:,i], h[i])
            except:
                states[:,i] = medfilter(states[:,i], 40)
            next_states[:-1,i]=states[1:,i]
            w=int(h[i]/2)
            next_states[-1,i]=np.mean(next_states[next_states.shape[0]-1-w:,i])
    else:
        states=states   
        
    if with_start_state:
        start_state=states[0,:]
        grasp_states=np.tile(start_state, (actions.shape[0], 1))
        actions=np.concatenate((actions,grasp_states),axis=1)
    else:
        actions=actions
    
    states_ls.append(states)
    test_actions_ls.append(actions[:states.shape[0]])
    next_states_ls.append(next_states)
    sas=np.concatenate([states,actions[:states.shape[0]],next_states],axis=1)
    checks=f_valid(states)
    sasc=np.concatenate([sas,checks],axis=1)
    #sas_ls.append(sas)
    test_sas_ls.append(sas[checks])
    test_sasc_ls.append(sasc)

test_state_dim=states_ls[0].shape[1]
test_action_dim=actions_ls[0].shape[1]
#test_ds_path=obj_dir+'/test/avi_test_separate_'+data_type+data_mode
test_ds_path=obj_dir+'/test/test_separate_'+data_type+data_mode+'_'+suffix+'f'
with open(test_ds_path,'wb') as f:
    #pickle.dump([sas_ls,sas_ls,test_state_dim,test_action_dim,[get_test_ground_truth(sas,test_state_dim,test_action_dim) for sas in sas_ls],actions_ls],f)
    test_traj_gt_ls=[get_test_ground_truth(sas,test_state_dim,test_action_dim) for sas in test_sas_ls]
    pickle.dump([test_sas_ls,test_sasc_ls,test_state_dim,test_action_dim,test_traj_gt_ls,test_actions_ls],f)
    
def make_train_test(ls,mix_idx_ls):
    return [i for j, i in enumerate(ls) if j not in mix_idx_ls],[ls[i] for i in mix_idx_ls]
if mix:
    all_ds_ls=sas_ls+test_sas_ls
    all_ds_all_ls=sasc_ls+test_sasc_ls
    all_traj_gt_ls=train_traj_gt_ls+test_traj_gt_ls
    real_all_actions_ls=actions_ls+test_actions_ls
    train_ds_ls,test_ds_ls=make_train_test(all_ds_ls,mix_idx_ls)
    train_ds_all_ls,test_ds_all_ls=make_train_test(all_ds_all_ls,mix_idx_ls)
    train_traj_gt_ls,test_traj_gt_ls=make_train_test(all_traj_gt_ls,mix_idx_ls)
    real_train_actions_ls,real_test_actions_ls=make_train_test(real_all_actions_ls,mix_idx_ls)
    
    train_ds_path=obj_dir+'/train_separate_'+data_type+data_mode+'_'+suffix+'f'
    print("total valid number of episodes for training:",len(train_ds_ls))
    with open(train_ds_path,'wb') as f:
        pickle.dump([train_ds_ls,train_ds_all_ls,train_state_dim,train_action_dim,train_traj_gt_ls,real_train_actions_ls],f)
    test_ds_path=obj_dir+'/test/test_separate_'+data_type+data_mode+'_'+suffix+'f'
    print("total valid number of episodes for testing:",len(test_ds_ls))
    with open(test_ds_path,'wb') as f:
        pickle.dump([test_ds_ls,test_ds_all_ls,test_state_dim,test_action_dim,test_traj_gt_ls,real_test_actions_ls],f)