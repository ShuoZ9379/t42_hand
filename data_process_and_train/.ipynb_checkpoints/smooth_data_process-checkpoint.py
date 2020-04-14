#never tear obj marker and paste again
#try centralize the center of obj marker on obj
#valid rmat and tvec should not change much during one idx of collection
from process_params import *
from sys import argv
#argv idx1 idx2 idx3 idx4 idx5 ..., filter_size, recali, include init state,  not remove d), interval, not remove e), length_e, not remove b), not remove c), ne, suffxi(v1_,avi_,...)
mix=False
mix_idx_ls=[]
filter_size=10
not_recali=True
include_init=True
not_remove_d=True
interval=10
not_remove_e=True
length_e=100
not_remove_b=True
not_remove_c=True
train_mode='ne'
suffix=''
if len(argv)>3:
    if argv[-2][-1]!='e':
        filter_size=int(argv[-9])
        not_recali=bool(argv[-8])
        include_init=bool(argv[-7])
        not_remove_d=bool(argv[-6])
        interval=int(argv[-5])
        not_remove_e=bool(argv[-4])
        length_e=int(argv[-3])
        not_remove_b=bool(argv[-2])
        not_remove_c=bool(argv[-1])
        mix_idx_ls=[int(argv[i+1]) for i in range(len(argv)-10)]
    else:
        filter_size=int(argv[-11])
        not_recali=bool(argv[-10])
        include_init=bool(argv[-9])
        not_remove_d=bool(argv[-8])
        interval=int(argv[-7])
        not_remove_e=bool(argv[-6])
        length_e=int(argv[-5])
        not_remove_b=bool(argv[-4])
        not_remove_c=bool(argv[-3])
        train_mode=argv[-2]
        suffix=argv[-2]
        mix_idx_ls=[int(argv[i+1]) for i in range(len(argv)-12)]
    if mix_idx_ls==[]:
        mix=False
    else:
        mix=True
dm='wm'
data_mode=data_mode[:-2]+dm
import glob,pickle,os,copy
import numpy as np

def cali(memory,cali_path):
    tmp=np.array(memory)
    tmp=tmp[tmp[:,-1]==False]
    tmp=np.array(list(tmp[:,2]))
    for j in [-1,-2,-3]:
        tmp=tmp[tmp[:,j]==False]
    print("index "+cali_path[-6]+" valid number of data for calibration:",tmp.shape[0])
    cali_info=np.mean(tmp[:,:12],axis=0)
    with open(cali_path,'wb') as f:
        pickle.dump(cali_info,f)
    return cali_info

def convert_to_nparr(memory,color):
    tmp=np.array(memory)
    tmp_1=np.array(list(tmp[:,2]))
    actions=np.array(list(tmp[:,3]))
    tmp_2=tmp[:,-1]
    tmp_2=tmp_2.astype(float)
    states=np.concatenate((tmp_1,tmp_2.reshape(-1,1)),axis=1)
    if color=='blue':
        new_actions=np.copy(actions)
        actions[:,0]=new_actions[:,1]
        actions[:,1]=new_actions[:,0]
    return states,actions

def get_transformed_states(states,rmat,tvec,color):
    transformed_states=np.zeros((states.shape[0],17))
    corner_pos=np.zeros((states.shape[0],2))
    transformed_states[:,1]=-rmat.T.dot((states[:,12:15]-tvec).T).T[:,0]
    transformed_states[:,0]=rmat.T.dot((states[:,12:15]-tvec).T).T[:,1]
    corner_pos[:,1]=-rmat.T.dot((states[:,15:18]-tvec).T).T[:,0]
    corner_pos[:,0]=rmat.T.dot((states[:,15:18]-tvec).T).T[:,1]
    transformed_states[:,2]=np.arctan2(corner_pos[:,1]-transformed_states[:,1],corner_pos[:,0]-transformed_states[:,0])
    for i in range(4):
        transformed_states[:,4+2*i]=-rmat.T.dot((states[:,18+3*i:21+3*i]-tvec).T).T[:,0]
        transformed_states[:,3+2*i]=rmat.T.dot((states[:,18+3*i:21+3*i]-tvec).T).T[:,1]
    for i in range(11,17):
        transformed_states[:,i]=states[:,i+19]
    ##########Need to be corrected later##############
    if color== 'blue':
        transformed_states[:,11:13]=transformed_states[:,11:13]
    ##########Need to be corrected later##############
    return transformed_states

def process_train_states_and_actions(states,actions,with_finger,with_angle,with_med_filter,with_start_state):
    checks=states[:,-4:]
    
    eps_ls=get_eps_ls(states,checks)
    
    states[:,:2] *= 1000.
    if with_angle:
        if with_finger:
            states = states[:,[0,1,11,12,2,3,4,5,6,7,8,9,10]]
            states[:,5:] *= 1000.
        else:
            states = states[:,[0,1,11,12,2]]        
    else:
        if with_finger:
            states = states[:,[0,1,11,12,3,4,5,6,7,8,9,10]]
            states[:,4:] *= 1000.
        else:
            states = states[:,[0, 1, 11, 12]]
    if with_med_filter:
        states=medfilter(eps_ls)
    else:
        states=states   
        
    if with_start_state:
        new_eps_ls=get_eps_ls(states,checks)
        checks_ls=np.split(checks,np.argwhere(checks[:,-1]==1).reshape(-1)+1,axis=0)
        checks_ls.pop(-1)
        all_grasp_states=np.zeros((actions.shape[0],states.shape[1]))
        index=0
        for i in range(len(new_eps_ls)):
            eps=new_eps_ls[i]
            grasp_states=np.tile(eps[0,:], (eps.shape[0], 1))
            all_grasp_states[index:index+eps.shape[0],:]=grasp_states
            if checks_ls[i][0,:].any():
                checks[index:index+eps.shape[0],:]=np.ones((eps.shape[0],checks.shape[1]))
                checks[index:index+eps.shape[0]-1,-1]=np.zeros(eps.shape[0]-1)
            index=index+eps.shape[0]
        actions=np.concatenate((actions,all_grasp_states),axis=1)
    else:
        actions=actions
    
    return states,actions,checks

def get_eps_ls(states,checks):
    ls=np.split(states,np.argwhere(checks[:,-1]==1).reshape(-1)+1,axis=0)
    ls.pop(-1)
    return ls

def medfilter(eps_ls,checks_ls,filter_size,include_init):
    state_dim=eps_ls[0].shape[1]
    new_states=np.empty((0,state_dim))
    if include_init:
        start_idx=0
    else:
        start_idx=1
    if state_dim==5 or state_dim==13:
        W=[filter_size for i in range(4)]+[None]+[filter_size for i in range(8)]
        for k in range(len(eps_ls)):
            eps=eps_ls[k]
            checks=checks_ls[k]
            valid_idx_ls=((checks[:,-1]==0).astype(int)+(checks[:,-2]==0).astype(int)+(checks[:,-3]==0).astype(int)+(checks[:,-4]==0).astype(int))==4
            for j in range(state_dim) :
                x=eps[:,j]
                cks=checks[:,j]
                x_new=np.copy(x)
                if j!=4:
                    w = int(W[j]/2)
                    for i in range(start_idx, x.shape[0]):
                        if i < w:
                            if valid_idx_ls[:i+w].any():
                                x_new[i] = np.mean(x[:i+w][valid_idx_ls[:i+w]])
                        elif i > x.shape[0]-w:
                            if valid_idx_ls[i-w:].any():
                                x_new[i] = np.mean(x[i-w:][valid_idx_ls[i-w:]])
                        else:
                            if valid_idx_ls[i-w:i+w].any():
                                x_new[i] = np.mean(x[i-w:i+w][valid_idx_ls[i-w:i+w]])
                    eps[:,j]=x_new
            new_states=np.concatenate((new_states,eps),0)
    else:
        W=[filter_size for i in range(4)]+[filter_size for i in range(8)]
        for k in range(len(eps_ls)):
            eps=eps_ls[k]
            checks=checks_ls[k]
            valid_idx_ls=((checks[:,-1]==0).astype(int)+(checks[:,-2]==0).astype(int)+(checks[:,-3]==0).astype(int)+(checks[:,-4]==0).astype(int))==4
            for j in range(state_dim) :
                x=eps[:,j]
                x_new=np.copy(x)
                w = int(W[j]/2)
                for i in range(start_idx, x.shape[0]):
                    if i < w:
                        if valid_idx_ls[:i+w].any():
                            x_new[i] = np.mean(x[:i+w][valid_idx_ls[:i+w]])
                    elif i > x.shape[0]-w:
                        if valid_idx_ls[i-w:].any():
                            x_new[i] = np.mean(x[i-w:][valid_idx_ls[i-w:]])
                    else:
                        if valid_idx_ls[i-w:i+w].any():
                            x_new[i] = np.mean(x[i-w:i+w][valid_idx_ls[i-w:i+w]])
                eps[:,j]=x_new
            new_states=np.concatenate((new_states,eps),0)
    return new_states

def get_final_dataset(states,actions,checks,valid_idx,real_len=None):
    next_states=np.roll(states,-1,axis=0)
    preprocess_sa=np.concatenate((states,actions),axis=1)
    preprocess_sas=np.concatenate((preprocess_sa,next_states),axis=1)
    #valid_idx=check_valid(checks)
    #valid_idx=f_check_valid(states,checks)
    final_sas=preprocess_sas[valid_idx]
    preprocess_sasc=np.concatenate((preprocess_sas,valid_idx.reshape(-1,1)),axis=1)
    if real_len==None:
        return final_sas,states.shape[1],actions.shape[1]
    else:
        #return final_sas,states.shape[1],actions.shape[1],preprocess_sas[:real_len,:]
        return final_sas,states.shape[1],actions.shape[1],preprocess_sasc[:real_len,:]
    
def check_valid(checks):
    valid_idx_ls=((checks[:,-1]==0).astype(int)+(checks[:,-2]==0).astype(int)+(checks[:,-3]==0).astype(int)+(checks[:,-4]==0).astype(int))==4
    prev_checks=np.roll(checks,-1,axis=0)
    valid_prev_idx_ls=((prev_checks[:,-1]==0).astype(int)+(prev_checks[:,-2]==0).astype(int)+(prev_checks[:,-3]==0).astype(int)+(prev_checks[:,-4]==0).astype(int))==4
    final_valid_idx_ls=((valid_idx_ls==1).astype(int)+(valid_prev_idx_ls==1).astype(int))==2
    return final_valid_idx_ls

def process_test_states_and_actions(states,actions,with_finger,with_angle,with_med_filter,with_start_state,filter_size,include_init):
    checks=states[:,-4:]
    checks_ls=get_eps_ls(checks,checks)
    
    states[:,:2] *= 1000.
    if with_angle:
        if with_finger:
            states = states[:,[0,1,11,12,2,3,4,5,6,7,8,9,10]]
            states[:,5:] *= 1000.
        else:
            states = states[:,[0,1,11,12,2]]        
    else:
        if with_finger:
            states = states[:,[0,1,11,12,3,4,5,6,7,8,9,10]]
            states[:,4:] *= 1000.
        else:
            states = states[:,[0, 1, 11, 12]]
    if with_med_filter:
        eps_ls=get_eps_ls(states,checks)
        states=medfilter(eps_ls,checks_ls,filter_size,include_init)
        #update checks in smooth (if len(valid_data)<=40, then checks first 3 columns =1)
    else:
        states=states   
        
    if with_start_state:
        new_eps_ls=get_eps_ls(states,checks)
        checks_ls=np.split(checks,np.argwhere(checks[:,-1]==1).reshape(-1)+1,axis=0)
        checks_ls.pop(-1)
        all_grasp_states=np.zeros((actions.shape[0],states.shape[1]))
        index=0
        for i in range(len(new_eps_ls)):
            eps=new_eps_ls[i]
            grasp_states=np.tile(eps[0,:], (eps.shape[0], 1))
            all_grasp_states[index:index+eps.shape[0],:]=grasp_states
            if checks_ls[i][0,:].any():
                checks[index:index+eps.shape[0],:]=np.ones((eps.shape[0],checks.shape[1]))
                checks[index:index+eps.shape[0]-1,-1]=np.zeros(eps.shape[0]-1)
            index=index+eps.shape[0]
        actions=np.concatenate((actions,all_grasp_states),axis=1)
    else:
        actions=actions
    checks_ls=np.split(checks,np.argwhere(checks[:,-1]==1).reshape(-1)+1,axis=0)
    checks_ls.pop(-1)
    actions_ls=np.split(actions,np.argwhere(checks[:,-1]==1).reshape(-1)+1,axis=0)
    actions_ls.pop(-1)
    return new_eps_ls,actions_ls,checks_ls

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

def f_check_valid(states,checks,not_remove_b,not_remove_c):
    ss=np.concatenate((states,np.roll(states,-1,axis=0)),axis=1)
    prev=ss[:,:2]
    prev_all=ss[:,:4]
    nxt=ss[:,states.shape[1]:states.shape[1]+2]
    nxt_all=ss[:,states.shape[1]:states.shape[1]+4]
    conti=np.linalg.norm(prev-nxt,axis=1)<=1.2
    non_equal=np.sum(prev_all-nxt_all==0,1)
    final_valid_idx_ls=check_valid(checks)==1
    if not not_remove_b:
        final_valid_idx_ls=final_valid_idx_ls+((conti==1).astype(int)==2)
    if not not_remove_c:
        final_valid_idx_ls=final_valid_idx_ls+((non_equal!=4).astype(int))
    return final_valid_idx_ls

def check_nonvalid_end(indices_arr,train_states,interval):
    to_check=train_states[indices_arr[-interval-1:],:2]
    diff=np.roll(to_check,-1,axis=0)-to_check
    diff=diff[:interval,:]
    valid=np.linalg.norm(diff,axis=1)<=1.2
    if valid.all():
        return False
    else:
        return True

def check_outlier(pos,checks):
    if checks[0,:].any():
        if checks[2,:].any():
            checks[1,0]=1.
        elif np.linalg.norm(pos[1,:]-pos[2,:])*1000>1.2:
            checks[1,0]=1.
        #### Assume there is no case of consecutive 2 poses which are closer than 1.2mm ####
            #if (not checks[2,:].any()) and (not checks[1,:].any():
            #    checks[1,0]=1.
    else:
        if np.linalg.norm(pos[1,:]-pos[0,:])*1000>1.2:
            checks[1,0]=1.
    return checks

def remove_outlier(transformed_states):
    states,checks=np.split(transformed_states,[-4],axis=1)
    pos=states[:,:2]
    for i in range(1,pos.shape[0]-1):
        checks[i-1:i+2,:]=check_outlier(pos[i-1:i+2,:],checks[i-1:i+2,:])
    return np.concatenate((states,checks),axis=1)

def check_bad_end(indices_arr,eps_pos,eps_checks,interval):
    to_check=eps_pos[indices_arr[-interval-1:],:]
    diff=np.roll(to_check,-1,axis=0)-to_check
    diff=diff[:interval,:]
    valid=np.linalg.norm(diff,axis=1)*1000<=1.2
    for i in range(interval):
        if not valid[i]:
            eps_checks[indices_arr[-interval+i:],0]=np.ones(interval-i)
            return eps_checks
    return eps_checks

def remove_end(transformed_states,interval):
    new_checks=np.empty((0,4))
    states,checks=np.split(transformed_states,[-4],axis=1)
    pos=states[:,:2]
    pos_ls=get_eps_ls(pos,checks)
    checks_ls=get_eps_ls(checks,checks)
    for i in range(len(pos_ls)):
        eps_pos,eps_checks=pos_ls[i],checks_ls[i]
        #valid_idx_ls=check_valid(eps_checks)
        valid_idx_ls=((eps_checks[:,-1]==0).astype(int)+(eps_checks[:,-2]==0).astype(int)+(eps_checks[:,-3]==0).astype(int)+(eps_checks[:,-4]==0).astype(int))==4
        indices_arr=np.where(valid_idx_ls==True)[0]
        if indices_arr.shape[0]!=0:
            eps_checks=check_bad_end(indices_arr,eps_pos,eps_checks,interval)
        new_checks=np.concatenate((new_checks,eps_checks),axis=0)
    return np.concatenate((states,new_checks),axis=1)

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
cali_dir=obj_dir+'/cali'
if not os.path.exists(cali_dir):
    os.makedirs(cali_dir)
test_dir=obj_dir+'/test'
if not os.path.exists(test_dir):
    os.makedirs(test_dir)

train_paths=[]
for idx in train_idx:
    train_paths.append(base_path+color+'_data/'+'zs_raw_train_'+obj+'_'+color+'_'+data_type+'_v'+idx+'.obj')
train_states=np.empty((0,17))
train_actions=np.empty((0,2))
for i in range(len(train_paths)):
    idx=train_paths[i][-5]
    with open(train_paths[i],'rb') as filehandler:
        #memory: (base_rmat,base_tvec,obj_pos,corner_pos,finger_poses,no_detect,no_paral,drop),done
        memory=pickle.load(filehandler,encoding='latin1')
    same_idx_test_path=base_path+color+'_data/'+'zs_raw_test_'+obj+'_'+color+'_'+data_type+'_v'+idx+'.obj'
    if os.path.exists(same_idx_test_path):
        with open(same_idx_test_path,'rb') as filehandler:
            test_memory=pickle.load(filehandler,encoding='latin1')
        memory_for_cali=memory+test_memory
    else:
        memory_for_cali=memory
    cali_path=cali_dir+'/'+data_mode+'_'+suffix+'_'+idx+'.cali' 
    if do_cali:
        cali_info=cali(memory_for_cali,cali_path)
    else:
        with open(cali_path,'rb') as f:
            cali_info=np.array(pickle.load(f))

    if not not_recali:
        rmat=cali_info[:9].reshape(3,3)
        tvec=cali_info[9:]
        states,actions=convert_to_nparr(memory,color)
        transformed_states=get_transformed_states(states,rmat,tvec,color)
        transformed_states=remove_outlier(transformed_states)
        if not not_remove_d:
            transformed_states=remove_end(transformed_states,interval)
        all_transformed_states=transformed_states
        if os.path.exists(same_idx_test_path):
            states,test_actions=convert_to_nparr(test_memory,color)
            test_transformed_states=get_transformed_states(states,rmat,tvec,color)
            test_transformed_states=remove_outlier(test_transformed_states)
            if not not_remove_d:
                test_transformed_states=remove_end(test_transformed_states,interval)
            all_transformed_states=np.concatenate((transformed_states,test_transformed_states),axis=0)
        states,checks=np.split(all_transformed_states,[-4],axis=1)    
        valid_idx_ls=((checks[:,-1]==0).astype(int)+(checks[:,-2]==0).astype(int)+(checks[:,-3]==0).astype(int)+(checks[:,-4]==0).astype(int))==4 
        tmp=np.array(memory_for_cali)
        tmp=np.array(list(tmp[:,2]))
        tmp=tmp[valid_idx_ls,:]
        print("index "+cali_path[-6]+" valid number of data for re-calibration:",tmp.shape[0])
        cali_info=np.mean(tmp[:,:12],axis=0)
        with open(cali_path,'wb') as f:
            pickle.dump(cali_info,f)
    
    rmat=cali_info[:9].reshape(3,3)
    tvec=cali_info[9:]
    states,actions=convert_to_nparr(memory,color)
    transformed_states=get_transformed_states(states,rmat,tvec,color)
    transformed_states=remove_outlier(transformed_states)
    if not not_remove_d:
        transformed_states=remove_end(transformed_states,interval)
    train_states=np.concatenate((train_states,transformed_states),axis=0)
    train_actions=np.concatenate((train_actions,actions),axis=0)
    
if not train_separate:
    train_states,train_actions,train_checks=process_train_states_and_actions(train_states,train_actions,with_finger,with_angle,with_med_filter,with_start_state)
    train_ds,train_state_dim,train_action_dim=get_final_dataset(train_states,train_actions,train_checks)

    print('total valid number of data for training:',train_ds.shape[0])
    train_ds_path=obj_dir+'/train_full_'+data_type+'_v'+train_idx+data_mode
    with open(train_ds_path,'wb') as f:
        pickle.dump([train_ds,train_state_dim,train_action_dim],f)
else:
    train_states_ls,train_actions_ls,train_checks_ls=process_test_states_and_actions(train_states,train_actions,with_finger,with_angle,with_med_filter,with_start_state,filter_size,include_init)
    train_ds_ls,train_ds_all_ls, train_traj_gt_ls,real_train_actions_ls=[],[],[],[]
    sm=0
    for j in range(len(train_states_ls)):
        train_states,train_actions,train_checks=train_states_ls[j],train_actions_ls[j],train_checks_ls[j]

        train_final_valid_idx_ls=list(f_check_valid(train_states,train_checks,not_remove_b,not_remove_c))
        if len(train_final_valid_idx_ls)!=0:
            while train_final_valid_idx_ls[-1]==False:
                train_final_valid_idx_ls.pop(-1)
                if len(train_final_valid_idx_ls)==0:
                    break
        if len(train_final_valid_idx_ls)==0:
            continue
        real_len=len(train_final_valid_idx_ls)
        train_valid_idx=f_check_valid(train_states,train_checks,not_remove_b,not_remove_c)
        train_valid_idx[real_len:]=np.zeros(train_states.shape[0]-real_len,dtype=bool)
        real_train_actions=train_actions[:real_len,:]
        train_ds,train_state_dim,train_action_dim,train_ds_all=get_final_dataset(train_states,train_actions,train_checks,train_valid_idx,real_len)
        train_traj_gt=get_test_ground_truth(train_ds,train_state_dim,train_action_dim)
        
        #print(str(j)+':'+str(train_ds.shape[0]))
        #Deal with too short episode
        if not not_remove_e:
            if train_ds.shape[0]>length_e:
                sm+=train_ds.shape[0]
                train_ds_ls.append(train_ds)
                train_ds_all_ls.append(train_ds_all)
                train_traj_gt_ls.append(train_traj_gt)
                real_train_actions_ls.append(real_train_actions)
        else:
            sm+=train_ds.shape[0]
            train_ds_ls.append(train_ds)
            train_ds_all_ls.append(train_ds_all)
            train_traj_gt_ls.append(train_traj_gt)
            real_train_actions_ls.append(real_train_actions)      
    train_ds_path=obj_dir+'/train_separate_'+data_type+'_v'+train_idx+data_mode+'_'+suffix+'f'
    print("total valid number of episodes for training:",len(train_ds_ls))
    print(sm)
    with open(train_ds_path,'wb') as f:
        pickle.dump([train_ds_ls,train_ds_all_ls,train_state_dim,train_action_dim,train_traj_gt_ls,real_train_actions_ls],f)

test_paths=[]
for idx in test_idx:
    test_paths.append(base_path+color+'_data/'+'zs_raw_test_'+obj+'_'+color+'_'+data_type+'_v'+idx+'.obj')
test_states=np.empty((0,17))
test_actions=np.empty((0,2))
test_ds_ls,test_ds_all_ls,test_traj_gt_ls,real_test_actions_ls=[],[],[],[]
for i in range(len(test_paths)):
    idx=test_paths[i][-5]
    with open(test_paths[i],'rb') as filehandler:
        test_memory=pickle.load(filehandler,encoding='latin1')
    cali_path=cali_dir+'/'+suffix+idx+'.cali'
    with open(cali_path,'rb') as f:
        cali_info=np.array(pickle.load(f))
    rmat=cali_info[:9].reshape(3,3)
    tvec=cali_info[9:]
    states,test_actions=convert_to_nparr(test_memory,color)
    test_transformed_states=get_transformed_states(states,rmat,tvec,color)
    test_transformed_states=remove_outlier(test_transformed_states)
    if not not_remove_d:
        test_transformed_states=remove_end(test_transformed_states,interval)
    
    test_states_ls,test_actions_ls,test_checks_ls=process_test_states_and_actions(test_transformed_states,test_actions,with_finger,with_angle,with_med_filter,with_start_state,filter_size,include_init)
    num=0
    for j in range(len(test_states_ls)):
        test_states,test_actions,test_checks=test_states_ls[j],test_actions_ls[j],test_checks_ls[j]
        test_final_valid_idx_ls=list(f_check_valid(test_states,test_checks,not_remove_b,not_remove_c))
        if len(test_final_valid_idx_ls)!=0:
            while test_final_valid_idx_ls[-1]==False:
                test_final_valid_idx_ls.pop(-1)
                if len(test_final_valid_idx_ls)==0:
                    break
        if len(test_final_valid_idx_ls)==0:
            continue
        real_len=len(test_final_valid_idx_ls)
        test_valid_idx=f_check_valid(test_states,test_checks,not_remove_b,not_remove_c)
        test_valid_idx[real_len:]=np.zeros(test_states.shape[0]-real_len,dtype=bool)
        real_test_actions=test_actions[:real_len,:]
        test_ds,test_state_dim,test_action_dim,test_ds_all=get_final_dataset(test_states,test_actions,test_checks,test_valid_idx,real_len)
        test_traj_gt=get_test_ground_truth(test_ds,test_state_dim,test_action_dim)
        
        #print(str(j)+':'+str(test_ds.shape[0]))
        #Deal with too short episode
        if not not_remove_e:
            if test_ds.shape[0]>length_e:
                print("index %s the %sth episode: real length for prediction is %s, valid trajectory ground truth length is %s, valid data number is %s" % (i, j,real_len,len(test_traj_gt),test_ds.shape[0]))
                test_ds_ls.append(test_ds)
                test_ds_all_ls.append(test_ds_all)
                test_traj_gt_ls.append(test_traj_gt)
                real_test_actions_ls.append(real_test_actions)
                test_ds_path=test_dir+'/test_'+data_type+'_v'+idx+data_mode+'_'+str(num)+'_'+suffix+'f'
                with open(test_ds_path,'wb') as f:
                    pickle.dump([test_ds,test_ds_all,test_state_dim,test_action_dim,test_traj_gt,real_test_actions],f)
                num+=1
        else:
            print("index %s the %sth episode: real length for prediction is %s, valid trajectory ground truth length is %s, valid data number is %s" % (i, j,real_len,len(test_traj_gt),test_ds.shape[0]))
            test_ds_ls.append(test_ds)
            test_ds_all_ls.append(test_ds_all)
            test_traj_gt_ls.append(test_traj_gt)
            real_test_actions_ls.append(real_test_actions)
            test_ds_path=test_dir+'/test_'+data_type+'_v'+idx+data_mode+'_'+str(num)+'_'+suffix+'f'
            with open(test_ds_path,'wb') as f:
                pickle.dump([test_ds,test_ds_all,test_state_dim,test_action_dim,test_traj_gt,real_test_actions],f)
            num+=1

test_ds_path=test_dir+'/test_separate_'+data_type+'_v'+test_idx+data_mode+'_'+suffix+'f'
print("total valid number of episodes for testing:",len(test_ds_ls))
with open(test_ds_path,'wb') as f:
    pickle.dump([test_ds_ls,test_ds_all_ls,test_state_dim,test_action_dim,test_traj_gt_ls,real_test_actions_ls],f)

def make_train_test(ls,mix_idx_ls):
    return [i for j, i in enumerate(ls) if j not in mix_idx_ls],[ls[i] for i in mix_idx_ls]
if mix:
    all_ds_ls=train_ds_ls+test_ds_ls
    all_ds_all_ls=train_ds_all_ls+test_ds_all_ls
    all_traj_gt_ls=train_traj_gt_ls+test_traj_gt_ls
    real_all_actions_ls=real_train_actions_ls+real_test_actions_ls
    train_ds_ls,test_ds_ls=make_train_test(all_ds_ls,mix_idx_ls)
    train_ds_all_ls,test_ds_all_ls=make_train_test(all_ds_all_ls,mix_idx_ls)
    train_traj_gt_ls,test_traj_gt_ls=make_train_test(all_traj_gt_ls,mix_idx_ls)
    real_train_actions_ls,real_test_actions_ls=make_train_test(real_all_actions_ls,mix_idx_ls)
    
    train_ds_path=obj_dir+'/train_separate_'+data_type+'_v'+train_idx+data_mode+'_'+suffix+'f'
    print("total valid number of episodes for training:",len(train_ds_ls))
    with open(train_ds_path,'wb') as f:
        pickle.dump([train_ds_ls,train_ds_all_ls,train_state_dim,train_action_dim,train_traj_gt_ls,real_train_actions_ls],f)
    test_ds_path=test_dir+'/test_separate_'+data_type+'_v'+test_idx+data_mode+'_'+suffix+'f'
    print("total valid number of episodes for testing:",len(test_ds_ls))
    with open(test_ds_path,'wb') as f:
        pickle.dump([test_ds_ls,test_ds_all_ls,test_state_dim,test_action_dim,test_traj_gt_ls,real_test_actions_ls],f)