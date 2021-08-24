import scipy.io
import numpy as np 
import pdb
import torch
from common.data_normalization import *
from common.pt_build_model import *
import matplotlib.pyplot as plt
import glob,pickle,os,copy,sys,random
import sys
from .hyperparameters import *
#argv num_test env_name lr nodes ho_rate model_ep data_file_suffix model_name_suffix(_v1, ...) seed xy_plot
def get_x_and_y(gt_states_for_plot):
    sai=gt_states_for_plot[:,1:2]+(gt_states_for_plot[:,3:4]*gt_states_for_plot[:,:1]+gt_states_for_plot[:,2:3]*gt_states_for_plot[:,1:2])
    cosai=-gt_states_for_plot[:,:1]-(gt_states_for_plot[:,2:3]*gt_states_for_plot[:,:1]-gt_states_for_plot[:,3:4]*gt_states_for_plot[:,1:2])
    return np.concatenate((sai,cosai),axis=1)

def get_y(gt_states_for_plot):
    cosai=-gt_states_for_plot[:,:1]-(gt_states_for_plot[:,2:3]*gt_states_for_plot[:,:1]-gt_states_for_plot[:,3:4]*gt_states_for_plot[:,1:2])
    return cosai

def main(num_test,env_name,lr,nodes,ho_rate, model_ep,data_file_suffix='train',model_name_suffix='',seed=0,xy_plot=0):
    dropout_rate = .0
    epoch_save_interval=5
    retrain=False
    batch_size = 200
    wd = .001
    nn_type = '1'
    both_nn_type='1'
    every_plot=1
    cuda = torch.cuda.is_available()
    dtype = torch.float

    num_test=int(num_test)
    lr=float(lr)
    nodes=int(nodes)
    ho_rate=float(ho_rate)
    model_ep=int(model_ep)
    seed=int(seed)
    xy_plot=int(xy_plot)

    train_ds_path='./mjo_eps_data/'+env_name+'_'+data_file_suffix+'_episode_data.pkl'
    with open(train_ds_path, 'rb') as pickle_file:
        processed_eps_ls,rwd_ls,done_ls,goal_ls = pickle.load(pickle_file)
    if model_name_suffix=='_all_success_pos':
        for i in range(len(processed_eps_ls)):
            processed_eps_ls[i]=np.concatenate((get_x_and_y(processed_eps_ls[i][:,:4]),processed_eps_ls[i][:,4:7],get_x_and_y(processed_eps_ls[i][:,7:11]),processed_eps_ls[i][:,11:]),axis=1)
    elif model_name_suffix=='_all_success_pos_nox':
        for i in range(len(processed_eps_ls)):
            processed_eps_ls[i]=np.concatenate((get_y(processed_eps_ls[i][:,:4]),processed_eps_ls[i][:,4:7],get_y(processed_eps_ls[i][:,7:11]),processed_eps_ls[i][:,11:]),axis=1)
    elif model_name_suffix=='_all_success_pos_novel':
        for i in range(len(processed_eps_ls)):
            processed_eps_ls[i]=np.concatenate((get_x_and_y(processed_eps_ls[i][:,:4]),processed_eps_ls[i][:,6:7],get_x_and_y(processed_eps_ls[i][:,7:11])),axis=1)
    elif model_name_suffix=='_all_success_pos_novel_nox':
        for i in range(len(processed_eps_ls)):
            processed_eps_ls[i]=np.concatenate((get_y(processed_eps_ls[i][:,:4]),processed_eps_ls[i][:,6:7],get_y(processed_eps_ls[i][:,7:11])),axis=1)
    
    if env_name=='Reacher-v2':
        if use_partial_state:
            state_dim=8
            x_loc=state_dim-2
            y_loc=state_dim-1
        else:
            state_dim=11
            x_loc=0
            y_loc=1
        action_dim=2
    elif env_name=='Acrobot-v1':
        if model_name_suffix=='_all_success_pos':
            state_dim=4
            x_loc=0
            y_loc=1
        elif model_name_suffix=='_all_success_pos_nox':
            state_dim=3
            y_loc=0
        elif model_name_suffix=='_all_success_pos_novel':
            state_dim=2
            x_loc=0
            y_loc=1
        elif model_name_suffix=='_all_success_pos_novel_nox':
            state_dim=1
            y_loc=0
        else:
            state_dim=6
            x_loc=0
            y_loc=1
        action_dim=1
    else:
        raise Exception('Not implemented!')

    test_idx=np.arange(num_test)

    task_ofs = state_dim + action_dim
    if env_name=='Reacher-v2':
        test_ds_ls = [torch.tensor(data, dtype=dtype) for data in processed_eps_ls[:num_test]]
    elif env_name=='Acrobot-v1':
        test_ds_ls=[] 
        for i in [158,164,190,209,336,345,348,383,406,439]:
            test_ds_ls.append(torch.tensor(processed_eps_ls[i], dtype=dtype))
    real_test_actions=[eps[:,state_dim:task_ofs].clone().detach() for eps in test_ds_ls]
    test_eps_len=[eps.shape[0] for eps in test_ds_ls]
    print("Rolling out " + str(len(test_ds_ls)) + " test trajectories.")

    cuda = False
    dtype = torch.float

    save_path = './trans_model_data'+model_name_suffix+'/'
    error_path = save_path+env_name+'_error/'
    pred_path = save_path+env_name+'_pred/'
    if not os.path.exists(pred_path):
        os.makedirs(pred_path)
    
    model_save_path = save_path + env_name + '_model_lr' + str(lr)+ '_nodes' + str(nodes) + '_seed'+str(seed)
    if ho_rate!=0:
        model_save_path+='_ho'+str(ho_rate)
    pred_fig_path = pred_path + 'traj_lr' + str(lr)+ '_nodes' + str(nodes) + '_seed'+str(seed)
    if model_ep!=0:
        model_save_path += '_epochs_'+str(model_ep)
        pred_fig_path += '_epochs_'+str(model_ep)

    with open(model_save_path, 'rb') as pickle_file:
        model = torch.load(pickle_file, map_location='cpu')
    if cuda: 
        model = model.to('cuda')
        model.norm = tuple(n.cuda() for n in model.norm)

    torch.manual_seed(seed)
    model.eval()
    for i in range(len(test_ds_ls)):
        prev_states = test_ds_ls[i][:,:state_dim]
        real_actions = real_test_actions[i]
        gt_states_for_plot = torch.cat((prev_states[:1,:],test_ds_ls[i][:,task_ofs:]),dim=0)
        if cuda:
            prev_states = prev_states.cuda()
            gt_states_for_plot = gt_states_for_plot.cuda()
            model = model.to('cuda')
            model.norm = tuple([n.cuda() for n in model.norm])
        with torch.no_grad():
            pred_states = model.run_traj(prev_states,real_actions)

        fig = plt.figure()
        if env_name=='Reacher-v2':
            if not use_partial_state:
                gt_states_for_plot=gt_states_for_plot[:,8:10]+gt_states_for_plot[:,4:6]
                pred_states=pred_states[:,8:10]+pred_states[:,4:6]
        elif env_name=='Acrobot-v1':
            if model_name_suffix not in ['_all_success_pos','_all_success_pos_nox','_all_success_pos_novel','_all_success_pos_novel_nox']:
                gt_states_for_plot=get_x_and_y(gt_states_for_plot)
                pred_states=get_x_and_y(pred_states)
        else:
            raise Exception('Not implemented!')
        if env_name=='Acrobot-v1' and xy_plot==0:
            plt.scatter(0, gt_states_for_plot[0, y_loc], s=150, c='k', marker="*",label='start')
            plt.plot(gt_states_for_plot[:, y_loc], color='blue', label='Ground Truth', marker='.', markersize=2, linewidth=1)
            plt.plot(pred_states[:, y_loc], color='red', label='NN Prediction')
            fig_loc=pred_fig_path +'_traj_'+str(i+1)+'_ypos.png'
        else:
            plt.scatter(gt_states_for_plot[0, x_loc], gt_states_for_plot[0, y_loc], s=150, c='k', marker="*",label='start')
            plt.plot(gt_states_for_plot[:, x_loc], gt_states_for_plot[:, y_loc], color='blue', label='Ground Truth', marker='.', markersize=2, linewidth=1)
            plt.plot(pred_states[:, x_loc], pred_states[:, y_loc], color='red', label='NN Prediction')
            plt.axis('scaled')
            if ho_rate!=0:
                fig_loc=pred_fig_path +'_traj_'+str(i+1)+'_ho'+str(ho_rate)+'_pos.png'
            else:
                fig_loc=pred_fig_path +'_traj_'+str(i+1)+'_pos.png'
        plt.title('Trajectory '+str(i+1)+ ' Pos Space')
        plt.legend()
        fig.set_size_inches(10, 10)
        fig.savefig(fig_loc)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        main(*sys.argv[1:])
    else:
        raise Exception('Missing environment!')
