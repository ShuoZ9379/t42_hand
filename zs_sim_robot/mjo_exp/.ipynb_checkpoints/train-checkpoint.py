import scipy.io
import numpy as np 
import pdb
import torch
import torch.utils.data
from common.data_normalization import *
from common.TrajNet import *
from common.utils_clean_traj import *
import matplotlib.pyplot as plt
import glob,pickle,os,copy,sys,random
import sys
from .hyperparameters import *
from .predict import get_x_and_y,get_y
#argv env_name lr nodes epochs model_name_suffix(_v1, ...) data_file_suffix seed


def main(env_name,lr,nodes,epochs=50,data_file_suffix='train',model_name_suffix='',seed=0):

    held_out = .1
    dropout_rate = .0
    epoch_save_interval=1
    retrain=False
    batch_size = 200
    wd = .001
    nn_type = '1'
    both_nn_type='1'
    every_plot=1
    cuda = torch.cuda.is_available()
    dtype = torch.float

    lr=float(lr)
    nodes=int(nodes)
    epochs=int(epochs)
    seed=int(seed)

    train_ds_path='./mjo_eps_data/'+env_name+'_'+data_file_suffix+'_episode_data.pkl'
    with open(train_ds_path, 'rb') as pickle_file:
        processed_eps_ls,rwd_ls,done_ls,goal_ls = pickle.load(pickle_file)
    if env_name=='Reacher-v2':
        if use_partial_state:
            state_dim=8
        else:
            state_dim=11
        action_dim=2
    elif env_name=='Acrobot-v1':
        if model_name_suffix=='_all_success_pos':
            state_dim=4
        elif model_name_suffix=='_all_success_pos_nox':
            state_dim=3
        elif model_name_suffix=='_all_success_pos_novel':
            state_dim=2
        elif model_name_suffix=='_all_success_pos_novel_nox':
            state_dim=1
        else:
            state_dim=6
        action_dim=1
    else:
        raise Exception('Not implemented!')

    task_ofs = state_dim + action_dim
    out = [torch.tensor(ep, dtype=dtype) for ep in processed_eps_ls]
    if model_name_suffix=='_all_success_pos':
        for i in range(len(out)):
            out[i]=torch.tensor(np.concatenate((get_x_and_y(out[i][:,:4]),out[i][:,4:7],get_x_and_y(out[i][:,7:11]),out[i][:,11:]),axis=1), dtype=dtype)
    elif model_name_suffix=='_all_success_pos_nox':
        for i in range(len(out)):
            out[i]=torch.tensor(np.concatenate((get_y(out[i][:,:4]),out[i][:,4:7],get_y(out[i][:,7:11]),out[i][:,11:]),axis=1), dtype=dtype)
    elif model_name_suffix=='_all_success_pos_novel':
        for i in range(len(out)):
            out[i]=torch.tensor(np.concatenate((get_x_and_y(out[i][:,:4]),out[i][:,6:7],get_x_and_y(out[i][:,7:11])),axis=1), dtype=dtype)
    elif model_name_suffix=='_all_success_pos_novel_nox':
        for i in range(len(out)):
            out[i]=torch.tensor(np.concatenate((get_y(out[i][:,:4]),out[i][:,6:7],get_y(out[i][:,7:11])),axis=1), dtype=dtype)     
    eps_len=[ep.shape[0] for ep in processed_eps_ls]
    val_size = int(sum(eps_len)*held_out)
    print("Training Data Size: " + str(sum(eps_len)-val_size) + " Validation Data Size:" + str(val_size))

    np.random.seed(seed)
    np.random.shuffle(out)
    save_path = './trans_model_data'+model_name_suffix+'/'
    error_path = save_path+env_name+'_error/'
    norm_path = save_path+env_name+'_normalization/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists(norm_path):
        os.makedirs(norm_path)
    if not os.path.exists(error_path):
        os.makedirs(error_path)
    model_save_path = save_path + env_name + '_model_lr' + str(lr)+ '_nodes' + str(nodes) + '_seed'+str(seed)
    error_save_path = error_path + 'err_lr' + str(lr)+ '_nodes' + str(nodes) + '_seed'+str(seed)
    trainer = TrajModelTrainer(env_name, out, val_size=val_size, model_save_path=model_save_path, error_save_path=error_save_path, norm_path=norm_path, state_dim=state_dim, action_dim=action_dim) 
    norm=trainer.norm

    if retrain:
        raise Exception('Not implemented!')
        '''
        with open(model_save_path, 'rb') as pickle_file:
            if cuda: 
                traj_model=torch.load(pickle_file)
            else: 
                traj_model=torch.load(pickle_file, map_location='cpu')
        with open(error_save_path, 'rb') as pickle_file:
            train_loss_ls, val_loss_ls, test_loss_ls = pickle.load(pickle_file)
        '''
        #torch.manual_seed(seed+len(train_loss_ls))
    else:
        torch.manual_seed(seed)
        traj_model = TrajNet(env_name, norm, nn_type=nn_type, nodes=nodes, state_dim=state_dim, action_dim=action_dim, dropout_rate=dropout_rate, both_nn_type='1')

    opt_traj = torch.optim.Adam(traj_model.parameters(), lr=lr, weight_decay=wd)
    if cuda: 
        traj_model = traj_model.to('cuda')
        traj_model.norm = tuple(n.cuda() for n in traj_model.norm)
    train_loss_ls,val_loss_ls = trainer.train(traj_model, opt_traj, every_plot, epochs=epochs, batch_size=batch_size, clip_grad_norm=True, loss_fn = torch.nn.MSELoss(),retrain=retrain,epoch_save_interval=epoch_save_interval)


    fig = plt.figure()
    plt.plot(train_loss_ls, color='k', label='Train Loss')
    if val_size:
        plt.plot(val_loss_ls, color='blue', label='Validation Loss')
    plt.xlabel("Training Epochs")
    plt.ylabel("Loss")
    plt.legend()
    fig.set_size_inches(10,10)
    fig.savefig(error_path+'All_Loss_lr' + str(lr)+ '_nodes' + str(nodes) + '_seed' + str(seed) +'.png')


if __name__ == '__main__':
    if len(sys.argv) > 1:
        main(*sys.argv[1:])
    else:
        raise Exception('Missing environment!')
