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
#argv ho_rate nodes epochs seed
#0.995: seed 0
#0.999: seed 2 worst


def main(ho_rate=0.995,nodes=512,epochs=20,seed=0):

    lr=0.00005
    held_out = .1
    dropout_rate = .1
    epoch_save_interval=10000000
    retrain=False
    batch_size = 64
    wd = .001
    nn_type = '1'
    both_nn_type='1'
    every_plot=1
    cuda = torch.cuda.is_available()
    dtype = torch.float

    nodes=int(nodes)
    epochs=int(epochs)
    ho_rate=float(ho_rate)
    seed=int(seed)

    if ho_rate in [0.5,0.6,0.7,0.8,0.9,0.95,0.99] or ho_rate==0:
        raise Exception("Already trained by Avishai!")

    train_ds_path='./sim_data_cont_v0_d4_m1_episodes.obj'
    with open(train_ds_path, 'rb') as pickle_file:
        processed_eps_ls = pickle.load(pickle_file,encoding='latin')
    state_dim=4
    action_dim=2

    task_ofs = state_dim + action_dim
    out = [torch.tensor(ep, dtype=dtype) for ep in processed_eps_ls]
    eps_len=[ep.shape[0] for ep in processed_eps_ls]
    all_eps_len=round(sum(eps_len)*(1-ho_rate))
    val_size = round(all_eps_len*held_out)
    print("Training Data Size: " + str(all_eps_len-val_size) + " Validation Data Size:" + str(val_size))

    np.random.seed(seed)
    np.random.shuffle(out)
    save_path = './trans_model_data/'
    env_name = 'gazebo_ah'
    error_path = save_path+env_name+'_error/'
    norm_path = save_path+env_name+'_normalization/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists(norm_path):
        os.makedirs(norm_path)
    if not os.path.exists(error_path):
        os.makedirs(error_path)
    model_save_path = save_path + env_name + '_model/sim_cont_trajT_bs512_model512_BS64_loadT'
    model_save_path+='_ho'+str(ho_rate)
    error_save_path = error_path + 'err_lr' + str(lr)+ '_nodes' + str(nodes) + '_seed'+str(seed)
    if ho_rate!=0:
        error_save_path+='_ho'+str(ho_rate)
    trainer = TrajModelTrainer(env_name, out, val_size=val_size, model_save_path=model_save_path, error_save_path=error_save_path, norm_path=norm_path, state_dim=state_dim, action_dim=action_dim, ho_rate=ho_rate) 
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
    if ho_rate==0:
        fig.savefig(error_path+'All_Loss_lr' + str(lr)+ '_nodes' + str(nodes) + '_seed' + str(seed) +'.png')
    else:
        fig.savefig(error_path+'All_Loss_lr' + str(lr)+ '_nodes' + str(nodes) + '_seed' + str(seed) +'_ho'+str(ho_rate)+'.png')

if __name__ == '__main__':
    if len(sys.argv) > 1:
        main(*sys.argv[1:])
    else:
        raise Exception('Missing environment!')
