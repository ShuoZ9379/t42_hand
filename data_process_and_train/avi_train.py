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
from sys import argv
from process_params import *
#argv lr heldout nntype seed dropout nodes epochs epoch_save_interval nm ne suffix(avi_, avi_v1_, ...)
#python3 avi_train.py 0.0001 0.1 2 0 0.1 200 50 1 nm ne avi_

seed=0
nn_type = '2'
held_out = .0
lr = .001
dropout_rate = .0
nodes=200
epochs=100
epoch_save_interval=1
dm='nm'
train_mode='ne'
suffix=''
if len(argv)>3:
    lr=float(argv[1])
    held_out=float(argv[2])
    nn_type=argv[3]
    seed=int(argv[4])
    dropout_rate=float(argv[5])
    nodes=int(argv[6])
    epochs=int(argv[7])
    epoch_save_interval=int(argv[8])
    if len(argv)>9:
        dm=argv[9]
        train_mode=argv[10]
        suffix=argv[11]
data_mode=data_mode[:-2]+dm

retrain=False
batch_size = 200
wd = .001
both_nn_type='1'
every_plot=1


if train_separate:
    #train_ds_path=base_path+color+'_data/'+obj+'/avi_train_separate_'+data_type+data_mode+'_f'
    #test_ds_path=base_path+color+'_data/'+obj+'/test/avi_test_separate_'+data_type+data_mode+'_f'
    train_ds_path=base_path+color+'_data/'+obj+'/train_separate_'+data_type+data_mode+'_'+suffix+'f'
    test_ds_path=base_path+color+'_data/'+obj+'/test/test_separate_'+data_type+data_mode+'_'+suffix+'f'
#Assume episodes are always separated
else:
    raise
with open(train_ds_path, 'rb') as pickle_file:
     train_ds_ls,train_ds_all_ls,state_dim,action_dim,train_traj_gt_ls,real_train_actions_ls = pickle.load(pickle_file)
with open(test_ds_path, 'rb') as pickle_file:
     test_ds_ls,test_ds_all_ls,state_dim,action_dim,test_traj_gt_ls,real_test_actions_ls = pickle.load(pickle_file)
task_ofs = state_dim + action_dim
out = [torch.tensor(ep, dtype=dtype) for ep in train_ds_ls]
test_out = [torch.tensor(ep, dtype=dtype) for ep in test_ds_ls]
eps_len=[ep.shape[0] for ep in train_ds_ls]
test_eps_len=[ep.shape[0] for ep in test_ds_ls]
val_size = int(sum(eps_len)*held_out)
print("Training Data Size: " + str(sum(eps_len)-val_size) + " Validation Data Size:" + str(val_size) + " Test Data Size:" + str(sum(test_eps_len)) )

cuda = torch.cuda.is_available()
cuda = False
dtype = torch.float
np.random.seed(seed)
np.random.shuffle(out)
#save_path = 'save_model/'
#error_fig_path = 'avi_error_fig/'
save_path = dm+'_'+train_mode+'_'+suffix+'save_model_f/'
error_fig_path = dm+'_'+train_mode+'_'+suffix+'error_fig_f/'
if not os.path.exists(save_path):
    os.makedirs(save_path)
if not os.path.exists(error_fig_path):
    os.makedirs(error_fig_path)
if not os.path.exists(save_path+'normalization/'):
    os.makedirs(save_path+'normalization/')
if not os.path.exists(save_path+'error/'):
    os.makedirs(save_path+'error/')
model_save_path = save_path + 'model_lr' + str(lr)+ '_' +'val' + str(held_out)+ '_' + 'seed' + str(seed) + '_nn_' + nn_type + '_dp_' + str(dropout_rate) +'_nodes_'+str(nodes)
error_save_path = save_path + 'error/err_lr' + str(lr)+ '_' +'val' + str(held_out)+ '_' + 'seed' + str(seed) + '_nn_' + nn_type + '_dp_' + str(dropout_rate)+'_nodes_'+str(nodes)
trainer = TrajModelTrainer(out, test_out, val_size=val_size, model_save_path=model_save_path, error_save_path=error_save_path, save_path=save_path, state_dim=state_dim, action_dim=action_dim) 
norm=trainer.norm

if retrain:
    with open(model_save_path, 'rb') as pickle_file:
        if cuda: traj_model=torch.load(pickle_file)
        else: traj_model=torch.load(pickle_file, map_location='cpu')
    with open(error_save_path, 'rb') as pickle_file:
        train_loss_ls, val_loss_ls, test_loss_ls = pickle.load(pickle_file)
    #torch.manual_seed(seed+len(train_loss_ls))
else:
    torch.manual_seed(seed)
    traj_model = TrajNet(norm, nn_type=nn_type, nodes=nodes, state_dim=state_dim, action_dim=action_dim, dropout_rate=dropout_rate, both_nn_type='1')

opt_traj = torch.optim.Adam(traj_model.parameters(), lr=lr, weight_decay=wd)
if cuda: 
    traj_model = traj_model.to('cuda')
    traj_model.norm = tuple(n.cuda() for n in traj_model.norm)
train_loss_ls,val_loss_ls,test_loss_ls = trainer.train(traj_model, opt_traj, every_plot, epochs=epochs, batch_size=batch_size, clip_grad_norm=True, loss_fn = torch.nn.MSELoss(),retrain=retrain,epoch_save_interval=epoch_save_interval)

#fig = plt.figure()
#plt.plot(train_loss_ls, color='k', label='Train Loss')
#if val_size:
#    plt.plot(val_loss_ls, color='blue', label='Validation Loss')
#plt.plot(test_loss_ls, color='red', label='Test Loss')
#plt.xlabel("Training Epochs")
#plt.ylabel("Loss")
#plt.legend()
#fig.set_size_inches(10,10)
#fig.savefig(error_fig_path+'All_Loss_lr' + str(lr)+ '_' +'val' + str(held_out)+ '_' + 'seed' + str(seed) + '_nn_' + nn_type + '_dp_' + str(dropout_rate)+'_nodes_'+str(nodes)+'.png')

fig = plt.figure()
plt.plot(train_loss_ls, color='k', label='Train Loss')
if val_size:
    plt.plot(val_loss_ls, color='blue', label='Validation Loss')
plt.xlabel("Training Epochs")
plt.ylabel("Loss")
plt.legend()
fig.set_size_inches(10,10)
fig.savefig(error_fig_path+'Loss_lr' + str(lr)+ '_' +'val' + str(held_out)+ '_' + 'seed' + str(seed) + '_nn_' + nn_type + '_dp_' + str(dropout_rate)+'_nodes_'+str(nodes)+'.png')
'''
fig = plt.figure()
plt.plot(train_loss_ls, color='k', label='Train Loss')
plt.xlabel("Training Epochs")
plt.ylabel("Loss")
plt.legend()
fig.set_size_inches(10,10)
fig.savefig(error_fig_path+'Train_Loss_lr' + str(lr)+ '_' +'val' + str(held_out)+ '_' + 'seed' + str(seed) + '_nn_' + nn_type + '_dp_' + str(dropout_rate)+'_nodes_'+str(nodes)+'.png')
'''