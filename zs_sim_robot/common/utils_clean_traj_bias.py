import numpy as np 
import pdb
import torch
import torch.utils.data
from common.data_normalization import *
import random
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt


cuda = torch.cuda.is_available()
#cuda = False
dtype = torch.float

class TrajModelTrainer():
    def __init__(self, env_name, episodes, val_size=0, model_save_path=None, error_save_path=None, norm_path = None, state_dim = 4, action_dim = 6, ho_rate=0.99995, save=True):
        self.env_name=env_name
        self.ho_rate=ho_rate
        self.task_ofs = state_dim+action_dim
        self.state_dim = state_dim
        self.new_state_dim = state_dim
        self.action_dim = action_dim
        self.save=save
        self.norm_path = norm_path
        self.model_save_path = model_save_path
        self.error_save_path = error_save_path
        self.dtype = torch.float
        self.val_size = val_size
        self.cuda=cuda
        if val_size:
            self.episodes=episodes[:-val_size]
            episodes=np.concatenate(episodes)
            np.random.shuffle(episodes)
            self.norm, data, val_data= self.get_norms(episodes, val_size)
            self.x_val_data, self.y_val_data = val_data
        else:
            self.episodes=episodes
            episodes=np.concatenate(episodes)
            np.random.shuffle(episodes)
            self.norm, data= self.get_norms(episodes, val_size)
        self.x_data, self.y_data = data

    def train(self, model, opt, every_plot, epochs = 30, batch_size = 64, clip_grad_norm=True, loss_fn = torch.nn.MSELoss(), retrain=False,epoch_save_interval=5):
        if self.cuda: 
            self.x_data=self.x_data.to('cuda')
            self.y_data=self.y_data.to('cuda')
            if self.val_size:
                self.x_val_data=self.x_val_data.to('cuda')
                self.y_val_data=self.y_val_data.to('cuda')
        dataset = torch.utils.data.TensorDataset(self.x_data, self.y_data)
        loader = torch.utils.data.DataLoader(dataset, batch_size = batch_size)

        if retrain:
            with open(self.error_save_path, 'rb') as pickle_file:
                train_loss_ls, val_loss_ls = pickle.load(pickle_file)
        else:
            train_loss_ls,val_loss_ls=[],[]

        x_mean_arr, x_std_arr, y_mean_arr, y_std_arr = self.norm
        for i in tqdm(range(epochs)):
            total_train_loss=0
            model.train()
            for batch_ndx, sample in enumerate(loader):
                opt.zero_grad()
                output = model(sample[0])

                loss = loss_fn(output, sample[1]) 
                total_train_loss += loss.data*sample[0].shape[0]
                loss.backward()
                if clip_grad_norm:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
                opt.step()
            train_loss=total_train_loss/self.x_data.shape[0]
            if not retrain and i % every_plot==0:
                print("First Training epoch: " + str(i+1)+" ,average_train_loss: " + str(train_loss.item()))
            elif i % every_plot==0:
                raise Exception("Not implemented!")
                #print("Re-Training epoch: " + str(i)+" ,average_train_loss: " + str(train_loss.item()))
            train_loss_ls.append(train_loss.item())

            model.eval()
            with torch.no_grad():
                if self.val_size:
                    val_output = model(self.x_val_data)
                    val_loss = loss_fn(val_output, self.y_val_data) 
                    val_loss_ls.append(val_loss.item())
                    if (i+1) % every_plot==0:
                        loss_print="average_validation_loss: " + str(val_loss.item())
                        print(loss_print)     
            if (i+1) % epoch_save_interval ==0:
                with open(self.model_save_path+'_epochs_'+str(i+1), 'wb') as pickle_file:
                    torch.save(model, pickle_file)
        if self.save:
            if self.env_name!='gazebo_ah':
                with open(self.model_save_path+'_epochs_'+str(epochs), 'wb') as pickle_file:
                    torch.save(model, pickle_file)
            else:
                with open(self.model_save_path, 'wb') as pickle_file:
                    torch.save(model, pickle_file)
        with open(self.error_save_path, 'wb') as pickle_file:
            pickle.dump([train_loss_ls,val_loss_ls],pickle_file)
        return train_loss_ls,val_loss_ls

    def get_norms(self, episodes, val_size):
        FULL_DATA = episodes
        if self.val_size:
            DATA = FULL_DATA[:-val_size,:]
            VAL_DATA=FULL_DATA[-val_size:,:]
            x_val_data = VAL_DATA[:, :self.task_ofs]
            y_val_data = VAL_DATA[:, -self.new_state_dim:] - VAL_DATA[:, :self.new_state_dim]
        else:
            DATA=FULL_DATA

        x_data = DATA[:, :self.task_ofs]
        y_data = DATA[:, -self.new_state_dim:] - DATA[:, :self.new_state_dim]

        full_x_data = FULL_DATA[:, :self.task_ofs]
        full_y_data = FULL_DATA[:, -self.new_state_dim:] - FULL_DATA[:, :self.new_state_dim]

        x_mean_arr = np.mean(full_x_data, axis=0)
        x_std_arr = np.std(full_x_data, axis=0)

        if self.env_name=='Reacher-v2' or self.env_name=='gazebo_ah':
            x_mean_arr = np.concatenate((x_mean_arr[:self.state_dim], np.array([0,0]), x_mean_arr[-self.state_dim:]))
            x_std_arr = np.concatenate((x_std_arr[:self.state_dim], np.array([1,1]), x_std_arr[-self.state_dim:]))
        elif self.env_name=='Acrobot-v1':
            x_mean_arr = np.concatenate((x_mean_arr[:self.state_dim], np.array([0]), x_mean_arr[-self.state_dim:]))
            x_std_arr = np.concatenate((x_std_arr[:self.state_dim], np.array([1]), x_std_arr[-self.state_dim:]))
        else:
            raise Exception("Not implemented!")

        y_mean_arr = np.mean(full_y_data, axis=0)
        y_std_arr = np.std(full_y_data, axis=0)

        x_data = z_score_normalize(x_data, x_mean_arr, x_std_arr)
        y_data = z_score_normalize(y_data, y_mean_arr, y_std_arr)
        if self.val_size:
            x_val_data = z_score_normalize(x_val_data, x_mean_arr, x_std_arr)
            y_val_data = z_score_normalize(y_val_data, y_mean_arr, y_std_arr)
        if self.norm_path:
            if self.env_name!='gazebo_ah':
                if self.ho_rate==0:
                    with open(self.norm_path+'normalization_arr', 'wb') as pickle_file:
                        pickle.dump(((x_mean_arr, x_std_arr),(y_mean_arr, y_std_arr)), pickle_file)
                else:
                    with open(self.norm_path+'normalization_arr_ho'+str(self.ho_rate), 'wb') as pickle_file:
                        pickle.dump(((x_mean_arr, x_std_arr),(y_mean_arr, y_std_arr)), pickle_file)
            else:
                with open(self.norm_path+'normalization_arr_sim_cont_trajT_bs512_model512_BS64_loadT_ho'+str(self.ho_rate)+'_py2', 'wb') as pickle_file:
                    pickle.dump(((x_mean_arr, x_std_arr),(y_mean_arr, y_std_arr)), pickle_file)

        x_data = torch.tensor(x_data, dtype=self.dtype)
        y_data = torch.tensor(y_data, dtype=self.dtype)
        if self.val_size:
            x_val_data = torch.tensor(x_val_data, dtype=self.dtype)
            y_val_data = torch.tensor(y_val_data, dtype=self.dtype)

        x_mean_arr = torch.tensor(x_mean_arr, dtype=self.dtype)
        x_std_arr = torch.tensor(x_std_arr, dtype=self.dtype)
        y_mean_arr = torch.tensor(y_mean_arr, dtype=self.dtype)
        y_std_arr = torch.tensor(y_std_arr, dtype=self.dtype)

        if self.cuda:
            x_mean_arr = x_mean_arr.cuda()
            x_std_arr = x_std_arr.cuda()
            y_mean_arr = y_mean_arr.cuda()
            y_std_arr = y_std_arr.cuda()

        if self.val_size:
            return (x_mean_arr, x_std_arr, y_mean_arr, y_std_arr), (x_data, y_data), (x_val_data, y_val_data)
        else:
            return (x_mean_arr, x_std_arr, y_mean_arr, y_std_arr), (x_data, y_data)
