import numpy as np 
import pdb
import torch
import torch.utils.data
from common.data_normalization import *
import random
import pickle

import matplotlib.pyplot as plt


cuda = torch.cuda.is_available()
cuda = False
dtype = torch.float


def clean_data(episodes):
    DATA = np.concatenate(episodes)
    yd_pos = DATA[:, -4:-2] - DATA[:, :2]
    y2 = np.sum(yd_pos**2, axis=1)
    max_dist = np.percentile(y2, 99.84)
    # max_dist = np.percentile(y2, 99.6)

    skip_list = [np.sum((ep[:, -4:-2] - ep[:, :2])**5, axis=1)>max_dist for ep in episodes]
    divided_episodes = []
    for i,ep in enumerate(episodes):
        if np.sum(skip_list[i]) == 0:
            divided_episodes += [ep]

        else: 
            ep_lists = np.split(ep, np.argwhere(skip_list[i]).reshape(-1))
            divided_episodes += ep_lists

    divided_episodes = [ep[3:-3] for ep in divided_episodes]

    length_threshold = 30
    return list(filter(lambda x: len(x) > length_threshold, divided_episodes))


def softmax(states, true_states):
    mse_fn = torch.nn.MSELoss(reduction='none')
    mse = mse_fn(states[...,:,:2], true_states[...,:states.shape[-2],:2])
    mse = torch.mean(mse, -1) #Sum two position losses at each time step to get the Euclidean distance 
    loss = torch.logsumexp(mse, -1) #Softmax divergence over the path
    loss = torch.mean(loss) #Sum over batch
    return loss

def pointwise(states, true_states, scaling = True):
    ts = true_states[...,:states.shape[-2],:4]
    ts = ts.view_as(states)
    # if true_states.shape[0] == 1:
    # pdb.set_trace()
    if scaling:
        mse_fn = torch.nn.MSELoss(reduction='none')
        scaling = 1/((torch.arange(states.shape[-2]-1, dtype=torch.float)+1))
        if cuda: scaling = scaling.cuda()
        # loss_temp = mse_fn(states[...,1:,:2], true_states[...,1:states.shape[-2],:2])
        # loss_temp += mse_fn(states[...,1:,2:4], true_states[...,1:states.shape[-2],2:4])*.1
        loss_temp = mse_fn(states[...,1:,:2], ts[...,1:,:2])
        loss_temp += mse_fn(states[...,1:,2:4], ts[...,1:,2:4])*.1
        loss = torch.einsum('...kj,k->', [loss_temp, scaling])/loss_temp.numel()
        return loss
    else: 
        mse_fn = torch.nn.MSELoss()
        # return mse_fn(states[...,1:,:2], ts[...,:2])
        return mse_fn(states[...,:2], ts[...,:2])


#--------------------------------------------------------------------------------------------------------------------

class TrajModelTrainer():
    def __init__(self, task, episodes,  method=None, save=True, model_save_path=None, save_path = None, state_dim = 4, 
            action_dim = 6, task_ofs = 10, reg_loss = None, nn_type=None, held_out = .1):
        self.task_ofs = task_ofs
        self.state_dim = state_dim
        self.new_state_dim = state_dim
        self.action_dim = action_dim
        self.task = task
        self.method = method
        self.save=save
        self.save_path = save_path
        self.model_save_path = model_save_path
        self.dtype = torch.float
        self.cuda = cuda
        self.held_out = held_out

        self.norm, data, self.episodes= self.get_norms(episodes, held_out)
        self.x_data, self.y_data = data
        self.nn_type= nn_type
        self.reg_loss = reg_loss        



    def pretrain(self, model, opt, train_load = True, epochs = 30, batch_size = 64, clip_grad_norm=True, loss_fn = torch.nn.MSELoss()):
        if cuda: 
            self.x_data=self.x_data.to('cuda')
            self.y_data=self.y_data.to('cuda')
        dataset = torch.utils.data.TensorDataset(self.x_data, self.y_data)
        loader = torch.utils.data.DataLoader(dataset, batch_size = batch_size)

        # pdb.set_trace()
        x_mean_arr, x_std_arr, y_mean_arr, y_std_arr = self.norm
        for i in range(epochs):
            print("Pretraining epoch: " + str(i))
            total_loss = 0
            for batch_ndx, sample in enumerate(loader):
                opt.zero_grad()

                if self.nn_type == 'LSTM':
                    output = model(sample[0])
                else:
                    output = model(sample[0])

                loss = loss_fn(output, sample[1]) 
                if self.reg_loss != None: 
                    loss += self.reg_loss(model)
                total_loss += loss.data
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
                opt.step()
            print("total_loss: " + str(total_loss))
        # if self.save:
        with open(self.model_save_path, 'wb') as pickle_file:
            torch.save(model, pickle_file)


    def visualize(self, model,  episode):
        if isinstance(episode, list):
            [visualize(model, ep) for ep in episode]
        states = model.run_traj(episode)

        episode = episode.cpu().detach().numpy()
        states = states.cpu().detach().numpy()
        # pdb.set_trace()

        plt.figure(1)
        plt.plot(episode[..., 0], episode[..., 1], color='blue', label='Ground Truth', marker='.')
        plt.plot(states[..., 0], states[ ..., 1], color='red', label='NN Prediction')
        plt.axis('scaled')
        plt.legend()
        plt.show()


    def get_norms(self, episodes, held_out):
        full_dataset = episodes

        val_size = int(len(episodes)*held_out)

        val_data = episodes[-val_size:]
        val_data = val_data[:min(10, len(val_data))]
        episodes = episodes[:-val_size]

        DATA = np.concatenate(episodes)
        FULL_DATA = np.concatenate(full_dataset)

        x_data = DATA[:, :self.task_ofs]
        y_data = DATA[:, -self.new_state_dim:] - DATA[:, :self.new_state_dim]


        full_x_data = FULL_DATA[:, :self.task_ofs]
        full_y_data = FULL_DATA[:, -self.new_state_dim:] - FULL_DATA[:, :self.new_state_dim]

        x_mean_arr = np.mean(full_x_data, axis=0)
        x_std_arr = np.std(full_x_data, axis=0)

        x_mean_arr = np.concatenate((x_mean_arr[:self.state_dim], np.array([0,0]), x_mean_arr[-self.state_dim:]))
        x_std_arr = np.concatenate((x_std_arr[:self.state_dim], np.array([1,1]), x_std_arr[-self.state_dim:]))
        # x_mean_arr[self.state_dim:-self.state_dim] *= 0
        # x_std_arr[self.state_dim:-self.state_dim] *= 0
        # x_std_arr[self.state_dim:-self.state_dim] += 1

        y_mean_arr = np.mean(full_y_data, axis=0)
        y_std_arr = np.std(full_y_data, axis=0)

        x_data = z_score_normalize(x_data, x_mean_arr, x_std_arr)
        y_data = z_score_normalize(y_data, y_mean_arr, y_std_arr)
        if self.save_path:
            with open(self.save_path+'/normalization_arr/normalization_arr', 'wb') as pickle_file:
                pickle.dump(((x_mean_arr, x_std_arr),(y_mean_arr, y_std_arr)), pickle_file)

        x_data = torch.tensor(x_data, dtype=self.dtype)
        y_data = torch.tensor(y_data, dtype=self.dtype)

        x_mean_arr = torch.tensor(x_mean_arr, dtype=self.dtype)
        x_std_arr = torch.tensor(x_std_arr, dtype=self.dtype)
        y_mean_arr = torch.tensor(y_mean_arr, dtype=self.dtype)
        y_std_arr = torch.tensor(y_std_arr, dtype=self.dtype)

        if self.cuda:
            x_mean_arr = x_mean_arr.cuda()
            x_std_arr = x_std_arr.cuda()
            y_mean_arr = y_mean_arr.cuda()
            y_std_arr = y_std_arr.cuda()

        return (x_mean_arr, x_std_arr, y_mean_arr, y_std_arr), (x_data, y_data), episodes



    def batch_train(self, model, opt, val_data = None, epochs = 500, batch_size = 8, loss_type = 'pointwise', sub_chance = 0.0, scaling=True):
        j=0
        episodes= self.episodes
        if cuda:
            episodes = [ep.to('cuda') for ep in episodes]
            if val_data: 
                val_data = [ep.to('cuda') for ep in val_data]
        print('\nBatched trajectory training with batch size ' + str(batch_size))
        for epoch in range(epochs):
            grad_norms = []

            print('Epoch: ' + str(epoch))
            np.random.shuffle(episodes)

            total_loss = 0
            # pdb.set_trace()
            total_completed = 0
            total_distance = 0
            switch = True
            thresh = 150


            batch_lists = [episodes[i: min(len(episodes), i+ batch_size)] for i in range(0, len(episodes), batch_size)] 
            episode_lengths = [[len(ep) for ep in batch] for batch in batch_lists]
            min_lengths = [min(episode_length) for episode_length in episode_lengths]
            rand_maxes = [[len(episode) - min_length for episode in batch_list] for batch_list, min_length in zip(batch_lists,min_lengths)]
            rand_starts = [[random.randint(0, rmax) for rmax in rmaxes] for rmaxes in rand_maxes]
            batch_slices = [[episode[start:start+length] for episode, start in zip(batch, starts)] for batch, starts, length in zip(batch_lists, rand_starts, min_lengths)]

            batches = [torch.stack(batch, 0) for batch in batch_slices] 

            accum = 8//batch_size
 
            for i, batch in enumerate(batches): 
                if accum == 0 or j % accum ==0: 
                    opt.zero_grad()

                j += 1

                states = model.run_traj(batch, threshold = thresh, sub_chance=sub_chance)

                loss = pointwise(states, batch, scaling=scaling)
                # loss = softmax(states, batch)
                # loss = pointwise(states, batch, scaling = False)
                completed = (states.shape[-2] == batch.shape[-2])
                dist = states.shape[-2]
                total_loss += loss.data
                total_completed += completed
                total_distance += dist               

                # pdb.set_trace()
                loss.backward()
                if accum == 0 or j % accum ==0:
                    if self.reg_loss: loss += self.reg_loss(model)

                    torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
                    opt.step()

            if val_data:
                total_loss = 0
                total_completed = 0
                total_distance = 0
                for i, episode in enumerate(val_data[:len(val_data)//2]):
                    states = model.run_traj(episode, threshold = thresh)
                    completed = (states.shape[-2] == episode.shape[-2])
                    dist = states.shape[-2]
                    total_completed += completed
                    total_distance += dist

                states_list = []
                loss_list = []
                eps = val_data[len(val_data)//2:]
                for i, episode in enumerate(val_data[len(val_data)//2:]):
                    states = model.run_traj(episode, threshold = None)
                    # val_loss = softmax(states, episode)
                    val_loss = pointwise(states, episode, scaling=False)
                    states_list.append(states)
                    loss_list.append(val_loss)
                    total_loss += val_loss.data
                print('Loss: ' + str(total_loss/(len(val_data)/2)))
                print('completed: ' + str(total_completed/(len(val_data)/2)))
                print('Average time before divergence: ' + str(total_distance/(len(val_data)/2)))

                # model.model = model.model.eval()
                episode = random.choice(val_data)

            else:
                print('Loss: ' + str(total_loss/len(batches)))
                print('completed: ' + str(total_completed/len(batches)))
                print('Average time before divergence: ' + str(total_distance/len(batches)))

            # with open(self.model_save_path, 'wb') as pickle_file:
            #     torch.save(model, pickle_file)
