import torch
import torch.nn.functional as F
from common.data_normalization import *
from common.pt_build_model import *
import pdb
import numpy as np
import random

dtype = torch.float
cuda = torch.cuda.is_available()
#cuda = False

class ResBlock(torch.nn.Module):
    def __init__(self, model, coeff = 1):
        super(ResBlock, self).__init__()
        self.model = model
        self.coeff = coeff       
        self.model.add_start=False
    def forward(self, x):
        temp = self.model(x)
        min_len = min(temp.shape[-1], x.shape[-1])
        # pdb.set_trace()
        # out = x
        out = x[...,:min_len] + temp[...,:min_len]*self.coeff
        return out

class TrivialNet(torch.nn.Module):
    def forward(self, x):
        return x

# def transfer(x, state_dim): 
#     return torch.cat((x[...,:state_dim], x[...,state_dim:state_dim+2]*-1,  x[...,state_dim+2:]), -1) 

class TrajNet(torch.nn.Module):
    def __init__(self, norms, model = None ,nn_type='1', nodes=200, state_dim = 4, action_dim = 6, dropout_rate=.1, both_nn_type='1'):
        super(TrajNet, self).__init__()
        self.task_ofs = state_dim + action_dim
        self.nn_type=nn_type
        self.nodes=nodes
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.dropout_rate=dropout_rate
        self.both_nn_type=both_nn_type
        self.dtype = torch.float
        self.norm = norms

        if model:
            self.model = model
        else:
            self.model = pt_build_model(nn_type, nodes, state_dim + action_dim, state_dim, dropout_p=dropout_rate, both_nn_type=both_nn_type)

        if cuda: 
            self.model = self.model.cuda()
            self.norm = tuple(n.cuda() for n in self.norm)
        self.res = None
        self.coeff = 1

   


    def run_traj(self, gt_states,real_actions):
        x_mean_arr, x_std_arr, y_mean_arr, y_std_arr = self.norm
        state = gt_states[0,:]
        states = [state]
        if cuda:
            state = state.cuda()
            gt_states = gt_states.cuda()
        for i in range(real_actions.shape[0]):
            action = real_actions[i,:]
            if cuda: action = action.cuda() 
            inpt = torch.cat((state, action), -1)
            inpt = z_score_norm_single(inpt, x_mean_arr, x_std_arr)
            state_delta = self.forward(inpt)
            state_delta = z_score_denorm_single(state_delta, y_mean_arr, y_std_arr)
            state= state_delta + state
            states.append(state)
        states = torch.stack(states,0)
        states = states.detach().numpy()
        return states
    def run_traj_true_load(self, gt_states,real_actions_loads):
        x_mean_arr, x_std_arr, y_mean_arr, y_std_arr = self.norm
        state = gt_states[0,:]
        states = [state]
        if cuda:
            state = state.cuda()
            gt_states = gt_states.cuda()
        for i in range(real_actions_loads.shape[0]):
            action = real_actions_loads[i,2:]
            if cuda: action = action.cuda() 
            inpt = torch.cat((state, action), -1)
            inpt = z_score_norm_single(inpt, x_mean_arr, x_std_arr)
            state_delta = self.forward(inpt)
            state_delta = z_score_denorm_single(state_delta, y_mean_arr, y_std_arr)
            state= state_delta + state
            states.append(state)
            if i!=real_actions_loads.shape[0]-1:
                state[2:4]=real_actions_loads[i+1,:2]
        states = torch.stack(states,0)
        states = states.detach().numpy()
        return states

    def forward(self, x):
        out = self.model(x)
        return out




class ScaleNet(TrajNet):
    def forward(self, x):
        out = self.model(x)
        try:
            if self.res != None:
                if self.training:
                    out += self.res(x)*self.coeff
                else:
                    r = self.res(x)
                    residual = r/(1+r)
                    return out + residual
        except Exception as e:
            pass
        return out


class GPNetclass(TrajNet):
    def forward(self, x):
        out = self.model(x)
        try:
            if self.res != None:
                out += self.res(x)
        except Exception as e:
            pass
        return out

# class LSTMStateTrajNet(TrajNet):
class LSTMStateTrajNet(torch.nn.Module):
    def __init__(self, task, norms, model = None ,state_dim = 4, action_dim = 2, task_ofs = 10, reg_loss = None):
        super(LSTMStateTrajNet, self).__init__()
        self.task_ofs = task_ofs
        self.state_dim = state_dim
        self.task = task
        self.action_dim = action_dim
        # self.action_dim = 2
        self.dtype = torch.float

        self.norm = norms
        self.reg_loss = reg_loss 
        self.LSTM = False
        h = 100
        self.h = h
        # self.l1 = torch.nn.GRU(action_dim+state_dim, state_dim, dropout = .1, batch_first = True)
        self.l1 = torch.nn.GRU(action_dim+state_dim, h, dropout = .25, batch_first = True, num_layers=2)
        self.l2 = torch.nn.GRU(h, state_dim, dropout = .25, batch_first = True)
        self.prediction = 'state'
        self.add_start=True

    def run_traj(self, batch, threshold = 50, sub_chance = 0.0):
        x_mean_arr, x_std_arr, y_mean_arr, y_std_arr = self.norm 
        inpt = batch[...,:self.state_dim+self.action_dim]

        # state0 = batch[...,0,:self.state_dim]
        # state_stack = torch.stack([state0]*batch.shape[-2],-2)
        # actions = batch[...,self.state_dim:self.state_dim+self.action_dim]
        # inpt = torch.cat([state_stack, actions], -1)

        inpt = z_score_normalize(inpt, x_mean_arr, x_std_arr)
        out  = self.forward(inpt)
        out = z_score_denormalize(out, x_mean_arr, x_std_arr)

        # out = z_score_denormalize(out, y_mean_arr, y_std_arr)
        # out = torch.cumsum(out, -2)
        # state_stack = torch.stack([state0]*out.shape[-2],-2)
        # if self.add_start:
        #     out = out + state_stack
        return out

    def forward(self, inpt, given_hidden = None, norm=True):
        self.system = self.task[-1]

        x_mean_arr, x_std_arr, y_mean_arr, y_std_arr = self.norm
        hidden = given_hidden

        if len(inpt.shape) == 1:
            inpt = inpt.unsqueeze(0)
        if len(inpt.shape) == 2:
            inpt = inpt.unsqueeze(1)

        if hidden == None or hidden == 'fill':
            m1 = torch.zeros(2, inpt.shape[0] ,self.h)
            sd1 = torch.ones(2, inpt.shape[0] ,self.h)

            m2 = torch.zeros(1, inpt.shape[0] ,self.state_dim)
            sd2 = torch.ones(1, inpt.shape[0] ,self.state_dim)

            h1_distro = torch.distributions.normal.Normal(m1, sd1)
            h2_distro = torch.distributions.normal.Normal(m2, sd2)

            if self.LSTM:
                hc1 = (h1_distro.sample(), h1_distro.sample())
                hc2 = (h2_distro.sample(), h2_distro.sample())
                hidden = (hc1, hc2)
            # hidden = (h1_distro.sample(), inpt.transpose(0,1)[:1,:,:self.state_dim])
            hidden = (h1_distro.sample(), h2_distro.sample()*0)
            if cuda:
                hidden= (hidden[0].cuda(), hidden[1].cuda())


        # if norm: inpt = z_score_normalize(inpt, x_mean_arr, x_std_arr)

        f1, h1 = self.l1(inpt, hidden[0])    
        # dyn = self.dyn(inpt)
        mid = F.dropout(f1, .2)
        # mid += dyn 
        f2, h2 = self.l2(mid, hidden[1])
        out = f2.squeeze(1)

        # if norm: out = z_score_denormalize(out, x_mean_arr, x_std_arr)

        if given_hidden == None:
            return out
        return out, (h1,h2)

class LSTMTrajNet(LSTMStateTrajNet):
    def __init__(self, task, norms, model = None ,state_dim = 4, action_dim = 2, task_ofs = 10, reg_loss = None):
        super(LSTMTrajNet, self).__init__(task, norms)
        self.task_ofs = task_ofs
        self.state_dim = state_dim
        self.task = task
        # self.action_dim = action_dim
        self.action_dim = 2
        self.dtype = torch.float

        self.norm = norms
        self.reg_loss = reg_loss 
        self.LSTM = False
        h = 100
        self.h = h
        # self.l1 = torch.nn.GRU(action_dim+state_dim, state_dim, dropout = .1, batch_first = True)
        self.l1 = torch.nn.GRU(action_dim+state_dim, h, dropout = .25, batch_first = True, num_layers=2)
        self.l2 = torch.nn.GRU(h, state_dim, dropout = .25, batch_first = True)
        self.prediction = 'delta'
        self.model = pt_build_model('1', state_dim + action_dim, state_dim, dropout_p=.1)


    def run_traj(self, batch, threshold = 50, sub_chance = 0.0):
        dim = 3
        if len(batch.shape) == 2:
            batch = batch.unsqueeze(0)
            dim = 2
        x_mean_arr, x_std_arr, y_mean_arr, y_std_arr = self.norm
        true_states = batch[...,:,:self.state_dim]
        state = batch[...,0,:self.state_dim]
        states = []#state.view(1, self.state_dim)
        sim_deltas = []
        if cuda:
            state = state.cuda()
            true_states = true_states.cuda()

        
        hidden = 'fill'
        for i in range(batch.shape[1]):
            states.append(state)

            if random.random() < sub_chance: 
                state=true_states[:,i]

            if cuda: action = action.cuda() 
            # if 'transfer' in self.task:
            #     action[...,:2]*= -1


            inpt = torch.cat((state, action), -1)
            inpt = z_score_norm_single(inpt, x_mean_arr, x_std_arr)

            state_delta, hidden = self.forward(inpt, hidden, norm=False)

            state_delta = z_score_denorm_single(state_delta, y_mean_arr, y_std_arr)
            # if self.task in ['transferA2B', 'transferB2A']: 
            #     state_delta *= torch.tensor([-1,-1,1,1], dtype=torch.float)
            sim_deltas.append(state_delta)

            state= state_delta + state

            #May need random component here to prevent overfitting
            if threshold and i%10:
                with torch.no_grad():
                    mse_fn = torch.nn.MSELoss()
                    mse = mse_fn(state[...,:2], true_states[...,i,:2])
                if mse > threshold:
                    states = torch.stack(states, -2)
                    return states

        states = torch.stack(states, -2)
        if dim == 2:
            states=states.squeeze(0)
        return states


class LatentNet(torch.nn.Module):
    def __init__(self, task, norms, model = None, state_dim = 4, action_dim = 6, task_ofs = 10, reg_loss = None):
        super(LatentNet, self).__init__()
        self.task_ofs = task_ofs
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.dtype = torch.float

        self.norm = norms
        self.reg_loss = reg_loss   
        self.internal_state_dim = state_dim
        if model:
            self.model = model
            for param in self.model.parameters():
                param.requires_grad = False
        else:
            self.model = TrajNet(task, norms)

        if model.task[-1] == 'A':
            model.task = 'transferA2B'
        else:
            model.task = 'transferB2A'
        # self.encoder = ResBlock(pt_build_model('1', state_dim+action_dim, state_dim+action_dim, dropout_p=.1))
        # self.decoder = ResBlock(pt_build_model('1', state_dim, state_dim, dropout_p=.1))

        self.encoder = ResBlock(LSTMStateTrajNet(task,norms,state_dim=state_dim+action_dim, action_dim=0))
        self.encoder.coeff = 0
        self.decoder = ResBlock(LSTMStateTrajNet(task,norms,state_dim=state_dim, action_dim=action_dim))
        self.decoder = ResBlock(LSTMTrajNet(task,norms,state_dim=state_dim, action_dim=action_dim))

        if cuda:
            self.model.norm = tuple(n.cuda() for n in self.model.norm)
            self.encoder = self.encoder.cuda()
            self.decoder = self.decoder.cuda()
            self.model = self.model.to('cuda')



    def run_traj(self, batch, threshold = 50, sub_chance=0.0):
        true_states = batch[...,:,:self.state_dim]
        actions = batch[...,self.state_dim:self.state_dim+self.action_dim]

        # encoded_true_states = self.encoder(true_states)

        # pass_batch = torch.cat([encoded_true_states, actions], -1)
        pass_batch = batch[...,:self.state_dim+self.action_dim]
        pass_batch = self.encoder(pass_batch)
        projected_states = self.model.run_traj(pass_batch, threshold=threshold, sub_chance=sub_chance)
        states = projected_states
        # mod_actions = actions.view(-1, actions.shape[-2], actions.shape[-1])[:, :projected_states.shape[1]]
        mod_actions = actions[..., :projected_states.shape[-2], :]
        if projected_states.shape[0] == 1:
            projected_states = projected_states.squeeze(0)
        # pdb.set_trace()
        projected_states = torch.cat([projected_states, mod_actions], -1)
        states = self.decoder(projected_states)

        return states

    def forward(self, x):
        state = x[...,:self.state_dim]
        actions = x[...,:,self.state_dim:self.state_dim+self.action_dim]
        x_mean_arr, x_std_arr, y_mean_arr, y_std_arr = self.norm
        pass_batch = self.encoder(x[...,:self.state_dim+self.action_dim])
        # projected_states = self.model.run_traj(pass_batch)

        state_delta = self.model(pass_batch)

        state_delta = z_score_denorm_single(state_delta, y_mean_arr, y_std_arr)
        projected_states= state_delta + state
        projected_states = torch.cat([projected_states, actions], -1)
        state_delta = self.decoder(projected_states).squeeze(1)- state
        state_delta = z_score_norm_single(state_delta , y_mean_arr, y_std_arr)

        return state_delta
        # pass_batch = self.encoder(x[...,:self.state_dim+self.action_dim])
        # return self.model(pass_batch)


class LatentDeltaNet(torch.nn.Module):
    def __init__(self, task, norms, model = None, state_dim = 4, action_dim = 6, task_ofs = 10, reg_loss = None):
        super(LatentDeltaNet, self).__init__()
        self.task_ofs = task_ofs
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.dtype = torch.float

        self.norm = norms
        self.reg_loss = reg_loss   
        self.internal_state_dim = state_dim
        if model:
            self.model = model
            for param in self.model.parameters():
                param.requires_grad = False

            if model.task[-1] == 'A':
                model.task = 'transferA2B'
            else:
                model.task = 'transferB2A'
        else:
            self.model = TrajNet(task, norms)
            

        # self.encoder = ResBlock(pt_build_model('1', state_dim+action_dim, state_dim+action_dim, dropout_p=.1))
        # self.decoder = ResBlock(pt_build_model('1', state_dim, state_dim, dropout_p=.1))

        self.encoder = ResBlock(LSTMStateTrajNet(task,norms,state_dim=state_dim+action_dim, action_dim=0))
        self.encoder.coeff = 0
        # self.decoder = ResBlock(LSTMStateTrajNet(task,norms,state_dim=state_dim, action_dim=action_dim))
        # self.decoder = TrajNet(task,norms,state_dim=state_dim, action_dim=action_dim)
        self.decoder = LSTMTrajNet(task,norms,state_dim=state_dim, action_dim=action_dim)

        if cuda:
            self.model.norm = tuple(n.cuda() for n in self.model.norm)
            self.encoder = self.encoder.cuda()
            self.decoder = self.decoder.cuda()
            self.model = self.model.to('cuda')

        self.coeff = 1



    def run_traj(self, batch, threshold = 50, sub_chance=0.0):
        if batch.shape[0] == 1:
            batch = batch.squeeze(0)
        x_mean_arr, x_std_arr, y_mean_arr, y_std_arr = self.norm
        true_states = batch[...,:self.state_dim]
        actions = batch[...,self.state_dim:self.state_dim+self.action_dim]

        # encoded_true_states = self.encoder(true_states)

        # pass_batch = torch.cat([encoded_true_states, actions], -1)
        pass_batch = batch[...,:self.state_dim+self.action_dim]
        # pass_batch = self.encoder(pass_batch)
        # projected_states = self.model.run_traj(pass_batch, threshold=threshold, sub_chance=sub_chance)
        projected_states = self.model.run_traj(pass_batch, threshold=None, sub_chance=sub_chance)
        states = projected_states
        # mod_actions = actions.view(-1, actions.shape[-2], actions.shape[-1])[:, :projected_states.shape[1]]
        mod_actions = actions[..., :projected_states.shape[-2], :]
        if projected_states.shape[0] == 1:
            projected_states = projected_states.squeeze(0)
        # pdb.set_trace()
        projected_states = torch.cat([projected_states, mod_actions], -1)

        projected_states = z_score_normalize(projected_states, x_mean_arr, x_std_arr)
        deltas = self.decoder(projected_states)
        deltas2 = z_score_denormalize(deltas, y_mean_arr, y_std_arr)*self.coeff
        distance = torch.cumsum(deltas2, dim=-2)
        # pdb.set_trace()
        states  = distance + states 

        return states

    def forward(self, x):
        state = x[...,:self.state_dim]
        actions = x[...,:,self.state_dim:self.state_dim+self.action_dim]
        x_mean_arr, x_std_arr, y_mean_arr, y_std_arr = self.norm
        pass_batch = x[...,:self.state_dim+self.action_dim]
        # pass_batch = self.encoder(pass_batch)
        # projected_states = self.model.run_traj(pass_batch)

        state_delta = self.model(pass_batch)

        state_delta = z_score_denorm_single(state_delta, y_mean_arr, y_std_arr)
        # projected_states= state_delta + state
        # projected_states = torch.cat([projected_states, actions], -1)
        # state_delta2 = 0
        # projected_states = z_score_normalize(pass_batch, x_mean_arr, x_std_arr)
        state_delta2 = self.decoder(pass_batch)#.squeeze(1)
        # state_delta2 = z_score_denorm_single(state_delta2, y_mean_arr, y_std_arr)
        state_delta = state_delta2 + state_delta

        return state_delta
        # pass_batch = self.encoder(x[...,:self.state_dim+self.action_dim])
        # return self.model(pass_batch)

class TimeIndepLatentDeltaNet(LatentDeltaNet):
    def __init__(self, task, norms, model = None, state_dim = 4, action_dim = 6, task_ofs = 10, reg_loss = None):
        super(TimeIndepLatentDeltaNet, self).__init__(task, norms, model = model, state_dim = state_dim, 
            action_dim = action_dim, task_ofs = task_ofs, reg_loss = reg_loss)
        self.decoder = TrajNet(task, norms,state_dim=state_dim, action_dim=action_dim)
        for param in self.model.parameters():
            param.requires_grad = False



class TimeIndepChainedLatentDeltaNet(TrajNet):
    def __init__(self, task, norms, model = None, state_dim = 4, action_dim = 6, task_ofs = 10, reg_loss = None):
        super(TimeIndepChainedLatentDeltaNet, self).__init__(task, norms, model = model, state_dim = state_dim, 
            action_dim = action_dim, task_ofs = task_ofs, reg_loss = reg_loss)
        # self.task = task
        # self.state_dim = state_dim
        # self.action_dim = action_dim
        # self.dtype = torch.float
        # self.norm = norms
        self.decoder = TrajNet(task, norms,state_dim=state_dim, action_dim=action_dim)
        # self.coeff = 1
        # for param in self.model.parameters():
        #     param.requires_grad = False


    def forward(self, x):
        # if 'transfer' in self.task:
        #     x = transfer(x, self.state_dim)
        out = super().forward(x)
        out += self.decoder(x)*0
        # .decoder(x)*0#self.coeff
        return out
