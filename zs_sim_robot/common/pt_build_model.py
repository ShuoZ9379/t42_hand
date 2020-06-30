import torch
import torch.nn.functional as F
import pdb
import numpy as np

dtype = torch.float




class SplitModel(torch.nn.Module):
    def __init__(self, nodes, input_dim, output_dim, dropout_p = .1, both_nn_type='1'):
        super(SplitModel, self).__init__()
        self.pos_model = pt_build_model(both_nn_type, nodes, input_dim, output_dim//2, dropout_p)
        self.load_model = pt_build_model(both_nn_type, nodes, input_dim, output_dim//2, dropout_p)

    def forward(self, x):
        pos_out = self.pos_model(x)
        load_out = self.load_model(x)
        return torch.cat((pos_out, load_out), -1)


# class LSTMNet(torch.nn.Module):
#     def __init__(self, action_dim, state_dim):
#         super(LSTMNet, self).__init__()
#         self.state_dim = state_dim
#         self.action_dim = action_dim
#         self.l1 = torch.nn.GRU(action_dim, state_dim, dropout = .1, batch_first = True)
#         self.l2 = torch.nn.GRU(state_dim, 100, dropout = .1, batch_first = True, num_layers=1)
#         self.l3 = torch.nn.GRU(100, state_dim, dropout = .1, batch_first = True)

#     def forward(self, base_state, actions):
#         traj_length = actions.shape[-2]

#         state = base_state.unsqueeze(0)
#         if len(actions.shape) == 2:
#             actions = actions.unsqueeze(1)
#             state = state.unsqueeze(1)
#         # pdb.set_trace()
#         # f1, _ = self.l1(actions, (state, state*0))
#         f1, _ = self.l1(actions, state)        
#         f2, _ = self.l2(f1)
#         f3, _ = self.l3(f2)
#         tiled_state = state.transpose(0,1).repeat(1, traj_length, 1)
#         # pdb.set_trace()
#         out = tiled_state + f3 # Add the starting postion memory location to the output

#         return out

class LSTMNet(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LSTMNet, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        h = 50
        self.h = h
        self.LSTM = False
        # self.l1 = torch.nn.GRU(action_dim+state_dim, state_dim, dropout = .1, batch_first = True)
        # self.dyn = torch.nn.Sequential(
        #         torch.nn.Linear(input_dim, h),
        #         torch.nn.SELU(),
        #         torch.nn.AlphaDropout(),
        #         torch.nn.Linear(h, h),
        #         torch.nn.SELU(),
        #         torch.nn.AlphaDropout(),
        #     )
        if self.LSTM:
            self.l1 = torch.nn.LSTM(input_dim, h, dropout = .1, batch_first = True, num_layers=2)
            self.l2 = torch.nn.LSTM(h, output_dim, dropout = .1, batch_first = True)
        else:
            self.l1 = torch.nn.GRU(input_dim, h, dropout = .1, batch_first = True, num_layers=2)
            self.l2 = torch.nn.GRU(h, output_dim, dropout = .1, batch_first = True)

    def forward(self, inpt, hidden = None):

        # state = base_state.unsqueeze(0)
        if len(inpt.shape) == 1:
            inpt = inpt.unsqueeze(0)
        if len(inpt.shape) == 2:
            inpt = inpt.unsqueeze(1)


        m1 = torch.zeros(2, inpt.shape[0] ,self.h)
        sd1 = torch.zeros(2, inpt.shape[0] ,self.h)

        m2 = torch.zeros(1, inpt.shape[0] ,self.output_dim)
        sd2 = torch.zeros(1, inpt.shape[0] ,self.output_dim)

        h1_distro = torch.distributions.normal.Normal(m1, sd1)
        h2_distro = torch.distributions.normal.Normal(m2, sd2)

        if hidden == None:
            if self.LSTM:
                hc1 = (h1_distro.sample(), h1_distro.sample())
                hc2 = (h2_distro.sample(), h2_distro.sample())
                hidden = (hc1, hc2)
            else:
                hidden = (h1_distro.sample(), h2_distro.sample())
            # pdb.set_trace()

        f1, h1 = self.l1(inpt, hidden[0])    
        # dyn = self.dyn(inpt)
        mid = f1
        # mid += dyn 
        f2, h2 = self.l2(mid, hidden[1])
        # else:
        #     f1, h1 = self.l1(inpt)     
        #     f2, h2 = self.l2(f1)
        out = f2.squeeze(1)# + inpt[...,:self.output_dim]# Add the starting postion memory location to the output

        return out, (h1,h2)

#         return out

# class LSTMNet(torch.nn.Module):
#     def __init__(self, action_dim, state_dim):
#         super(LSTMNet, self).__init__()
#         self.state_dim = state_dim
#         self.action_dim = action_dim
#         h = 100
#         # self.l1 = torch.nn.GRU(action_dim+state_dim, state_dim, dropout = .1, batch_first = True)
#         self.l1 = torch.nn.GRU(action_dim+state_dim, h, dropout = .1, batch_first = True, num_layers=2)
#         self.l2 = torch.nn.GRU(h, state_dim, dropout = .1, batch_first = True)

#     def forward(self, base_state, actions):
#         traj_length = actions.shape[-2]

#         state = base_state.unsqueeze(0)
#         if len(actions.shape) == 2:
#             actions = actions.unsqueeze(1)
#             state = state.unsqueeze(1)

#         tiled_state = state.transpose(0,1).repeat(1, traj_length, 1)
#         inpt = torch.cat((actions, tiled_state), -1)
#         # pdb.set_trace()

#         # f1, _ = self.l1(actions, (state, state*0))
#         f1, _ = self.l1(inpt)     
#         f2, _ = self.l2(f1)
#         # pdb.set_trace()
#         out = tiled_state + f2 # Add the starting postion memory location to the output

#         return out



def pt_build_model(nn_type, nodes, input_dim, output_dim, dropout_p=.1, both_nn_type='1'):
    if nn_type == '0':
        model = torch.nn.Sequential(
              torch.nn.Linear(input_dim, 32),
              torch.nn.SELU(),
              torch.nn.AlphaDropout(dropout_p),
              torch.nn.Linear(32, 32),
              torch.nn.SELU(),
              torch.nn.AlphaDropout(dropout_p),
              torch.nn.Linear(32, output_dim),
        )
    elif nn_type == '1':
        model = torch.nn.Sequential(
              torch.nn.Linear(input_dim, nodes),
              torch.nn.ReLU(),
              torch.nn.Dropout(dropout_p),
              torch.nn.Linear(nodes, nodes),
              torch.nn.ReLU(),
              torch.nn.Dropout(dropout_p),
              # torch.nn.SELU(),
              # torch.nn.AlphaDropout(dropout_p),
              torch.nn.Linear(nodes, output_dim),
        )
    elif nn_type == '2':
        return SplitModel(nodes, input_dim, output_dim, dropout_p, both_nn_type)

    elif nn_type == 'LSTM':
        # model = LSTMNet(2, output_dim)
        model = LSTMNet(input_dim, output_dim)


    # pdb.set_trace()
    return model




class LinearTransformedModel(torch.nn.Module):
    def __init__(self, old_model, input_dim, output_dim):
        super(LinearTransformedModel, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.model = old_model
        for param in self.model.parameters():
            param.requires_grad = False

        # self.lt = torch.nn.Linear(input_dim, input_dim)
        self.lt_inv = torch.nn.Linear(output_dim, output_dim)

        self.mat = torch.autograd.Variable(torch.tensor(np.identity(input_dim), dtype = dtype), requires_grad=True)

        self.b = np.diag(np.array([0.]))
        self.b = torch.autograd.Variable(torch.tensor(self.b, dtype = dtype), requires_grad=True)

        self.mat_inv = self.mat[:output_dim, :output_dim].transpose(0,1)
            # Initialize transform as identity matrix
    def forward(self, inpt):
        trans_in = torch.matmul(inpt,self.mat)
        out = self.model(trans_in)
        detrans_out = torch.matmul(out,self.mat_inv)

        return self.lt_inv(out)
        return detrans_out

    def get_consistency_loss(self, inpt):
        # trans_state = self.lt(inpt)[:self.output_dim]
        # detrans_state = self.lt_inv(trans_state)
        # mse_fn = torch.nn.MSELoss()
        # return mse_fn(detrans_state, inpt[:self.output_dim])


        trans_state = torch.mm(self.mat, inpt)[:self.output_dim]
        detrans_state = torch.mm(self.mat_inv, trans_state)
        mse_fn = torch.nn.MSELoss()
        return mse_fn(detrans_state, inpt[:self.output_dim])

class NonlinearTransformedModel(torch.nn.Module):
    def __init__(self, old_model, input_dim, output_dim, state_dim=4):
        super(NonlinearTransformedModel, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # self.model = old_model
        # for param in self.model.parameters():
        #     param.requires_grad = False

        self.model = LinearTransformedModel(old_model, input_dim, output_dim)

        h = 64
        h = 128
        self.transform_model = torch.nn.Sequential(
              # torch.nn.Linear(input_dim, 128),
              torch.nn.Linear(input_dim+output_dim, h),
              torch.nn.SELU(),
              torch.nn.AlphaDropout(.1),
              torch.nn.Linear(h, h),
              # torch.nn.SELU(),
              # torch.nn.AlphaDropout(.1),
              torch.nn.Tanh(),
              torch.nn.Dropout(.1),
        )
        self.A_model = torch.nn.Linear(h, output_dim**2)
        self.D_model = torch.nn.Linear(h, output_dim)
        self.gate = torch.nn.Sequential(
                torch.nn.Linear(h, output_dim),
                torch.nn.Sigmoid()
                )

        # self.rotation_matrix = np.diag(np.array([-1,-1,1,1]))
        self.rotation_matrix = np.diag(np.array([1,1,1,1]))
        self.iden = torch.autograd.Variable(torch.tensor(self.rotation_matrix, dtype = dtype))


            # Initialize transform as identity matrix
    def forward(self, inpt):
        out = self.model(inpt)#.detach()
        # return out
        new_inpt = torch.cat((inpt, out), dim=-1)
        feats = self.transform_model(new_inpt)

        D = self.D_model(feats)
        alpha = self.gate(feats)
        skip = False
        skip = True

        gradient_decay = .1

        if skip:
            # transformed_out = (1-alpha)*out + D*alpha
            transformed_out = alpha*out + D
        else:
            A = self.A_model(feats)*.05
            if len(feats.shape) == 1:
            	A = A.view(self.output_dim, self.output_dim)
            elif len(feats.shape) == 2:
            	A = A.view(-1, self.output_dim, self.output_dim)
            else: 
            	print("Unhandled shape")
            	pdb.set_trace()

            mat = A + self.iden

            if len(feats.shape) == 1:
                t_out = torch.matmul(mat, out)
            elif len(feats.shape) == 2:
                out2 = torch.unsqueeze(out, -1)
                t_out = torch.bmm(mat,out2).squeeze(-1)

            transformed_out = (1-alpha)*t_out + D*alpha
            # transformed_out = t_out + D
        # pdb.set_trace()
        return transformed_out

    def set_base_model_train(self, setting):
        for param in self.model.parameters():
            param.requires_grad = setting



class BNNWrapper(torch.nn.Module):
    def __init__(self, model, input_dim, output_dim):
        super(BNNWrapper, self).__init__()
        self.model = model
        self.input_dim = input_dim
        self.output_dim = output_dim

    def get_distro(self,x):
        # pdb.set_trace()
        output = self.model(x)
        means = output[...,: self.output_dim]
        stds = F.elu(output[..., self.output_dim:]) + 1
            #Make sure Standard Deviation > 0
        # pdb.set_trace()
        distro = torch.distributions.normal.Normal(means, stds)
        return distro

    def forward(self,x, true_state=None):
        distro = self.get_distro(x)
        interp = .1
        sample = distro.sample()*interp + (1-interp)*means
        if true_state is not None: 
            # sample = distro.mean
            log_p = distro.log_prob(true_state)
            nan_locs = (log_p != log_p) #Get locations where log_p is undefined
            if nan_locs.any():
                pdb.set_trace()
            log_p[nan_locs] = 0 #Set the loss in those locations to 0
            return sample, log_p
            
        # sample = distro.sample()

        return sample
        # return distro.mean


class StddevNet(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(StddevNet, self).__init__()
        self.mean_l1 = torch.nn.Linear(input_dim, 128)
        self.std_l1 = torch.nn.Linear(output_dim, 128)
        self.std_dev_model = torch.nn.Sequential(
              torch.nn.Linear(128, 128),
              torch.nn.SELU(),
              torch.nn.AlphaDropout(.1),
              torch.nn.Linear(128, output_dim),
        )
    def forward(self, x, std_devs = None):
        # pdb.set_trace()
        l1 = self.mean_l1(x)
        if type(std_devs) != type(None): l1 += self.std_l1(std_devs)
        feats = F.alpha_dropout(F.selu(l1), .1)
        return F.elu(self.std_dev_model(feats))+1


class DividedBNN(torch.nn.Module):
    def __init__(self, mean_model, input_dim, output_dim):
        super(DividedBNN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.mean_model = mean_model
        # for param in self.mean_model.parameters():
        #     param.requires_grad = False
        self.std_dev_model = StddevNet(input_dim, output_dim)
        self.sensor_noise = torch.autograd.Variable(torch.tensor(0.0, dtype=torch.float), requires_grad=True)


    def get_reading(self,x, std_devs=None):
        means = self.mean_model(x)
        out_stds = self.std_dev_model(x, std_devs)
        distro = torch.distributions.normal.Normal(means, out_stds)
        interp = .8
        sample = distro.sample()
        return sample, sample*interp + (1-interp)*means, out_stds

    def forward(self,x, std_devs=None, true_state=None):
        # pdb.set_trace()

        means = self.mean_model(x)
        out_stds = self.std_dev_model(x, std_devs)
        distro = torch.distributions.normal.Normal(means, out_stds)

        # interp = torch.sigmoid(self.sensor_noise)
        # interp = .01
        interp = .75
        sample = distro.sample()*interp + (1-interp)*means
        # sample = means
        
        if true_state is not None:
            log_p = distro.log_prob(true_state)
            nan_locs = (log_p != log_p) #Get locations where log_p is undefined
            if nan_locs.any():
                pdb.set_trace()
            log_p[nan_locs] = 0 #Set the loss in those locations to 0
            if type(std_devs) == type(None): return sample, log_p
            return sample, out_stds, log_p
            
        # sample = distro.sample()
        if type(std_devs) == type(None): return sample
        return sample, out_stds




