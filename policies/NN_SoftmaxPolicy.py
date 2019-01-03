import numpy as np
import torch as tr
import torch.nn as nn
from torch.distributions import Categorical

#######################################
# SoftmaxPolicy via NN
#######################################


class Policy:
    def __init__(self, env, hidden_dim: tuple=(100, 100),
                 activation: nn=nn.Tanh, lr: float=0.1):

        #   calling Super Class's constructor
        self.input_dim = env.obs_dim()
        self.output_dim = env.act_dim()
        self.hidden_dim = hidden_dim
        self.lr = lr

        #   create nn
        self.act = activation
        self.network = nn.Sequential()
        hidden_dim = self.input_dim
        i = 0
        for i, next_hidden_dim in enumerate(self.hidden_dim):
            self.network.add_module('linear' + i.__str__(),
                                    nn.Linear(hidden_dim, next_hidden_dim))
            self.network.add_module('activation' + i.__str__(), self.act)
            hidden_dim = next_hidden_dim
        self.network.add_module('linear' + (i+1).__str__(),
                                nn.Linear(hidden_dim, self.output_dim))

        #   set last layer weights and bias small
        for p in list(self.network.parameters())[-2:]:
            p.data *= 1e-2

        #   get net shape and size
        self.net_shapes = [p.data.numpy().shape
                           for p in self.network.parameters()]
        self.net_sizes = [p.data.numpy().size
                          for p in self.network.parameters()]

    def get_parameters(self):
        params = np.concatenate([p.contiguous().view(-1).data.numpy()
                                for p in self.network.parameters()])
        return params.copy()

    def set_parameters(self, new_param):
        current_idx = 0
        for idx, param in enumerate(self.network.parameters()):
            temp_param = \
                new_param[current_idx:current_idx + self.net_sizes[idx]]
            temp_param = temp_param.reshape(self.net_shapes[idx])
            param.data = tr.from_numpy(temp_param).float()
            current_idx += self.net_sizes[idx]
        return

    def get_action(self, state, greedy=False):
        prob = tr.softmax(self.network(tr.from_numpy(state).float()), dim=0)
        return Categorical(prob).sample().numpy().squeeze()

    def get_log_prob(self, states, actions):
        log_prob = tr.log_softmax(self.network(tr.from_numpy(states).float()),
                                  dim=1)
        return log_prob.gather(1, tr.from_numpy(actions).long())

    def get_kl(self, states):
        prob1 = tr.softmax(self.network(tr.from_numpy(states).float()), dim=1)
        prob0 = prob1.detach()
        kl = prob0 * (tr.log(prob0) - tr.log(prob1))
        return kl.sum(1, keepdim=True)

