"""

"""

import numpy as np
import torch as tr
import torch.nn as nn


class Policy:

    # Init
    # ===============================================================
    def __init__(self, env, hidden_dim: tuple=(64, 64),
                 activation: nn=nn.Tanh, log_std=None):

        """ init """
        self.input_dim = env.obs_dim()
        self.output_dim = env.act_dim()
        self.hidden_dim = hidden_dim
        self.act = activation
        self.log_std = tr.from_numpy(np.log(env.act_high/2)).float()\
            if log_std is None else log_std

        # create nn
        self.network = Network(self.input_dim, self.output_dim,
                               self.hidden_dim, self.act, self.log_std)

        # get net shape and size
        self.net_shapes = [p.data.numpy().shape
                           for p in self.network.parameters()]
        self.net_sizes = [p.data.numpy().size
                          for p in self.network.parameters()]

        self.length = \
            len(np.concatenate([p.contiguous().view(-1).data.numpy()
                                for p in self.network.parameters()]))

    # Utility Functions
    # ===============================================================
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

    # Main Functions
    # ===============================================================
    def get_action(self, state, greedy=False):
        """

        :param state:
        :param greedy:
        :return:
        """
        mean, log_std = self.network.forward(
            tr.from_numpy(state.reshape(1, -1)).float())
        if greedy:
            return mean.detach().numpy().squeeze()
        else:
            std = tr.exp(log_std).detach().numpy().squeeze()
            noise = std * np.random.randn(self.output_dim)
            return mean.detach().numpy().squeeze() + noise

    def get_log_prob(self, states, actions):
        """

        :param states:
        :param actions:
        :return:
        """
        mean, log_std = self.network.forward(tr.from_numpy(states).float())

        actions = tr.from_numpy(actions).float()
        log_prob = - (actions - mean) ** 2
        log_prob /= (2.0 * tr.exp(log_std) ** 2 + 1e-10)
        log_prob -= log_std + 0.5 * self.output_dim * np.log(2 * np.pi)
        return log_prob.sum(1, keepdim=True)

    def get_kl(self, states):
        """

        :param states:
        :return:
        """
        mean, log_std = self.network.forward(tr.from_numpy(states).float())
        std = tr.exp(log_std)

        fixed_mean = mean.detach()
        fixed_log_std = log_std.detach()
        fixed_std = std.detach()

        kl = (fixed_std ** 2 + (fixed_mean - mean) ** 2)
        kl /= (2.0 * std ** 2 + 1e-10)
        kl += log_std - fixed_log_std - 0.5
        return kl.sum(1, keepdim=True)


class Network(nn.Module):
    """Neural Network class realising the neural network for the
    baseline.

    Methods
    ---------
    forward(x)
        Calculates network output for input x
    """

    def __init__(self, input_dim: int = 1, output_dim: int = 1,
                 hidden_dim: tuple = (128, 128), activation: nn = nn.Tanh,
                 log_std=0):
        """
        :param input_dim: Input dimension of the neural network
        :type input_dim: int

        :param output_dim: Output dimension of the neural network
            (= env.obs_dim)
        :type output_dim: int

        :param hidden_dim: Dimensions for each hidden layer in the
            neural network
        :type hidden_dim: tuple

        :param activation: Activation function for each node in the
            neural network
        :type activation: function

        :param log_std: Log of standard deviation used for exploration
        :type log_std: float
        """

        super(Network, self).__init__()
        self.__net = nn.Sequential()

        # create NN
        curr_dim = input_dim
        i = 0
        for i, next_dim in enumerate(hidden_dim):
            self.__net.add_module('linear' + i.__str__(),
                                  nn.Linear(curr_dim,
                                            next_dim))
            self.__net.add_module('activation' + i.__str__(),
                                  activation())
            curr_dim = next_dim
        self.__net.add_module('linear' + (i + 1).__str__(),
                              nn.Linear(curr_dim, output_dim))

        # set weights and bias in last layer small for fast convergence
        for p in list(self.__net.parameters())[-2:]:
            p.data *= 1e-2

        # set log_std
        self.log_std = nn.Parameter(tr.ones(1, output_dim) * log_std)

    def forward(self, x):
        """Function returning the neural network output for a given
        input x.

        :param x: Represents the network input of size input dim
            (env.obs_dim)
        :type x: array_like

        :return: Output of the neural network
        :rtype: array of float, array of float
        """

        mean = self.__net(x)
        log_std = self.log_std.expand_as(mean)
        return mean, log_std
