"""Module contains two classes:
    Policy: Uses a neural network to represent a gaussian policy.
    Network: Implement a neural network with PyTorch
Together these classes are a neural network based gaussian policy for
mapping the environment states to actions.

:Date: 2019-03-11
:Version: 1
:Authors:
    - Janosch Moos
    - Kay Hansel
    - Cedric Derstroff
"""

import numpy as np
import torch as tr
import torch.nn as nn


class Policy:
    """The policy class implements a neural network based on PyTorch to
    represent a gaussian distribution. Hence the output of the NN is the
    mean and logarithmic standard deviation of the distribution.
    The policy can be used explorative as well as greedy. In the greedy
    case the standard deviation is not used.

    Methods
    -------
    get_parameters()
        Returns the current policy network parameters

    set_parameters(new_params)
        Overrides the network parameters

    get_hidden_dim()
        Returns the dimensions of the hidden layers

    get_action(state, greedy=False)
        Determines the action to take at a certain state

    get_log_prob(states, actions)
        Calculates the logarithmic probabilities of given actions in
        corresponding states

    get_kl(states)
        Calculates the KL-divergence between the new and old policy
    """

    def __init__(self, env, hidden_dim: tuple = (64, 64),
                 activation: nn.Module = nn.Tanh(), log_std=None):
        """
        :param env: Contains the gym environment the simulations are
            performed on
        :type env: Environment

        :param hidden_dim: Dimensions for each hidden layer in the
            neural network
        :type hidden_dim: tuple

        :param activation: Activation function for each node in the
            neural network
        :type activation: nn.Module

        :param log_std: Log of standard deviation used for exploration
        :type log_std: array_like
        """

        self.__output_dim = env.act_dim()
        self.__hidden_dim = hidden_dim

        if log_std is None:
            log_std = tr.from_numpy(np.log(env.act_high/2)).float()

        # create nn
        self.__network = Network(env.obs_dim(), self.__output_dim,
                                 self.__hidden_dim, activation, log_std)

        # get net shape and size
        self.__net_shapes = [p.data.numpy().shape
                             for p in self.__network.parameters()]
        self.__net_sizes = [p.data.numpy().size
                            for p in self.__network.parameters()]

        self.__length = \
            len(np.concatenate([p.contiguous().view(-1).data.numpy()
                                for p in self.__network.parameters()]))

    # getter only properties
    @property
    def length(self):
        """Returns the number of parameters in the policy network.

        :return: the number of policy parameters
        :rtype: int
        """
        return self.__length

    @property
    def network(self):
        """Returns the policy network.

        :return: the policy network
        :rtype: Network
        """
        return self.__network

    # Utility Functions
    # ===============================================================
    def get_parameters(self):
        """Returns the current network parameters (weights).

        :return: Copy of the network parameters
        :rtype: array_like
        """
        params = np.concatenate([p.contiguous().view(-1).data.numpy()
                                 for p in self.__network.parameters()])
        return params.copy()

    def set_parameters(self, new_param):
        """Overrides the network parameters (weights) with a new set of
        parameters.

        :param new_param: A new set of parameters. Needs to be of
            correct size for the neural networks dimensions.
        :type new_param: array_like
        """

        current_idx = 0
        for idx, param in enumerate(self.__network.parameters()):
            temp_param = \
                new_param[current_idx:current_idx + self.__net_sizes[idx]]
            temp_param = temp_param.reshape(self.__net_shapes[idx])
            param.data = tr.from_numpy(temp_param).float()
            current_idx += self.__net_sizes[idx]

    @property
    def hidden_dim(self):
        """Returns the dimensions for the hidden layers of the neural
        network.

        :return: Dimensions of hidden layers
        :rtype: tuple
        """

        return self.__hidden_dim

    # Main Functions
    # ===============================================================
    def get_action(self, state, greedy=False):
        """Evaluates the state input and calculates the best possible
        action. For greedy set to false the returned action will be
        sampled from a gaussian distribution with greedy action as mean
        and the exponential of log_std as standard deviation.

        :param state: Represents the current observation from the
            environment
        :type state: array_like

        :param greedy: Determines whether the action will be evaluated
            greedy or explorative
        :type greedy: bool

        :return: Action to take
        :rtype: array_like
        """
        mean, log_std = self.__network.forward(
            tr.from_numpy(state.reshape(1, -1)).float())
        if greedy:
            return mean.detach().numpy().squeeze()
        else:
            std = tr.exp(log_std).detach().numpy().squeeze()
            noise = std * np.random.randn(self.__output_dim)
            return mean.detach().numpy().squeeze() + noise

    def get_log_prob(self, states, actions):
        """Calculates the logarithmic probability of an action after
        a gaussian distribution with greedy action as mean and the
        exponential of log_std as standard deviation. The mean is
        calculated using the given state.

        :param states: Represents the observations from the
            environment (can be multiple states)
        :type states: array_like

        :param actions: Chosen actions the logarithmic probability
            should be calculated for (can be multiple actions)
        :type actions: array_like

        :return: Logarithmic probabilities of the given actions
        :rtype: array_like
        """

        mean, log_std = self.__network.forward(tr.from_numpy(states).float())

        actions = tr.from_numpy(actions).float()
        log_prob = - (actions - mean) ** 2
        log_prob /= (2.0 * tr.exp(log_std) ** 2 + 1e-10)
        log_prob -= log_std + 0.5 * self.__output_dim * np.log(2 * np.pi)
        return log_prob.sum(1, keepdim=True)

    def get_kl(self, states):
        """Calculates the Kullback-Leibler divergence between the old
        and new policy in order to fulfill the constraint of the natural
        gradient optimization problem.

        :param states: Represents the observations from the
            environment (can be multiple states)
        :type states: array_like

        :return: KL-divergence
        :rtype: array_like
        """

        mean, log_std = self.__network.forward(tr.from_numpy(states).float())
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

     Attributes
    ----------
    log_std
        Log of standard deviation used for exploration

    Methods
    ---------
    forward(x)
        Calculates network output for input x
    """

    def __init__(self, input_dim: int = 1, output_dim: int = 1,
                 hidden_dim: tuple = (128, 128),
                 activation: nn.Module = nn.Tanh(), log_std=0):
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
        :type activation: nn.Module

        :param log_std: Log of standard deviation used for exploration
        :type log_std: array_like
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
                                  activation)
            curr_dim = next_dim
        self.__net.add_module('linear' + (i + 1).__str__(),
                              nn.Linear(curr_dim, output_dim))

        # set weights and bias in last layer small for fast convergence
        for p in list(self.__net.parameters())[-2:]:
            p.data *= 1e-2

        # set log_std
        self.__log_std = nn.Parameter(tr.ones(1, output_dim) * log_std)

    @property
    def log_std(self):
        """Returns the logarithm of the standard deviation which is used for
        the exploration

        :return: the log of standard deviation used for exploration
        """
        return self.__log_std

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
        log_std = self.__log_std.expand_as(mean)
        return mean, log_std
