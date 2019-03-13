"""Module containing the baseline class used in the natural policy
gradient for increased performance as well as a neural network class
realising a PyTorch network as estimator for the baseline.

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


class Baseline:
    """The baseline class represents an estimator for estimating the
    value function in order to improve the algorithm performance by
    reducing the variance of the gradient estimator.

    Jan Peters and Stefan Schaal, Reinforcement learning of motor skills
    with policy gradients, Journal of the International Neural Network
    Society, European Neural Network Society & Japanese Neural Network
    Society, 21, 682-697 (2008)

    The baseline is realized by a neural network of variable size. To
    improve convergence of the neural network a batch normalization is
    used.

    Methods
    -------
    train(trajectories)
        Updates the network parameters based on new trajectory data

    predict(trajectories)
        Predicts a return value for each transition in the trajectory
    """

    def __init__(self, env, hidden_dim: tuple = (128, 128),
                 activation: nn.Module = nn.Tanh(), batch_size: int = 64,
                 epochs: int = 10, lr: float = 1e-3):
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

        :param batch_size: Size of batches used for batch normalization
        :type batch_size: int

        :param epochs: Amount of epochs learned per update
        :type epochs: int

        :param lr: Learning rate for the optimizer
        :type lr: float
        """

        self.__hidden_dim = hidden_dim
        self.__batch_size = batch_size
        self.__epochs = epochs

        # create nn
        self.__network = Network(env.obs_dim(), 1,
                                 self.__hidden_dim, activation)

        # Create Loss function and Adam Optimizer
        self.__loss_fct = nn.MSELoss()
        self.__optimizer = tr.optim.Adam(self.network.parameters(), lr=lr)

    # getter only properties
    @property
    def network(self):
        """Returns the policy network.

        :return the neural network
        :rtype: Network
        """
        return self.__network

    @property
    def hidden_dim(self):
        """Returns the dimensions for the hidden layers of the neural
        network.

        :return: Dimensions of hidden layers
        :rtype: tuple
        """

        return self.__hidden_dim

    @property
    def epochs(self):
        """Returns the number of epochs run during each update.

        :return: Number of epochs
        :rtype: int
        """

        return self.__epochs

    # Utility Functions
    # ===============================================================
    @staticmethod
    def __get_data(trajectories):
        if isinstance(trajectories, list):
            obs = np.concatenate([t["observations"]
                                  for t in trajectories])
            rew = np.concatenate([t["rewards"]
                                  for t in trajectories]).reshape(-1, 1)
            if "values" in trajectories:
                val = np.concatenate([t["values"]
                                      for t in trajectories]).reshape(-1, 1)
            else:
                val = np.zeros_like(rew).reshape(-1, 1)
        else:
            obs = trajectories["observations"]
            rew = trajectories["rewards"].reshape(-1, 1)
            if "values" in trajectories:
                val = trajectories["values"].reshape(-1, 1)
            else:
                val = np.zeros_like(rew).reshape(-1, 1)

        return obs, val

    # Main Functions
    # ===============================================================
    def train(self, trajectories):
        """This function updates the neural network parameters based on
        new trajectories given as input using the adam optimizer.

        Diederik P. Kingma and Jimmy Ba, Adam: A Method for Stochastic
        Optimization, 3rd International Conference for Learning
        Representations (2015)

        :param trajectories: Contains a set of trajectories each being a
        dictionary with information about every transition performed in
        the trajectory simulation
        :type trajectories: list of dict
        """

        data, values = self.__get_data(trajectories)

        for e in range(self.__epochs):
            permuted_idx = np.random.permutation(len(values))
            for batch in range(int(len(values)/self.__batch_size)-1):
                idx = tr.LongTensor(permuted_idx[batch*64:(batch+1)*64])
                inputs = tr.from_numpy(data).float()[idx]
                labels = tr.from_numpy(values).float()[idx]

                self.__optimizer.zero_grad()
                predicted = self.network(inputs)
                loss = self.__loss_fct(predicted, labels)

                # back propagation
                loss.backward()

                # update parameters
                self.__optimizer.step()

    def predict(self, trajectories):
        """Predicts a return value for each transition in the trajectory

        :param trajectories: Contains a set of trajectories each being a
        dictionary with information about every transition performed in
        the trajectory simulation
        :type trajectories: list of dict

        :return: Predictions
        :rtype: array of float
        """
        obs, _ = self.__get_data(trajectories)
        return self.network(tr.from_numpy(obs).float())\
            .detach().numpy().squeeze()


class Network(nn.Module):
    """Neural Network class realising the neural network for the
    baseline.

    Methods
    ---------
    forward(x)
        Calculates network output for input x
    """

    def __init__(self, input_dim: int = 1, output_dim: int = 1,
                 hidden_dim: tuple = (128, 128),
                 activation: nn.Module = nn.Tanh()):
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
            self.__net.add_module('Batch' + i.__str__(),
                                  nn.BatchNorm1d(next_dim))
            self.__net.add_module('activation' + i.__str__(), activation)
            curr_dim = next_dim
        self.__net.add_module('linear' + (i + 1).__str__(),
                              nn.Linear(curr_dim, output_dim))

        # set weights and bias in last layer small for fast convergence
        for p in list(self.__net.parameters())[-2:]:
            p.data *= 1e-2

    def forward(self, x):
        """Function returning the neural network output for a given
        input x.

        :param x: Represents the network input of size input dim
            (env.obs_dim)
        :type x: array_like

        :return: Output of the neural network
        :rtype: float
        """

        value = self.__net(x)
        return value
