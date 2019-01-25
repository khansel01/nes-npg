import numpy as np
import torch as tr
import torch.nn as nn

#######################################
# NES
#######################################


#TODO comments
class NES:
    def __init__(self, env, eta_sigma, eta_mu, population_size=5,
                 sigma_lower_bound=1e-10, max_iter=100):

        self.__population_size = population_size
        self.__eta_sigma = eta_sigma
        self.__eta_mu = eta_mu
        self.env = env
        self.max_iter = max_iter
        self.sigma_lower_bound = sigma_lower_bound

    def do(self, f, mu, sigma, policy):
        stop = False
        generation = 0

        # random number generator for drawing samples z_k
        sampler = np.random.RandomState()

        rewards = np.array([])
        stds = np.array([])

        while not stop:
            # draw samples
            s = sampler.normal(0, 1, (self.__population_size, len(mu)))

            z = mu + sigma * s

            # evaluate fitness
            fitness, g = f(policy, self.env, z, 1)
            # fitness, g = self.env.roll_out(policy, z)


            # compute utilities
            s_sorted, u = self.__utility(s, fitness)
            # s_sorted, u = s, fitness

            # compute gradients
            j_mu = u @ s_sorted
            j_sigma = u @ (s_sorted**2 - 1)

            # update parameters
            mu += self.__eta_mu * sigma * j_mu
            sigma *= np.exp(self.__eta_sigma / 2 * j_sigma)

            # sigma has to be positive
            if np.any(sigma < self.sigma_lower_bound):
                sigma[sigma < self.sigma_lower_bound] = self.sigma_lower_bound

            print(generation, np.mean(g), max(g), sigma, mu)
            rewards = np.append(rewards, np.mean(g))
            stds = np.append(stds, np.std(g))

            generation += 1

            # until stopping criterion is met
            stop = generation >= self.max_iter

        return mu, sigma, rewards, stds

    def __utility(self, s, f):
        indices = np.argsort(f, kind="mergesort")
        # descending
        s_sorted = s[indices[::-1]]
        log_half = np.log(0.5 * self.__population_size + 1)
        log_k = np.log(np.arange(1, self.__population_size + 1))
        numerator = np.maximum(0, log_half - log_k)
        u = numerator / np.sum(numerator) - 1 / self.__population_size

        return s_sorted, u

    # -------------------------------------------------------------------------
    # Policy


class PolicyNN:

    """ Init """
    """==============================================================="""
    def __init__(self, env, hidden_dim: tuple = (64, 64),
                 activation: nn = nn.Tanh):

        """ init """
        self.input_dim = env.obs_dim()
        self.output_dim = env.act_dim()
        self.hidden_dim = hidden_dim
        self.act = activation

        """ create nn """
        self.network = Network(self.input_dim, self.output_dim,
                               self.hidden_dim, self.act)

        """ get net shape and size """
        self.net_shapes = [p.data.numpy().shape
                           for p in self.network.parameters()]
        self.net_sizes = [p.data.numpy().size
                          for p in self.network.parameters()]

        self.length = \
            len(np.concatenate([p.contiguous().view(-1).data.numpy()
                                for p in self.network.parameters()]))

    """ Utility Functions """
    """==============================================================="""
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

    """ Main Functions """
    """==============================================================="""
    def get_action(self, state, greedy=False):
        action = self.network.forward(tr.from_numpy(state).float())
        return action.detach().numpy().squeeze()


class PolicySquare:

    """ Init """
    """==============================================================="""

    def __init__(self, env):
        """ init """
        self.input_dim = env.obs_dim()
        self.output_dim = env.act_dim()

        self.length = self.input_dim ** 2
        self.__params = np.zeros(self.length)

    """ Utility Functions """
    """==============================================================="""

    def get_parameters(self):
        return self.__params

    def set_parameters(self, new_param):
        self.__params = new_param

    """ Main Functions """
    """==============================================================="""

    def get_action(self, state):
        o = np.reshape(state, (-1, 1))
        x = np.reshape(o * np.transpose(o), (1, -1)) @ self.__params
        return [np.sum(x)]


class Network(nn.Module):
    def __init__(self, input_dim: int = 1, output_dim: int = 1,
                 hidden_dim: tuple = (128, 128), activation: nn = nn.Tanh):

        """ init """
        super(Network, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.act = activation()
        self.net = nn.Sequential()

        """ create NN """
        hidden_dim = self.input_dim
        i = 0
        for i, next_hidden_dim in enumerate(self.hidden_dim):
            self.net.add_module('linear' + i.__str__(),
                                nn.Linear(hidden_dim,
                                          next_hidden_dim))
            # self.net.add_module('Batch' + i.__str__(),
            #                     nn.BatchNorm1d(next_hidden_dim))
            self.net.add_module('activation' + i.__str__(), self.act)
            hidden_dim = next_hidden_dim
        self.net.add_module('linear' + (i + 1).__str__(),
                            nn.Linear(hidden_dim, self.output_dim))

        """ set last layer weights and bias small """
        for p in list(self.net.parameters())[-2:]:
            p.data *= 1e-2

    def forward(self, x):
        action = self.net(x)
        return action