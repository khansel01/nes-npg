import numpy as np
import torch as tr
import torch.nn as nn
from models.NN_GaussianPolicy import Policy

#######################################
# NES
#######################################


# TODO comments
class NES:
    def __init__(self, env, policy, eta_sigma=None,
                 eta_mu=None, population_size=None,
                 sigma_lower_bound=1e-10, episodes=100, hidden_dim=(8,)):

        if policy == 'nn':
            self.__policy = PolicyNN(env, hidden_dim=hidden_dim)
        elif policy == 'gaussian':
            self.__policy = Policy(env, hidden_dim=hidden_dim)
        elif policy == 'square':
            self.__policy = PolicySquare(env)
        else:
            self.__policy = PolicySquare(env)

        # pre calculate value fro performance
        log_d = np.log(self.__policy.length)

        if population_size is not None:
            self.__population_size = population_size
        else:
            self.__population_size = 4 + int(3 * log_d)

        if eta_sigma is not None:
            self.__eta_sigma = eta_sigma
        else:
            self.__eta_sigma = (3 + log_d) / np.sqrt(self.__policy.length) / 5

        self.__eta_mu = eta_mu if eta_mu is not None else 1

        self.__env = env
        self.__episodes = episodes
        self.__sigma_lower_bound = sigma_lower_bound

        # utility is always equal hence we can pre compute it here
        log_half = np.log(0.5 * self.__population_size + 1)
        log_k = np.log(np.arange(1, self.__population_size + 1))
        numerator = np.maximum(0, log_half - log_k)
        self.__u = numerator / np.sum(numerator) - 1 / self.__population_size

    def do(self, seed=None, sigma_init=1.0):

        if seed is not None:
            tr.random.manual_seed(seed)
            np.random.seed(seed)

        stop = False
        generation = 0
        mu = np.zeros(self.__policy.length)

        if sigma_init <= self.__sigma_lower_bound:
            sigma_init = self.__sigma_lower_bound

        sigma = np.ones(self.__policy.length) * sigma_init

        # random number generator for drawing samples z_k
        sampler = np.random.RandomState(seed)

        means = np.array([])
        stds = np.array([])

        u_eta_sigma_half = 0.5 * self.__eta_sigma * self.__u
        u_eta_mu = self.__eta_mu * self.__u

        while not stop:
            # draw samples
            s = sampler.normal(0, 1, (self.__population_size, len(mu)))

            z = mu + sigma * s

            # evaluate fitness
            fitness, g = self.f_norm(self.__policy, self.__env, z)
            # fitness, g = self.f(z)

            # compute utilities
            s_sorted = s[np.argsort(fitness, kind="mergesort")[::-1]]

            # # compute gradients
            # j_mu = self.__u @ s_sorted
            # j_sigma = self.__u @ (s_sorted ** 2 - 1)
            #
            # # update parameters
            # mu += self.__eta_mu * sigma * j_mu
            # sigma *= np.exp(self.__eta_sigma / 2 * j_sigma)

            # use pre computed values
            mu += sigma * (u_eta_mu @ s_sorted)
            sigma *= np.exp(u_eta_sigma_half @ (s_sorted ** 2 - 1))

            # sigma has to be positive
            sigma[sigma < self.__sigma_lower_bound] = self.__sigma_lower_bound

            # safe values for learning curve
            m = np.mean(fitness)

            means = np.append(means, m)
            stds = np.append(stds, np.std(fitness))

            print(generation, m, max(fitness),
                  np.mean(g), max(g), max(sigma))

            generation += 1

            # until stopping criterion is met
            stop = generation >= self.__episodes

        self.__policy.set_parameters(mu)

        return self.__policy, sigma, means, stds

    # fitness functions
    def f(self, w, n_roll_outs: int = 1):

        samples = np.size(w, 0)
        f = np.zeros(samples)
        steps = np.zeros(samples)

        seed = self.__env.get_seed()

        for s in range(samples):
            self.__policy.set_parameters(w[s])

            self.__env.seed(seed)

            trajectories: dict = self.__env.roll_out(self.__policy,
                                                     n_roll_outs=n_roll_outs)

            rewards = np.concatenate([t["rewards"]
                                      for t in trajectories]).reshape(-1, 1)

            steps[s] = np.array(
                [t["steps"] for t in trajectories]).sum() / n_roll_outs

            f[s] = rewards.sum() / n_roll_outs

        return f, steps

    @staticmethod
    def f_norm(policy, env, w, n_roll_outs: int = 1):

        samples = np.size(w, 0)
        f = np.zeros(samples)
        steps = np.zeros(samples)

        seed = np.random.randint(2**32 - 1)

        for s in range(samples):

            policy.set_parameters(w[s])
            env.seed(seed)

            NES.roll_out(policy, env, f, steps, s, n_roll_outs)

        return f, steps

    @staticmethod
    def roll_out(policy, env, f, steps, s, n_roll_outs):

        rewards = 0
        t = 0

        for i in range(n_roll_outs):

            done = False
            obs = env.reset()

            while not done:
                a = policy.get_action(obs, greedy=True)
                obs, r, done, _ = env.step(a)
                rewards += r
                t += 1

        f[s] = rewards / n_roll_outs
        steps[s] = t / n_roll_outs

    # ------------------------------------------------------------------------
    # Policy


class PolicyNN:
    """ Init """
    """==============================================================="""

    def __init__(self, env, hidden_dim: tuple = (64, 64),
                 activation: nn = nn.Tanh):
        """ init """
        self.__input_dim = env.obs_dim()
        self.__output_dim = env.act_dim()
        self.hidden_dim = hidden_dim
        self.act = activation

        """ create nn """
        self.network = Network(self.__input_dim, self.__output_dim,
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

    def get_action(self, state, greedy=True):
        return self.network.forward(tr.from_numpy(state).float()
                                    ).detach().numpy().squeeze().reshape(-1)


class PolicySquare:
    """ Init """
    """==============================================================="""

    def __init__(self, env):
        """ init """
        self.__input_dim = env.obs_dim() + 1
        self.__output_dim = env.act_dim()

        self.__indices = np.tril_indices(self.__input_dim)

        self.__params = np.zeros((int(self.__input_dim
                                      * (self.__input_dim + 1) / 2),
                                  self.__output_dim))

        self.length = np.size(self.__params)

    """ Utility Functions """
    """==============================================================="""

    def get_parameters(self):
        return self.__params.reshape(1)

    def set_parameters(self, new_param):
        self.__params = new_param.reshape(-1, self.__output_dim)

    """ Main Functions """
    """==============================================================="""

    def get_action(self, state, greedy=True):
        o = np.reshape(np.append(state, 1), (-1, 1))
        x = (o * o.transpose())[self.__indices] @ self.__params
        return np.array(np.sum(x)).reshape(-1)


class Network(nn.Module):
    def __init__(self, input_dim: int = 1, output_dim: int = 1,
                 hidden_dim: tuple = (128, 128), activation: nn = nn.Tanh):

        """ init """
        super(Network, self).__init__()
        self.__input_dim = input_dim
        self.__output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.act = activation()
        self.net = nn.Sequential()

        """ create NN """
        hidden_dim = self.__input_dim
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
                            nn.Linear(hidden_dim, self.__output_dim))

        """ set last layer weights and bias small """
        for p in list(self.net.parameters())[-2:]:
            p.data *= 1e-2

    def forward(self, x):
        action = self.net(x)
        return action
