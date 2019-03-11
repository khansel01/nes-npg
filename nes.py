"""Module containing the core class of the Natural Evolution Strategies
"""

import numpy as np


class NES:
    """Core class of the NES algorithm. Contains all relevant Parameters except
    for the policy. Important functions are "do" to run a single training step
    as well as "fitness" for evaluating samples during training.
    """

    def __init__(self, n_parameters, eta_sigma=None,
                 eta_mu=None, population_size=None,
                 sigma_lower_bound=1e-10, sigma_init=1.0):

        self.normalizer = None

        # pre calculate value for performance
        log_d = np.log(n_parameters)

        # calculate population size if not specified
        if population_size is not None:
            self.__population_size = population_size
        else:
            self.__population_size = 4 + int(3 * log_d)

        # calculate eta_sigma if not specified
        if eta_sigma is not None:
            self.__eta_sigma = eta_sigma
        else:
            self.__eta_sigma = (3 + log_d) / np.sqrt(n_parameters) / 5

        # set eta_mu
        self.__eta_mu = eta_mu if eta_mu is not None else 1

        # define lower bound for sigma to avoid artifacts in calculations
        self.__sigma_lower_bound = sigma_lower_bound

        # utility is always equal hence we can pre compute it here
        log_half = np.log(0.5 * self.__population_size + 1)
        log_k = np.log(np.arange(1, self.__population_size + 1))
        numerator = np.maximum(0, log_half - log_k)
        self.__u = numerator / np.sum(numerator) - 1 / self.__population_size

        self.__mu = np.zeros(n_parameters)

        # define sigma
        if sigma_init <= self.__sigma_lower_bound:
            sigma_init = self.__sigma_lower_bound

        self.__sigma = np.ones(n_parameters) * sigma_init
        self.__sigma_init = sigma_init

        # random number generator for drawing samples z_k
        seed: int = int(np.random.rand() * 2**32 - 1)
        self.__sampler = np.random.RandomState(seed)

        self.__u_eta_sigma_half = 0.5 * self.__eta_sigma * self.__u
        self.__u_eta_mu = self.__eta_mu * self.__u

    # Main Functions
    # ===============================================================
    def do(self, env, policy, n_roll_outs):
        """Runs a single training step:
        1. draw a set of parameter samples
        2. Gets an evaluation (fitness) for all samples using n simulations
        (roll-outs) for each sample on the environment
        3. Updates parameters based on samples sorted by their fitness
        """

        self.__mu = policy.get_parameters()

        # draw samples from search distribution
        s = self.__sampler.normal(0, 1, (self.__population_size,
                                         len(self.__mu)))

        z = self.__mu + self.__sigma * s

        # evaluate fitness
        fitness, steps = self.fitness(policy, env, z, n_roll_outs)

        # sort samples according to fitness
        s_sorted = s[np.argsort(fitness, kind="mergesort")[::-1]]

        # update parameters
        self.__mu += self.__sigma * (self.__u_eta_mu @ s_sorted)
        self.__sigma *= np.exp(self.__u_eta_sigma_half
                               @ (s_sorted ** 2 - 1))

        # sigma has to be positive
        self.__sigma[self.__sigma < self.__sigma_lower_bound] =\
            self.__sigma_lower_bound

        policy.set_parameters(self.__mu)

        return fitness, steps

    # Utility Functions
    # ===============================================================
    @staticmethod
    def fitness(policy, env, w, n_roll_outs: int = 1):
        """ Evaluates the fitness of each sample in a set of samples. This is
        done by running a simulation on the environment using the same seed
        for all trials.
        """

        samples = np.size(w, 0)
        f = np.zeros(samples)
        steps = np.zeros(samples)

        # define seed to be the same for each sample
        # numpy is used to get deterministic outcomes during testing
        # seed = env.get_seed()
        seed = np.random.randint(0, 90000)

        for s in range(samples):
            # set sample as policy parameters
            policy.set_parameters(w[s])

            env.seed(seed)

            trajectories: dict = env.roll_out(policy, n_roll_outs=n_roll_outs,
                                              greedy=True)

            t_steps = []
            t_reward = []
            for t in trajectories:
                t_steps.append(t["time_steps"])
                t_reward.append(t["total_reward"])

            steps[s] = np.sum(t_steps) / n_roll_outs
            f[s] = np.sum(t_reward) / n_roll_outs

        return f, steps

    def get_title(self):
        """Generates a title for plotting results containing all relevant
        parameters and the algorithm name
        """

        return r"NES $\lambda = {}, "  \
               r"\sigma_0 = {}, " \
               r"\eta_\sigma = {:.4f}, " \
               r"\eta_\mu = {}$".format(self.__population_size,
                                        self.__sigma_init,
                                        self.__eta_sigma,
                                        self.__eta_mu)

    @staticmethod
    def get_name():
        """returns algorithm name"""

        return 'NES'
