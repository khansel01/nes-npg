"""Module containing the core class of the Natural Evolution Strategies

:Date: 2019-03-11
:Version: 1
:Authors:
    - Cedric Derstroff
    - Janosch Moos
    - Kay Hansel
"""

from utilities.estimations import *


class NES:
    """Core class of the NES algorithm. Contains all relevant Parameters
    except for the policy. Important functions are "do" to run a single
    training step as well as "fitness" for evaluating samples during
    training.

    This class implements the Separable NES (SNES).

    Wierstra, D., Schaul, T., Glasmachers, T., Sun, Y., Peters, J.,
    Schmidhuber, J.: NaturalEvolution Strategies.  Journal of
    Machine Learning Research 15, 949–980 (2014).
    URL http://jmlr.org/papers/v15/wierstra14a.html

    Attributes
    -----------
    normalizer: Normalizer
        None

    Methods
    -----------
    do(env, policy, n_roll_outs)
        Runs a single training step:
        1. draw a set of parameter samples
        2. Gets an evaluation (fitness) for all samples using n
        simulations (roll-outs) for each sample on the environment
        3. Updates parameters based on samples sorted by their fitness
    """

    def __init__(self, n_parameters: int, eta_sigma: float = None,
                 eta_mu: float = None, population_size: int = None,
                 sigma_lower_bound: float = 1e-10, sigma_init: float = 1.0):
        """
        :param n_parameters: Number of parameters of the policy which
            will be trained
        :type n_parameters: int

        :param eta_sigma: Learning rate for training the variance
        :type eta_sigma: float

        :param eta_mu: Learning rate for training the parameters
        :type eta_mu: float

        :param population_size: Number of sampled policy parameters for
            learning
        :type population_size: int

        :param sigma_lower_bound: The lower bound of the variance,
            i.e. the lowest value the variance everything below will be
            set to this value
        :type sigma_lower_bound: float

        :param sigma_init: Initial value of the variance
        :type sigma_init: float
        """

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

        # define lower bound for sigma to avoid artifacts in
        # calculations
        self.__sigma_lower_bound = sigma_lower_bound

        # utility is always equal hence we can pre compute it here
        log_half = np.log(0.5 * self.__population_size + 1)
        log_k = np.log(np.arange(1, self.__population_size + 1))
        numerator = np.maximum(0, log_half - log_k)
        utilities = numerator / np.sum(numerator) - 1 / self.__population_size

        # define sigma
        if sigma_init <= self.__sigma_lower_bound:
            sigma_init = self.__sigma_lower_bound

        self.__sigma = np.ones(n_parameters) * sigma_init
        self.__sigma_init = sigma_init

        # random number generator for drawing samples z_k
        seed: int = int(np.random.rand() * 2**32 - 1)
        self.__sampler = np.random.RandomState(seed)

        # pre calculate values for performance
        self.__u_eta_sigma_half = 0.5 * self.__eta_sigma * utilities
        self.__u_eta_mu = self.__eta_mu * utilities

    # Main Functions
    # ===============================================================
    def do(self, env, policy, n_roll_outs):
        """Runs a single training step:
        1. draw a set of parameter samples
        2. Gets an evaluation (fitness) for all samples using n
        simulations (roll-outs) for each sample on the environment
        3. Updates parameters based on samples sorted by their fitness

        :param env: Contains the gym environment the simulations are
            performed on
        :type env: Environment

        :param policy: The policy to improve
        :type policy: Policy

        :param n_roll_outs: Number of roll outs per policy
        :type n_roll_outs: int

        :return: the array of the fitness of the policies and the array
            of the time steps until the policy encountered the done flag

        :rtype array, array
        """

        mu = policy.get_parameters()

        # draw samples from search distribution
        s = self.__sampler.normal(0, 1, (self.__population_size, len(mu)))
        z = mu + self.__sigma * s

        # evaluate fitness
        fitness, steps = estimate_fitness(policy, env, z, n_roll_outs)

        # sort samples according to fitness
        s_sorted = s[np.argsort(fitness, kind="mergesort")[::-1]]

        # update parameters
        mu += self.__sigma * (self.__u_eta_mu @ s_sorted)
        self.__sigma *= np.exp(self.__u_eta_sigma_half @ (s_sorted ** 2 - 1))

        # sigma has to be positive
        self.__sigma[self.__sigma < self.__sigma_lower_bound] =\
            self.__sigma_lower_bound

        policy.set_parameters(mu)

        return fitness, steps

    # getter only properties
    @property
    def title(self):
        """Generates a title for plotting results containing all
        relevant parameters and the algorithm name

        :return: the title for the plots
        :rtype str
        """

        return r"NES $\lambda = {}, "  \
               r"\sigma_0 = {}, " \
               r"\eta_\sigma = {:.4f}, " \
               r"\eta_\mu = {}$".format(self.__population_size,
                                        self.__sigma_init,
                                        self.__eta_sigma,
                                        self.__eta_mu)

    @property
    def name(self):
        """Returns algorithm name

        :return: 'NES'
        :rtype str
        """

        return 'NES'
