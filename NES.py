import numpy as np

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

    def do(self, f, mu, sigma):
        stop = False
        generation = 0

        # random number generator for drawing samples z_k
        sampler = np.random.RandomState()

        z = np.zeros((self.__population_size, len(mu)))

        while not stop:
            # sample
            s = sampler.normal(0, 1, (self.__population_size, len(mu)))
            z = mu + sigma * s

            # evaluate fitness
            fitness, g = f(self.env, z)

            # calculate log derivatives
            # log_d_mu = self.__calc_log_derivatives_mu(z, mu, sigma)
            log_d_mu = s

            # calculate fisher
            fisher_sigma = np.zeros((len(mu), len(mu)))
            # fisher_mu is the identity

            # log_d_sigma = 0.5 * (np.dot(np.transpose(s), s) - 1 / sigma)

            sum_sigma = 0
            sum_mu = 0
            log_d_sigma = np.zeros(
                (self.__population_size, len(mu), len(mu)))

            for i in range(self.__population_size):
                z_minus_mu = np.transpose(z[i] - mu)
                d = np.multiply(z_minus_mu,
                                z_minus_mu.reshape(-1, 1)) / sigma ** 2
                log_d_sigma[i] = (0.5 * d - 1 / sigma)
                sum_sigma += log_d_sigma[i] * fitness[i]
                sum_mu += log_d_mu[i] * fitness[i]
                fisher_sigma += log_d_sigma[i] * np.transpose(log_d_sigma[i])

            j_sigma = sum_sigma / self.__population_size
            j_mu = sum_mu / self.__population_size

            fisher_sigma /= self.__population_size


            # print(mu + self.__eta_mu * sigma * 1. / len(s) * np.dot(fitness, s))
            # update search space
            mu += self.__eta_mu * j_mu
            # print(mu)

            # We aren't using covariance, so we just need the diagonal
            # sigma += np.diagonal(self.__eta_sigma *
            #                      self.__inverse(fisher_sigma) * j_sigma)
            sigma += self.__eta_sigma / 2. * 1. / len(s) * sigma * np.dot(
                fitness, s ** 2 - 1.)

            # sigma has to be positive
            if np.any(sigma < self.sigma_lower_bound):
                sigma[sigma < self.sigma_lower_bound] = self.sigma_lower_bound

            generation += 1
            print(generation, np.mean(g), max(g), sigma, mu)

            # until stopping criterion is met
            stop = generation >= self.max_iter

        return mu, sigma


    @staticmethod
    def __calc_log_derivatives_mu(z, mu, sigma):
        return (z - mu) / sigma[: None]

    @staticmethod
    def __inverse(matrix):
        return np.linalg.inv(np.diag(np.diagonal(matrix)))
        # u, s, v = np.linalg.svd(matrix)
        # # print(u, s, v)
        # s = np.diag(s**-1)
        # return v @ (s @ u.T)

    # -------------------------------------------------------------------------
    # code from the internet to check

    def optimize(self, func, mu, sigma):
        """
        Evolution strategies using the natural gradient of multinormal search distributions.
        Does not consider covariances between parameters.
        See Wierstra et al. (2014). Natural evolution strategies. Journal of Machine Learning Research, 15(1), 949-980.
        """

        rng = np.random.RandomState()

        generation = 0

        learning_rate_mu = self.__eta_mu
        learning_rate_sigma = default_learning_rate_sigma(mu.size)

        while True:
            s = rng.normal(0, 1, size=(self.__population_size, *np.shape(mu)))
            z = mu + sigma * s

            fitness, g = func(self.env, z)

            utility = fitness

            # update parameter of search distribution via natural gradient descent
            mu += learning_rate_mu * sigma * 1. / len(s) * np.dot(utility, s)
            sigma += learning_rate_sigma / 2. * 1. / len(s) * sigma * np.dot(utility, s ** 2 - 1.)
            # print(np.shape(sigma), np.shape(np.dot(utility, s)), np.shape(s), sigma)

            # enforce lower bound on sigma to avoid negative values
            if np.any(sigma < self.sigma_lower_bound):
                sigma[sigma < self.sigma_lower_bound] = self.sigma_lower_bound

            print(generation, np.mean(g), max(g), sigma, mu)
            generation += 1

            # exit if max iterations reached
            if generation > self.max_iter or np.all(sigma < 1e-10):
                break

        return mu, sigma


def default_learning_rate_sigma(dimensions):
    """
    See Wierstra et al. (2014). Natural evolution strategies. Journal of Machine Learning Research, 15(1), 949-980.
    """
    return (3 + np.log(dimensions)) / (5. * np.sqrt(dimensions))