import numpy as np
import torch as tr

#######################################
# NES
#######################################


# TODO comments
class NES:
    def __init__(self, n_parameters, eta_sigma=None,
                 eta_mu=None, population_size=None,
                 sigma_lower_bound=1e-10,
                 seed=None, sigma_init=1.0):

        # pre calculate value fro performance
        log_d = np.log(n_parameters)

        if population_size is not None:
            self.__population_size = population_size
        else:
            self.__population_size = 4 + int(3 * log_d)

        if eta_sigma is not None:
            self.__eta_sigma = eta_sigma
        else:
            self.__eta_sigma = (3 + log_d) / np.sqrt(n_parameters) / 5

        self.__eta_mu = eta_mu if eta_mu is not None else 1

        self.__sigma_lower_bound = sigma_lower_bound

        # utility is always equal hence we can pre compute it here
        log_half = np.log(0.5 * self.__population_size + 1)
        log_k = np.log(np.arange(1, self.__population_size + 1))
        numerator = np.maximum(0, log_half - log_k)
        self.__u = numerator / np.sum(numerator) - 1 / self.__population_size

        if seed is not None:
            tr.random.manual_seed(seed)
            np.random.seed(seed)

        self.__mu = np.zeros(n_parameters)

        if sigma_init <= self.__sigma_lower_bound:
            sigma_init = self.__sigma_lower_bound

        self.__sigma = np.ones(n_parameters) * sigma_init
        self.__sigma_init = sigma_init

        # random number generator for drawing samples z_k
        self.__sampler = np.random.RandomState(seed)

        self.__u_eta_sigma_half = 0.5 * self.__eta_sigma * self.__u
        self.__u_eta_mu = self.__eta_mu * self.__u

    """ Main Functions """
    """==============================================================="""
    def do(self, env, policy, n_roll_outs):
        # draw samples
        s = self.__sampler.normal(0, 1, (self.__population_size,
                                         len(self.__mu)))

        z = self.__mu + self.__sigma * s

        # evaluate fitness
        fitness, steps = self.f_norm(policy, env,
                                     z, n_roll_outs)

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

    """ fitness functions """
    """==============================================================="""
    def f(self, policy, env, w, n_roll_outs: int = 1):

        samples = np.size(w, 0)
        f = np.zeros(samples)
        steps = np.zeros(samples)

        seed = env.get_seed()

        for s in range(samples):
            policy.set_parameters(w[s])

            env.seed(seed)

            trajectories: dict = env.roll_out(policy, n_roll_outs=n_roll_outs,
                                              greedy=True)

            rewards = np.concatenate([t["rewards"]
                                      for t in trajectories]).reshape(-1, 1)

            steps[s] = np.array(
                [t["time_steps"] for t in trajectories]).sum() / n_roll_outs

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

    def get_title(self):
        return r"NES $\lambda = {}, "  \
               r"\sigma_0 = {}, " \
               r"\eta_\sigma = {:.4f}, " \
               r"\eta_\mu = {}$".format(self.__population_size,
                                        self.__sigma_init,
                                        self.__eta_sigma,
                                        self.__eta_mu)
