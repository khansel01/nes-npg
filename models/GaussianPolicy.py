import numpy as np

#######################################
# GaussPolicy
#######################################


class Policy:

    def __init__(self, env):
        self.__obs_dim = env.obs_dim()
        self.__act_dim = env.act_dim()
        # TODO weight act dim
        self.weights = [] # np.random.sample(self.__obs_dim*(self.__act_dim+1))
        self.greedy = False
        self.eps = 0

    # TODO: Comment is missing
    # --
    def get_action(self, state, greedy=False):
        greedy = self.greedy if greedy is None else greedy

        self.weights = np.random.sample((len(state), (self.__act_dim+1))) \
            if self.weights == [] else self.weights

        mean = self.__get_mean(state)
        sigma = self.__get_variance(state)
        if greedy:
            return mean
        else:
            return mean + sigma*np.random.randn()

    # TODO: Comment is missing, should the state be splitted ?
    # --
    def get_log_grad(self, state, action):

        self.weights = np.random.sample((len(state), self.__act_dim+1)) \
            if self.weights == [] else self.weights

        eps = np.finfo(np.float32).eps.item()
        self.eps = eps
        log_grad = np.zeros((self.weights.size, action.size))

        mean = self.__get_mean(state)
        sigma = self.__get_variance(state)

        log_grad[:self.weights.size//2, :] = \
            ((action - mean) / (sigma**2 + eps) * state).T
        log_grad[self.weights.size//2:, :] = \
            (((action - mean)**2 / (sigma**2 + eps) - 1) * state).T
        return log_grad

    # TODO: Comment is missing
    # --
    def __get_mean(self, state):
        return state @ self.weights[:, 0].reshape(-1, 1)

    # TODO: Comment is missing
    # --
    def __get_variance(self, state):
        return np.exp(state @ self.weights[:, 1]).reshape(-1, 1)

