import numpy as np

#######################################
# Softmax- and GaussPolicy
#######################################


class SoftmaxPolicy:

    def __init__(self, env):
        self.__obs_dim = env.obs_dim()
        self.__act_dim = env.act_dim()
        # TODO weight act dim
        self.weights = [] # np.random.sample(self.__obs_dim*(self.__act_dim+1))
        self.greedy = False
        self.eps = 0

    # TODO: Comment is missing
    # --
    def set_greedy(self, boolean: bool):
        self.greedy = boolean
        return

    # TODO: Comment is missing
    # --
    def get_action(self, state, greedy=None):
        greedy = self.greedy if greedy is None else greedy

        self.weights = np.random.sample(len(state)*(self.__act_dim)) \
            if self.weights == [] else self.weights

        x = self.__get_prob(state)
        print(state)
        if greedy:
            return np.argmax(x)
        else:
            return np.random.choice(len(x), p=x)

    # TODO: Comment is missing
    # --
    def get_log_grad(self, state, action):
        self.weights = np.random.sample(len(state)*(self.__act_dim)) \
            if self.weights == [] else self.weights

        log_grad = self.__get_p_grad(state)[action, :]\
                   / self.__get_prob(state)[0, action]
        return state.T @ log_grad[None, :]

    # Returns the Jacobian matrix of the policy with respect to
    # the parameters w.
    def __get_p_grad(self, state):
        prob = self.__get_prob(state)
        prob = prob.reshape(-1, 1)
        return np.diagflat(prob) - np.dot(prob, prob.T)

    # Returns array of shape (1, n_actions) containing the probabilities
    # of each action.
    def __get_prob(self, state):
        x = np.dot(state, self.weights.reshape(
            self.__obs_dim, self.__act_dim, order='F'))
        x = np.exp(x)
        return x / np.sum(x)


class GaussianPolicy:

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

        self.weights = np.random.sample(len(state)*(self.__act_dim+1)) \
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

        self.weights = np.random.sample(len(state)*(self.__act_dim+1)) \
            if self.weights == [] else self.weights

        eps = np.finfo(np.float32).eps.item()
        self.eps = eps
        log_grad = np.zeros((len(self.weights), action.size))

        mean = self.__get_mean(state)
        sigma = self.__get_variance(state)

        log_grad[:len(self.weights)//2, :] = \
            (action.T - mean) / (sigma**2 + eps) * state.T
        log_grad[len(self.weights)//2:, :] = \
            ((action.T - mean)**2 / (sigma**2 + eps) - 1) * state.T
        return log_grad

    # TODO: Comment is missing
    # --
    def __get_mean(self, state):
        return state @ self.weights[:len(self.weights)//2]

    # TODO: Comment is missing
    # --
    def __get_variance(self, state):
        return np.exp(state @ self.weights[len(self.weights)//2:])
