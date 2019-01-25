import numpy as np

#######################################
# SoftmaxPolicy
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
    def set_greedy(self, boolean: bool):
        self.greedy = boolean
        return

    # TODO: Comment is missing
    # --
    def get_action(self, state, greedy=None):
        greedy = self.greedy if greedy is None else greedy
        self.weights = np.random.sample((len(state), self.__act_dim)) \
            if self.weights == [] else self.weights

        x = self.__get_prob(state)
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
                   / self.__get_prob(state)[action]
        return state.reshape(-1, 1) @ log_grad[None, :]

    # Returns the Jacobian matrix of the policy with respect to
    # the parameters w.
    def __get_p_grad(self, state):
        prob = self.__get_prob(state)
        prob = prob.reshape(-1, 1)
        return np.diagflat(prob) - np.dot(prob, prob.T)

    # Returns array of shape (1, n_actions) containing the probabilities
    # of each action.
    def __get_prob(self, state):
        x = np.dot(state, self.weights)
        x = np.exp(x)
        return x / np.sum(x)
