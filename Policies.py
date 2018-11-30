import numpy as np

#######################################
# Softmax- and GaussPolicy
#######################################


class SoftmaxPolicy:

    # TODO: Comment is missing
    # --
    def get_action(self, state, weights, greedy=False):
        x = self.__get_prob(state, weights)
        if greedy:
            return np.argmax(x)
        else:
            return np.random.choice(len(x[0]), p=x[0])

    # TODO: Comment is missing
    # --
    def get_log_grad(self, state, weights, action):
        log_grad = self.__get_p_grad(state, weights)[action, :]\
                   / self.__get_prob(state, weights)[0, action]
        return state.T @ log_grad[None, :]

    # Returns the Jacobian matrix of the policy with respect to
    # the parameters w.
    def __get_p_grad(self, state, weights):
        prob = self.__get_prob(state, weights)
        prob = prob.reshape(-1, 1)
        return np.diagflat(prob) - np.dot(prob, prob.T)

    # Returns array of shape (1, n_actions) containing the probabilities
    # of each action.
    def __get_prob(self, state, weights):
        x = np.dot(state, weights)
        x = np.exp(x)
        return x / np.sum(x)


class GaussianPolicy:

    # TODO: Comment is missing
    # --
    def get_action(self, state, weights, greedy=False):
        mean = self.__get_mean(state[0], weights[:, 0])
        sigma = self.__get_variance(state[0], weights[:, 1])

        if greedy:
            return mean
        else:
            return np.random.normal(mean, sigma, size=None)

    # TODO: Comment is missing, should the state be splitted ?
    # --
    def get_log_grad(self, state, weights, action):
        eps = np.finfo(np.float32).eps.item()
        log_grad = np.zeros(weights.shape)

        mean = self.__get_mean(state[0], weights[:, 0])
        sigma = self.__get_variance(state[0], weights[:, 1])

        log_grad[:, 0] = (action - mean) / (sigma**2 + eps) * state[0]

        log_grad[:, 1] = ((action - mean)**2 / (sigma**2 + eps) - 1) * state[0]
        return log_grad

    # TODO: Comment is missing
    # --
    def __get_mean(self, state, weights):
        return weights.T @ state

    # TODO: Comment is missing
    # --
    def __get_variance(self, state, weights):
        return np.exp(weights.T @ state)
