import numpy as np

class SoftmaxPolicy:

    # Returns array of shape (1, n_actions) containing the probabilities
    # of each action.
    def get_action_prob(self, state, w):
        # state.shape = (1,n) // w.shape = (n, n_actions)
        x = np.dot(state, w)
        x = np.exp(x)
        return x/np.sum(x)

    # Returns the Jacobian matrix of the policy with respect to
    # the parameters w.
    def get_p_grad(self, state, w):
        prob = self.get_action_prob(state, w)
        prob = prob.reshape(-1, 1)
        return np.diagflat(prob) - np.dot(prob, prob.T)
