import numpy as np

#######################################
# NPG
#######################################


#TODO comments
class NPG:
    def __init__(self, _delta=0.05):
        np.random.seed(1)
        self.__delta = _delta

    def do(self, trajectories, policy):
        observations = np.concatenate([t["observations"]
                                       for t in trajectories])
        actions = np.concatenate([t["actions"]
                                  for t in trajectories]).reshape(-1, 1)

        advantage = np.concatenate([t["advantages"]
                                    for t in trajectories]).reshape(-1, 1)

        #   vanilla gradient for each step
        log_grad = np.zeros((policy.weights.size, len(actions)))
        for i in range(len(actions)):
            log_grad[:, i:i + 1] = policy.get_log_grad(
                observations[i, :], actions[i]).reshape((-1, 1), order='F')
        #log_grad = policy.get_log_grad(observations, actions)

        #   vanilla gradient for each step
        vpg = log_grad @ advantage
        vpg /= log_grad.shape[1]

        #   Fisher matrix
        fisher = log_grad @ log_grad.T
        fisher /= log_grad.shape[1]
        fisher = np.diagonal(fisher)[None, :]
        fisher = np.diagflat(fisher)
        inv_fisher = self.__compute_inverse(fisher)

        #   update step
        nominator = vpg.T @ (inv_fisher @ vpg)
        learning_rate = np.sqrt(self.__delta / nominator)
        step = np.multiply(learning_rate, (inv_fisher @ vpg))
        policy.weights += step.reshape(policy.weights.shape, order='F')
        return

    @staticmethod
    def __compute_inverse(matrix):
        u, s, v = np.linalg.svd(matrix)
        s = np.diag(s**-1)
        return v.T @ (s @ u.T)





