import numpy as np


class PolicySquare:
    """ Init """
    """==============================================================="""

    def __init__(self, env):
        """ init """
        self.__input_dim = env.obs_dim() + 1
        self.__output_dim = env.act_dim()

        self.__indices = np.tril_indices(self.__input_dim)

        self.hidden_dim = "square"

        self.__params = np.zeros((int(self.__input_dim
                                      * (self.__input_dim + 1) / 2),
                                  self.__output_dim))

        self.length = np.size(self.__params)

    """ Utility Functions """
    """==============================================================="""

    def get_parameters(self):
        return self.__params.reshape(1)

    def set_parameters(self, new_param):
        self.__params = new_param.reshape(-1, self.__output_dim)

    """ Main Functions """
    """==============================================================="""

    def get_action(self, state, greedy=True):
        o = np.reshape(np.append(state, 1), (-1, 1))
        x = (o * o.transpose())[self.__indices] @ self.__params
        return np.array(np.sum(x)).reshape(-1)

