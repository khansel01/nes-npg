"""Module containing the core class of the Deep Q-Network

:Date: 2019-04-05
:Version: 1
:Authors:
    - Janosch Moos
"""

from utilities.estimations import *
import numpy as np

class DQN:
    """Core class of the DQN.

    - SOURCE

    Attributes
    -----------
    normalizer: Normalizer
        None

    Methods
    -----------
    do(env, policy, n_roll_outs)
        Runs a single training step:
        1.
        2.
        3.
    """

    def __init__(self, baseline, batch_size: int, lr: int = 0.9,
                 _gamma: float = 0.98):
        self.__batch_size = batch_size
        self.__lr = lr
        self.__gamma = _gamma
        self.__memory = []
        self.__baseline = baseline
        self.normalizer = None

    def do(self, env, policy, n_roll_outs):
        """Performs a single update step of the algorithm by first
        simulating n roll outs on the given environment. Experience
        reply is used to increase the performance.

        :param env: The environment the simulations are run on
        :type env: Environment

        :param policy: The decision making policy
        :type policy: Policy

        :param n_roll_outs: The number of roll outs to perform
        :type n_roll_outs: int

        :return: Returns the episodic returns and the time steps
                 of the simulations
        :rtype: array of float, array of float
        """

        trajectories = env.roll_out(policy,
                                    n_roll_outs=n_roll_outs,
                                    render=False,
                                    normalizer=self.normalizer)

        self.__add_to_memory(trajectories)
        batch = self.__experience_replay()
        # update baseline
        estimate_value(trajectories, self.__gamma)
        self.__baseline.train(trajectories)

    def __experience_replay(self):
        """Implements the experience replay. Samples a random batch
        from the memory used for the next update.

        :return: Batch of samples from memory
        :rtype Array_like
        """

        if len(self.__memory) < self.__batch_size:
            return self.__memory
        else:
            return np.random.choice(self.__memory, self.__batch_size)

    def __add_to_memory(self, trajectories):
        """Adds given trajectories to memory for experience replay

        :param trajectories: Given trajectories containing the
               transitions
        :type trajectories: Array of dict
        """

        for i in trajectories:
            np.concatenate(self.__memory, trajectories[i])

    @property
    def title(self):
        """Generates a title for plotting results containing all
        relevant parameters and the algorithm name

        :return: the title for the plots
        :rtype str
        """

        return r"DQN $\alpha = {}, $\gamma = {}$, batch size: {}"\
            .format(self.__lr,
                    self.__gamma,
                    self.__batch_size)

    @property
    def name(self):
        """Returns algorithm name

        :return: 'DQN'
        :rtype str
        """

        return 'DQN'