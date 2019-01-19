import numpy as np

#######################################
# Normalize the observations
#######################################


class Normalizer:
    def __init__(self, environment):
        self.__N = 1
        self.__mean = np.zeros(environment.obs_dim())
        self.__std = np.ones(environment.obs_dim())

    def update(self, trajectories):

        """ get observations"""
        obs = np.concatenate([t["observations"]
                              for t in trajectories])

        # TODO fit the mean and std even with the respect to the old roll outs
        """ update mean and std """
        old_mean = self.__mean

        self.__mean = self.__N * self.__mean + obs.shape[0] * obs.mean(axis=0)
        self.__mean /= self.__N + obs.shape[0]

        old_std = self.__std

        self.__std = self.__N * (
                    old_std ** 2 + (old_mean - self.__mean) ** 2)
        self.__std += obs.shape[0] * (
                obs.std(axis=0) ** 2 + (obs.mean(axis=0) - self.__mean) ** 2)
        self.__std -= old_std ** 2 + obs.std(axis=0) ** 2
        self.__std /= self.__N + obs.shape[0] - 1.0
        self.__std = np.sqrt(self.__std)

        self.__N += obs.shape[0]
        return

    """ normalize current observations """
    def transform(self, observation):
        return (observation - self.__mean)/self.__std

