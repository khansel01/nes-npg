"""Contains the Normalizer class

:Date: 2019-03-11
:Version: 1
:Authors:
    - Cedric Derstroff
    - Janosch Moos
    - Kay Hansel
"""

import numpy as np


class Normalizer:
    """Normalizer class used for normalizing observations with zero mean

    Methods
    -----------
    update(trajectories)
        Update mean and standard deviation with the observations of the
        trajectories

    transform(observation)
        Normalize the observations
    """

    def __init__(self, environment, clip=None):
        self.__N = 1
        self.__mean = np.zeros(environment.obs_dim())
        self.__std = np.ones(environment.obs_dim())
        self.__clip = clip

    # Main Functions
    # ===============================================================
    def update(self, trajectories):
        """This function updates the normalizer with new trajectory data
        to adjust to new mean and std

        :param trajectories: Contains a set of trajectories each being a
        dictionary with information about every transition performed in
        the trajectory simulation
        :type trajectories: list of dict
        """

        # get observations
        obs = np.concatenate([t["observations"]
                              for t in trajectories])

        # update mean and std
        old_mean = self.__mean

        self.__mean = self.__N * self.__mean + obs.shape[0] * obs.mean(axis=0)
        self.__mean /= self.__N + obs.shape[0]

        old_std = self.__std

        self.__std = self.__N * (
                    old_std ** 2 + (old_mean - self.__mean) ** 2)
        self.__std += obs.shape[0] * (
                obs.std(axis=0) ** 2 + (obs.mean(axis=0) - self.__mean) ** 2)

        self.__std /= self.__N + obs.shape[0]

        self.__std = np.sqrt(self.__std)

        self.__N += obs.shape[0]

    def transform(self, observation):
        """This function normalizes current observations

        :param observation: Unnormalized observations of the environment
        :type   observation: array_like

        :return: Normalized observations with zero mean
        :rtype: array_like
        """

        obs = (observation - self.__mean)/(self.__std + 1e-10)
        np.clip(obs, -self.__clip, self.__clip) if self.__clip is not None \
            else None
        return obs
