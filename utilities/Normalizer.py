import numpy as np

"""Contains the Normalizer class"""


class Normalizer:
    """Normalizer class used for normalizing observations with zero mean"""

    def __init__(self, environment, clip=None):
        self.__N = 1
        self.__mean = np.zeros(environment.obs_dim())
        self.__std = np.ones(environment.obs_dim())
        self.__clip = clip

    # Main Functions
    # ===============================================================
    def update(self, trajectories):
        """Updates the normalizer with new trajectory data to adjust to new
        mean and std
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

        # self.__std -= old_std ** 2 + obs.std(axis=0) ** 2
        # self.__std /= self.__N + obs.shape[0] - 1.0

        self.__std /= self.__N + obs.shape[0]

        self.__std = np.sqrt(self.__std)

        self.__N += obs.shape[0]
        return

    def transform(self, observation):
        """normalize current observations"""
        obs = (observation - self.__mean)/(self.__std + 1e-10)
        np.clip(obs, -self.__clip, self.__clip) if self.__clip is not None \
            else None
        return obs

