"""Module containing a wrapper for the gym environment providing useful
functions

:Date: 2019-03-11
:Version: 1
:Authors:
    - Kay Hansel
    - Cedric Derstroff
    - Janosch Moos
"""

import numpy as np
import gym
from gym.spaces.discrete import Discrete
from gym.spaces.box import Box
from quanser_robots.common import LabeledBox
from quanser_robots import GentlyTerminating


class Environment:
    """Wraps around the gym environment to add functionality
    - action clipping
    - perform roll-outs
    necessary because of slight differences between standard gym
    environments and quanser robot environments.
    """

    def __init__(self, gym_env, seed: int = 0,
                 horizon: int = None, clip: float = None):
        """
        :param gym_env: Name of the gym environment
        :type gym_env: str

        :param seed: The seed for the environment
        :type seed: int

        :param horizon: Number of maximal time steps in the simulation
            per roll out
        :type horizon: int or None

        :param clip: The maximal absolute value for the action,
        i.e the actions will be clipped to [-clip, clip]
        :type clip: float or None
        """
        env = GentlyTerminating(gym.make(gym_env))
        self.__env = env
        self.__horizon = self.__env.spec.timestep_limit if horizon is None\
            else horizon
        self.act_low = self.__env.action_space.low \
            if clip is None else -np.ones(1) * clip
        self.act_high = self.__env.action_space.high \
            if clip is None else np.ones(1) * clip
        self.seed(seed)
        self.__name = gym_env

    def __del__(self):
        self.__env.close()

    # Utility Functions
    # ===============================================================
    def close(self):
        """Wrapper method for close
        calls close of the gym environment

        :return: the return value of the close call
        """
        return self.__env.close()

    def seed(self, seed):
        """Wrapper method for seed
        calls seed of the gym environment

        :param seed: int

        :return: the return value of the seed call
        """
        return self.__env.seed(seed)

    def get_seed(self):
        """Sets a new seed to the environment and returns it

        :return: the seed of the environment
        :rtype: int
        """
        return self.__env.seed()[0]

    def render(self):
        """Wrapper method for render
        calls render of the gym environment

        :return: the return value of the render call

        :raise; NotImplementedError
        """
        return self.__env.render()

    def reset(self):
        """Wrapper method for reset
        calls reset of the gym environment

        :return: the return value of the reset call
        """
        return self.__env.reset()

    def step(self, action):
        """Wrapper method for step
        calls step of the gym environment

        :param action: the action to take on the environment

        :return: the observations
        """
        a = self.__act_clip(action)
        return self.__env.step(a)

    def obs_dim(self):
        """Function to check which instance of observation dimensions
        are given since standard gym and quanser environment are
        slightly different

        :return: the observation dimension or None
        """

        if isinstance(self.__env.observation_space, (LabeledBox, Box)):
            out = self.__env.observation_space.shape[0]
        elif isinstance(self.__env.observation_space, Discrete):
            out = self.__env.observation_space.n
        else:
            print("Warning: Unknown type of environment: {}"
                  .format(type(self.__env.observation_space)))
            out = None
        return out

    def act_dim(self):
        """Function to check which instance of action dimensions are
        given since standard gym and quanser environment are slightly
        different

        :return: the action dimension or None
        """

        if isinstance(self.__env.action_space, (LabeledBox, Box)):
            out = self.__env.action_space.shape[0]
        elif isinstance(self.__env.action_space, Discrete):
            out = self.__env.action_space.n
        else:
            print("Warning: Unknown type of environment: {}"
                  .format(type(self.__env.action_space)))
            out = None
        return out

    @property
    def name(self):
        """Returns the name of the environment

        :return: the name of the gym environment
        :rtype: str
        """
        return self.__name

    def to_string(self):
        """Returns a string representation of the environment

        :return: the name, horizon and the max action in a string
        :rtype: str
        """
        return "{}_{}_{}".format(self.__name, self.__horizon,
                                 self.act_high[0])

    def __act_clip(self, action):
        return np.clip(action, self.act_low, self.act_high)

    # Main Functions
    # ===============================================================
    def roll_out(self, policy, n_roll_outs: int = 1, normalizer=None,
                 render: bool = False, greedy: bool = False):
        """Performs n roll-outs (simulations) on the environment and
        creates a dictionary containing all relevant information for
        each roll-out such as reward, time steps run as well as actions,
        observations and done flag for each time step.

        :param policy: The policy used to decide which action to take
        :type policy: Policy

        :param n_roll_outs: Number of roll outs
        :type n_roll_outs: int

        :param normalizer: Normalizer to normalize the inputs
        :type normalizer: Normalizer or None

        :param render: If True the episodes will be rendered
        :type render: bool

        :param greedy: Determines whether the action will be evaluated
            greedy or explorative
        :type greedy: bool

        :return: the trajectories of the roll outs
        :rtype: list of dict
        """

        trajectories = []

        for s in range(n_roll_outs):

            observations = []
            actions = []
            rewards = []
            flag = []

            observation = self.__env.reset()
            if isinstance(observation, tuple):
                observation = np.asarray(observation)

            if normalizer is not None:
                observation = normalizer.transform(observation)

            step = 0
            total_reward = 0
            done = False
            # run episode until terminal state or defined horizon
            while step < self.__horizon and not done:

                if render:
                    self.__env.render()
                action = policy.get_action(observation.reshape(1, -1),
                                           greedy=greedy)

                next_observation, reward, done, _ =\
                    self.step(np.asarray(action))

                observations.append(observation)
                actions.append(action)
                rewards.append(reward)
                total_reward += reward

                if done:
                    flag.append(0)
                else:
                    flag.append(1)

                observation = next_observation
                if isinstance(observation, tuple):
                    observation = np.asarray(observation)

                if normalizer is not None:
                    observation = normalizer.transform(observation)

                step += 1

            # Save data as dictionary
            trajectory = dict(
                observations=np.array(observations),
                actions=np.array(actions),
                rewards=np.array(rewards),
                flags=np.array(flag),
                time_steps=step,
                total_reward=total_reward,
                )

            trajectories.append(trajectory)
        return trajectories
