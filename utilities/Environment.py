import numpy as np
import gym
from gym.spaces.discrete import Discrete
from gym.spaces.box import Box
from quanser_robots.common import LabeledBox
from quanser_robots import GentlyTerminating

"""Module containing a wrapper for the gym environment providing useful 
functions
"""


class Environment:
    """Wraps around the gym environment to add functionality
    - action clipping
    - perform roll-outs
    necessary because of slight differences between standard gym environments
    and quanser robot environments.
    """

    def __init__(self, gym_env, seed: int = 0,
                 horizon: int = None, clip: float = None):
        env = GentlyTerminating(gym.make(gym_env))
        self.__env = env
        self.__horizon = self.__env.spec.timestep_limit if horizon is None\
            else horizon
        self.__seed = seed
        self.act_low = self.__env.action_space.low \
            if clip is None else -np.ones(1) * clip
        self.act_high = self.__env.action_space.high \
            if clip is None else np.ones(1) * clip
        self.seed(self.__seed)
        self.__name = gym_env

    def __del__(self):
        self.__env.close()

    # Utility Functions
    # ===============================================================
    def close(self):
        return self.__env.close()

    def seed(self, seed):
        return self.__env.seed(seed)

    def get_seed(self):
        return self.__env.seed()[0]

    def render(self):
        return self.__env.render()

    def reset(self):
        return self.__env.reset()

    def step(self, action):
        a = self.__act_clip(action)
        return self.__env.step(a)

    def obs_dim(self):
        """Function to check which instance of observation dimensions are given
        since standard gym and quanser environment are slightly different
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
        """Function to check which instance of action dimensions are given
        since standard gym and quanser environment are slightly different
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

    def get_name(self):
        return self.__name

    def to_string(self):
        return "{}_{}_{}".format(self.__name, self.__horizon,
                                 self.act_high[0])

    def __act_clip(self, action):
        return np.clip(action, self.act_low, self.act_high)

    # Main Functions
    # ===============================================================
    def roll_out(self, policy, n_roll_outs: int = 1, normalizer=None,
                 render: bool = False, greedy: bool = False):
        """Performs n roll-outs (simulations) on the environment and creates a
        dictionary containing all relevant information for each roll-out such
        as reward, time steps run as well as actions, observations and done
        flag for each time step.
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

            observation = normalizer.transform(observation) \
                if normalizer is not None else observation

            step = 0
            total_reward = 0
            done = False
            # run episode until terminal state or defined horizon
            while step < self.__horizon and done is not True:

                self.__env.render() if render else None
                action = policy.get_action(observation.reshape(1, -1),
                                           greedy=greedy)

                next_observation, reward, done, _ =\
                    self.step(np.asarray(action))

                observations.append(observation)
                actions.append(action)
                rewards.append(reward)
                total_reward += reward
                flag.append(0) if done else flag.append(1)

                observation = next_observation
                if isinstance(observation, tuple):
                    observation = np.asarray(observation)
                observation = normalizer.transform(observation) \
                    if normalizer is not None else observation

                step += 1
                if done:
                    break

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
            self.__env.step(np.zeros(1))
        return trajectories

