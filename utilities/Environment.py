import numpy as np
import gym
from gym.spaces.discrete import Discrete
from gym.spaces.box import Box
import quanser_robots
from quanser_robots.common import LabeledBox

#######################################
# Environment
#######################################


class Environment:

    """ Init """
    """==============================================================="""
    def __init__(self, gym_env, seed: int = 0, horizon: int = None):
        env = gym.make(gym_env)
        self.__env = env
        self.__horizon = self.__env.spec.timestep_limit if horizon is None\
            else horizon
        self.__seed = seed
        self.seed(self.__seed)
        self.__name = gym_env

    """ Utility Functions """
    """==============================================================="""
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

    def __act_clip(self, action):
        if isinstance(self.__env.action_space, (LabeledBox, Box)):
            return np.clip(action, self.__env.action_space.low,
                           self.__env.action_space.high)
        else:
            return action

    """ Main Functions """
    """==============================================================="""
    def roll_out(self, policy, n_roll_outs: int = 1, normalizer=None,
                 render: bool = False):
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
            done = False
            while not done and step < self.__horizon:

                self.__env.render() if render else None
                action = policy.get_action(observation.reshape(1, -1))

                next_observation, reward, done, _ =\
                    self.step(np.asarray([action]))
                observations.append(observation)
                actions.append(action)
                rewards.append(reward)
                flag.append(0) if done else flag.append(1)

                observation = next_observation
                if isinstance(observation, tuple):
                    observation = np.asarray(observation)
                observation = normalizer.transform(observation) \
                    if normalizer is not None else observation

                step += 1
                if done:
                    break

            trajectory = dict(
                observations=np.array(observations),
                actions=np.array(actions),
                rewards=np.array(rewards),
                flags=np.array(flag),
                steps=step
                )

            trajectories.append(trajectory)

        return trajectories

