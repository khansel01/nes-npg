import numpy as np
import gym
from gym.spaces.discrete import Discrete
from gym.spaces.box import Box
from quanser_robots.common import LabeledBox

#######################################
# Environment
#######################################


class Environment:

    def __init__(self, gym_env, seed: int=0, horizon: bool=None):
        env = gym.make(gym_env)
        self.__env = env
        self.__horizon = self.__env.spec.timestep_limit if horizon is None\
            else horizon
        self.__seed = seed
        self.seed(self.__seed)

    def close(self):
        self.__env.close()
        return

    def seed(self, seed):
        self.__env.seed(seed)
        return

    def reset(self):
        return self.__env.reset()

    def step(self, action):
        return self.__env.step(action)

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

    def __act_clip(self, action):
        if isinstance(self.__env.action_space, (LabeledBox, Box)):
            return np.clip(action, self.__env.action_space.low,
                           self.__env.action_space.high)
        else:
            return action

    def roll_out(self, policy, amount: int = 1, render: bool = False,
                 greedy: bool = False):
        trajectories = []

        for s in range(amount):

            observations = []
            actions = []
            rewards = []

            observation = self.__env.reset()
            if isinstance(observation, tuple):
                observation = np.asarray(observation)

            step = 0
            done = False
            while done is not True and step < self.__horizon:

                self.__env.render() if render else None
                action = policy.get_action(observation, greedy=greedy)
                action = self.__act_clip(action)

                next_observation, reward, done, _ =\
                    self.step(np.asarray(action))
                observations.append(observation)
                actions.append(action)
                if reward < 1:
                    rewards.append(reward)
                else:
                    rewards.append(reward*4)

                observation = next_observation
                if isinstance(observation, tuple):
                    observation = np.asarray(observation)
                step += 1
                if done:
                    break

            trajectory = dict(
                observations=np.array(observations),
                actions=np.array(actions),
                rewards=np.array(rewards)
                )

            trajectories.append(trajectory)

        return trajectories

