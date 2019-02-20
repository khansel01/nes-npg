import time
import matplotlib.pyplot as plt
from utilities.Logger import Logger
import numpy as np

#######################################
# Agent
#######################################


class Agent:
    def __init__(self, env, policy, algorithm, render=False, plot=True):
        self.policy = policy
        self.env = env
        self.algorithm = algorithm
        self.render = render
        self.plot = plot
        self.logger = Logger()

    """ Utility Functions """
    """==============================================================="""

    def set_best_policy(self):

        rewards = np.concatenate(
            [episode["reward_mean"] for episode in self.logger.logger])\
            .squeeze()

        episode = self.logger.logger[rewards.argmax()]

        self.policy.set_parameters(episode["policy_parameters"])
        return

    def print(self, i_episode):

        episode = self.logger.logger[i_episode]

        s = "s" if episode["roll_outs"] > 1 else ""

        print("Episode {} with {} roll-out{}:\n"
              "finished after {} time steps and obtained a reward of {}.\n "
              .format(i_episode, episode["roll_outs"], s,
                      episode["time_mean"].squeeze(),
                      episode["reward_mean"].squeeze()))

    def plot_results(self):

        """ get data out of logger"""
        r_means = np.concatenate(
            [episode["reward_mean"] for episode in self.logger.logger])\
            .squeeze()
        r_stds = np.concatenate(
            [episode["reward_std"] for episode in self.logger.logger])\
            .squeeze()

        t_means = np.concatenate(
            [episode["time_mean"] for episode in self.logger.logger])\
            .squeeze()
        t_stds = np.concatenate(
            [episode["time_std"] for episode in self.logger.logger])\
            .squeeze()

        """ get length """
        length = r_stds.size

        """ plot """
        plt.subplot(2, 1, 1)
        plt.title(self.env.get_name() + "\n"
                  + self.algorithm.get_title()
                  + ", Policy: {}".format(self.policy.hidden_dim))

        plt.fill_between(np.arange(length),
                         r_means - r_stds, r_means + r_stds,
                         alpha=0.3, label='standard deviation',
                         color='green')

        plt.plot(np.arange(length), r_means, label='mean',
                 color='green')

        plt.legend()
        plt.xlabel('Episodes')
        plt.ylabel('Total reward')

        plt.subplot(2, 1, 2)
        plt.fill_between(np.arange(length),
                         t_means - t_stds, t_means + t_stds,
                         alpha=0.3, label='standard deviation')
        plt.plot(np.arange(length), t_means, label='mean')
        plt.legend()
        plt.xlabel('Episodes')
        plt.ylabel('Time steps')
        plt.show()
        return

    """ Main Functions """
    """==============================================================="""

    def train_policy(self, episodes, n_roll_outs: int = 1):

        for i_episode in range(episodes):

            """ update policy """
            returns, steps = self.algorithm.do(self.env, self.policy,
                                               n_roll_outs)

            """ log data """
            self.logger.log_data(returns, steps, n_roll_outs,
                                 self.policy.get_parameters())

            """ analyze episode """
            self.print(i_episode)

        if self.plot:
            self.plot_results()

    # TODO not finished
    ''' run benchmark test'''
    def benchmark_test(self, episodes: int = 100, render: bool = False):

        """ set policy parameters to best performed parameters"""
        self.set_best_policy()

        """ do roll outs"""
        trajectories = self.env.roll_out(self.policy, n_roll_outs=episodes,
                                         render=render, greedy=True)

        # rewards_sum = np.concatenate(
        #     [t["rewards"] for t in trajectories])

        # average = rewards_sum / episodes
        # print("Average Reward: ", average)
        # if average >= 195:
        #     return True
        # else:
        #     return False
        return

