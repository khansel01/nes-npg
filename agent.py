"""This module contains the agent class

:Date: 2019-03-11
:Version: 1
:Authors:
    - Janosch Moos
    - Kay Hansel
    - Cedric Derstroff
"""

import matplotlib.pyplot as plt
from utilities.Logger import Logger
import numpy as np
import pickle
import csv


class Agent:
    """Agent Class
    It wraps around environment, policy and algorithm. The class
    basically controls the training process, benchmarks and
    documentation of results.
    """

    def __init__(self, env, policy, algorithm, plot=True):
        """
        :param env: Contains the gym environment the simulations are
            performed on
        :type env: Environment

        :param policy: The policy to improve
        :type policy: Policy

        :param algorithm: The learning algorithm
        :type algorithm: NES or NPG

        :param plot: If True the results of Training and Benchmark will
            be plotted
        :type plot: bool
        """
        self.policy = policy
        self.env = env
        self.algorithm = algorithm
        self.plot = plot
        self.logger = Logger()

    # Utility Functions
    # ===============================================================
    def __print(self, i_episode):
        """Prints results for a given episode of the logged episodes"""

        episode = self.logger.logger[i_episode]

        s = "s" if episode["roll_outs"] > 1 else ""

        print("Episode {} with {} roll-out{}:\n"
              "finished after {} time steps and obtained a reward of {}.\n "
              .format(i_episode, episode["roll_outs"], s,
                      episode["time_mean"].squeeze(),
                      episode["reward_mean"].squeeze()))

    def __plot_results(self):
        """Generates plots after the training process containing the
        relevant information such as reward and time steps of each
        episode. In case more than one simulation was performed each
        episode, the mean and standard deviation for each are plotted.
        """

        # string for csv file
        string = 'trained_data/training_data_{}_{}.csv'\
            .format(self.env.to_string(), self.algorithm.get_name())

        # get data out of logger
        r_means = []
        r_stds = []
        t_means = []
        t_stds = []

        with open(string, 'w') as writerFile:
            for i, e in np.ndenumerate(self.logger.logger):
                r_means.append(e["reward_mean"])
                r_stds.append(e["reward_std"])
                t_means.append(e["time_mean"])
                t_stds.append(e["time_std"])

                """ write to csv file """
                writer = csv.writer(writerFile)
                writer.writerow([i[0], e["reward_mean"].squeeze(),
                                e["reward_mean"].squeeze()
                                - e["reward_std"].squeeze(),
                                e["reward_mean"].squeeze()
                                + e["reward_std"].squeeze(),
                                e["time_mean"].squeeze(),
                                e["time_mean"].squeeze()
                                - e["time_std"].squeeze(),
                                e["time_mean"].squeeze()
                                + e["time_std"].squeeze()])
            writerFile.close()

        r_means = np.concatenate(r_means).squeeze()
        r_stds = np.concatenate(r_stds).squeeze()
        t_means = np.concatenate(t_means).squeeze()
        t_stds = np.concatenate(t_stds).squeeze()

        # get length
        length = r_stds.size

        # plot
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

    # Main Functions
    # ===============================================================
    def train_policy(self, episodes, n_roll_outs: int = 1,
                     save: bool = False):
        """Basic overlay for training the algorithms. It controls the
        amount of episodes, logging and saving of policies and data.

        :param episodes: Number of episodes to Train
        :type episodes: int

        :param n_roll_outs: Number of roll outs
        :type n_roll_outs: int

        :param save: If True the policy is saved after every learning
            step
        :type save: bool
        """

        for i_episode in range(episodes):

            # update policy
            returns, steps = self.algorithm.do(self.env, self.policy,
                                               n_roll_outs)

            # log data
            self.logger.log_data(returns, steps, n_roll_outs,
                                 self.policy.get_parameters())

            # analyze episode
            self.__print(i_episode)

            if save:
                print("{:-^50s}".format(' Save '))
                pickle_out = open("trained_data/{}_{}.p".format(self.env.to_string(),
                                                   self.algorithm.get_name()),
                                  "wb")
                pickle.dump((self.policy, self.algorithm), pickle_out)
                pickle_out.close()

        if self.plot:
            self.__plot_results()

    def run_benchmark(self, episodes=100, render: bool = False):
        """Runs a benchmark test with a set amount of simulations
        (episodes) and plots results. There are three plots generated:
         1. Reward per episode
         2. Reward per time step of the first three episodes
         3. Reward per time step for all episodes
         The second and third plot do not take the mean but rather plot
         a curve for each episode.

        :param episodes: Number of episodes for the benchmark
        :type episodes: int

        :param render: If True the episodes will be rendered
        :type render: bool
        """

        # perform simulations
        trajectories = self.env.roll_out(self.policy, n_roll_outs=episodes,
                                         normalizer=self.algorithm.normalizer,
                                         greedy=True, render=render)

        total_rewards = []
        rewards = []
        time_steps = []
        for i, t in np.ndenumerate(trajectories):
            print(i[0] + 1,
                  "Reward reached: ", t["total_reward"])
            total_rewards.append(t["total_reward"])
            rewards.append(t["rewards"])
            time_steps.append(t["time_steps"])
        if render:
            return

        print("-------------------")
        print("Average reward: ", np.mean(total_rewards))
        print("Min reward:", np.min(total_rewards))
        print("Max reward:", np.max(total_rewards))

        if self.plot:
            self.__plot_benchmark(total_rewards, rewards, time_steps,
                                  trajectories)

    def __plot_benchmark(self, total_rewards, rewards, time_steps,
                         trajectories):

        # 1. Plot: Total reward
        plt.plot(np.arange(len(total_rewards)), total_rewards,
                 label='Total reward per episode', color='darkgreen')
        plt.fill_between(np.arange(len(total_rewards)),
                         0, total_rewards,
                         alpha=0.3, color='green')
        plt.legend()
        plt.xlabel('Trial')
        plt.ylim(bottom=0)
        plt.ylabel('Total reward')
        plt.title("Benchmark Result for " + self.env.get_name() + "\n"
                  + "with " + self.algorithm.get_title()
                  + ", Policy: {}".format(self.policy.hidden_dim))
        plt.show()

        # 2. Plot: reward per time step for only first 3 runs
        plt.plot(np.arange(trajectories[0]["time_steps"]),
                 trajectories[0]["rewards"],
                 label='1. Run', color='green')
        plt.plot(np.arange(trajectories[1]["time_steps"]),
                 trajectories[1]["rewards"],
                 label='2. Run', color='blue')
        plt.plot(np.arange(trajectories[2]["time_steps"]),
                 trajectories[2]["rewards"],
                 label='3. Run', color='red')
        plt.legend(["All Trials"])
        plt.xlabel('Time steps')
        plt.ylabel('Reward')
        plt.title("Reward per time step during benchmark of "
                  + self.env.get_name() + "\n"
                  + "with " + self.algorithm.get_title()
                  + ", Policy: {}".format(self.policy.hidden_dim))
        plt.show()

        # 3. Plot: reward per time step for all runs
        for r in rewards:
            plt.plot(np.arange(len(r)), r, linewidth=1)
        plt.legend(["Each Trial"])
        plt.xlabel('Time steps')
        plt.ylabel('Reward')
        plt.title("Reward per time step during benchmark of "
                  + self.env.get_name() + "\n"
                  + "with " + self.algorithm.get_title()
                  + ", Policy: {}".format(self.policy.hidden_dim))
        plt.show()

        # save in csv
        string = 'trained_data/benchmark_data_{}_{}.csv' \
            .format(self.env.to_string(), self.algorithm.get_name())

        with open(string, 'w') as writerFile:
            for step in range(max(time_steps)):
                step_rewards = [step]
                for r in rewards:
                    try:
                        step_rewards.append(r[step])
                    except IndexError:
                        step_rewards.append(None)
                writer = csv.writer(writerFile)
                writer.writerow(step_rewards)
            writerFile.close()
