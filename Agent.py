import time
import matplotlib.pyplot as plt
from utilities.Logger import Logger
import numpy as np
import pickle

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

        benchmark = np.asarray(
            [episode["benchmark_reward"] for episode in self.logger.logger])\
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
        plt.plot(np.arange(length), benchmark, label='benchmark')
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

        """ save in csv """
        string = 'trained_data/training_data_{}_{}.csv'\
            .format(self.env.to_string(), self.algorithm.get_name())
        array = np.asarray((r_means, r_stds, t_means, t_stds, benchmark))
        np.savetxt(string, array.T, delimiter=',', fmt='%10.5f',
                   header="r_means, r_stds, t_means, t_sdts, benchmark")
        return

    """ Main Functions """
    """==============================================================="""

    def train_policy(self, episodes, n_roll_outs: int = 1,
                     save: bool = False):

        for i_episode in range(episodes):

            """ update policy """
            returns, steps = self.algorithm.do(self.env, self.policy,
                                               n_roll_outs)

            """ do greedy run for plot purposes"""
            self.env.seed(0)
            eval: dict = self.env.roll_out(self.policy,
                                     normalizer=self.algorithm.normalizer,
                                     greedy=True)
            self.env.seed()

            """ log data """
            self.logger.log_data(returns, steps, n_roll_outs,
                                 self.policy.get_parameters(),
                                 eval[0]["total_reward"])

            """ analyze episode """
            self.print(i_episode)

            if save:
                print("{:-^50s}".format(' Save '))
                pickle_out = open("trained_data/{}_{}.p".format(self.env.to_string(),
                                                   self.algorithm.get_name()),
                                  "wb")
                pickle.dump((self.policy, self.algorithm), pickle_out)
                pickle_out.close()

        if self.plot:
            self.plot_results()

    ''' run benchmark test'''

    def run_benchmark(self, episodes=100, render: bool = False):

        """ Starting Benchmark """
        trajectories = self.env.roll_out(self.policy, n_roll_outs=episodes,
                                         # normalizer=None,
                                         normalizer=self.algorithm.normalizer,
                                         greedy=True, render=render)

        total_rewards = []
        for i, t in np.ndenumerate(trajectories):
            print(i[0] + 1,
                  "Reward reached: ", t["total_reward"])
            total_rewards.append(t["total_reward"])
        if render:
            return

        print("-------------------")
        print("Average reward: ", np.mean(total_rewards))
        print("Min reward:", np.min(total_rewards))
        print("Max reward:", np.max(total_rewards))

        """ 1. Plot: Total reward"""
        plt.plot(np.arange(len(total_rewards)), total_rewards,
                 label='Total reward per episode', color='darkgreen')
        plt.fill_between(np.arange(len(total_rewards)),
                         0, total_rewards,
                         alpha=0.3, color='green')
        plt.legend()
        plt.xlabel('Episodes')
        plt.ylim(bottom=0)
        plt.ylabel('Total reward')
        plt.show()

        """ 2. Plot: reward per time step"""
        plt.plot(np.arange(trajectories[0]["time_steps"]),
                 trajectories[0]["rewards"],
                 label='1. Run', color='green')
        plt.plot(np.arange(trajectories[1]["time_steps"]),
                 trajectories[1]["rewards"],
                 label='2. Run', color='blue')
        plt.plot(np.arange(trajectories[2]["time_steps"]),
                 trajectories[2]["rewards"],
                 label='3. Run', color='red')
        # plt.plot(np.arange(trajectories[3]["time_steps"]),
        #          trajectories[3]["rewards"],
        #          label='4. Run', color='orange')
        # plt.plot(np.arange(trajectories[4]["time_steps"]),
        #          trajectories[4]["rewards"],
        #          label='5. Run', color='pink')
        plt.legend()
        plt.xlabel('Time steps')
        plt.ylabel('Reward')
        plt.show()

        # """ save in csv """
        # string = 'trained_data/benchmark_data_{}_{}.csv'\
        #     .format(self.env.to_string(), self.algorithm.get_name())
        # print(time_rewards)
        # print(type(time_rewards))
        # array = np.asarray(time_rewards)
        # np.savetxt(string, array.T, delimiter=',', fmt='%10.5f')
        return

