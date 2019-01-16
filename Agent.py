import time
import matplotlib.pyplot as plt
from Estimations import *
from Logger import Logger

#######################################
# Agent
#######################################


class Agent:
    def __init__(self, env, policy, algorithm, baseline,
                 _lambda=0.95, _gamma=0.98, render=False, plot=True):
        self.policy = policy
        self.env = env
        self.algorithm = algorithm
        self.__lambda = _lambda
        self.__gamma = _gamma
        self.__eps = 1e-6
        self.baseline = baseline
        self.render = render
        self.plot = plot
        self.logger = Logger()

    def train_policy(self, episodes, amount: int=1, times: bool=False):

        for i_episode in range(episodes):

            #   roll out trajectories
            delta_t_e = time.time()
            if i_episode + 1 == episodes:
                trajectories = self.env.roll_out(self.policy, amount=amount,
                                                 render=self.render)
            else:
                trajectories = self.env.roll_out(self.policy, amount=amount,
                                                 render=False)

            #   update policy
            delta_t_p = time.time()

            print("log_grad:", self.policy.log_std.detach().numpy().squeeze())

            estimate_advantage(trajectories,
                               self.baseline, self.__gamma, self.__lambda)
            self.algorithm.do(trajectories, self.policy)

            delta_t_p = time.time() - delta_t_p

            #   update critic
            delta_t_c = time.time()

            estimate_value(trajectories, self.__gamma)
            self.baseline.train(trajectories)

            delta_t_c = time.time() - delta_t_c
            delta_t_e = time.time() - delta_t_e

            #   log data
            self.logger.log_data(trajectories, self.policy.get_parameters(),
                                 delta_t_c, delta_t_p, delta_t_e)

            #   analyze episode
            self.analyze(i_episode, times)

        self.show() if self.plot is True else None

        self.env.close()
        return False

    def benchmark_test(self, episodes: int=100, render: bool=False):
        self.set_best_policy()

        trajectories = self.env.roll_out(self.policy, amount=episodes,
                                         render=render)

        # rewards_sum = np.concatenate(
        #     [t["rewards"] for t in trajectories])

        # average = rewards_sum / episodes
        # print("Average Reward: ", average)
        # if average >= 195:
        #     return True
        # else:
        #     return False
        return

    def set_best_policy(self):
        rewards = np.concatenate(
            [episode["reward_mean"] for episode in self.logger.logger])\
            .squeeze()
        episode = self.logger.logger[rewards.argmax()]
        self.policy.set_parameters(episode["policy_parameters"])
        return

    def analyze(self, i_episode, times: bool = False):
        episode = self.logger.logger[i_episode]
        if times:
            print("Episode {} with {} roll-outs:\n "
                  "finished after {} and obtained a reward of {}.\n "
                  "Episode needs {} seconds.\n "
                  "Update the baseline needs {} seconds.\n "
                  "Update the policy needs {} sedonds.\n"
                  .format(i_episode, episode["roll_outs"].squeeze(),
                          episode["time_mean"].squeeze(),
                          episode["reward_mean"].squeeze(),
                          episode["time_episode"].squeeze(),
                          episode["time_critic"].squeeze(),
                          episode["time_policy"].squeeze()))
            return
        else:
            print("Episode {} with {} roll-outs:\n "
                  "finished after {} and obtained a reward of {}.\n "
                  .format(i_episode, episode["roll_outs"].squeeze(),
                          episode["time_mean"].squeeze(),
                          episode["reward_mean"].squeeze()))
            return

    def show(self):
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

        # get length
        length = r_stds.size

        #   plot
        plt.subplot(2, 1, 1)
        plt.fill_between(np.arange(length),
                         r_means - r_stds, r_means + r_stds,
                         alpha=0.3, label='standard deviation',
                         color='green')
        plt.plot(np.arange(length), r_means, label='mean',
                 color='green')
        plt.legend()
        plt.xlabel('Episodes')
        plt.ylabel('Rewards')

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

