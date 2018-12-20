import numpy as np
import matplotlib.pyplot as plt
from Baseline import Baseline
from Advantage_fcts import *

#######################################
# Agent
#######################################


class Agent:
    def __init__(self, env, policy, algorithm, feature=None,
                 _lambda=0.95, _gamma=0.98, render=False, plot=True):
        self.policy = policy
        self.env = env
        self.feature = feature
        self.algorithm = algorithm
        self.__lambda = _lambda
        self.__gamma = _gamma
        self.__eps = 1e-6
        self.baseline = Baseline(4 + 2, 1)
        self.render = render
        self.plot = plot
        #np.random.seed(1)

    def train_policy(self, episodes):
        mean_per_episode = []
        std_per_episode = []
        for i_episode in range(episodes):
            print("\nbegin episode: ", i_episode)

        #   roll out trajectories
            trajectories = self.env.roll_out(self.policy,
                                             features=self.feature,
                                             amount=5, render=self.render)

        #   log data
            timesteps = np.mean([len(t["rewards"]) for t in trajectories])
            rewards = [np.sum(t["rewards"]) for t in trajectories]
            mean = np.mean(rewards)
            std = np.std(rewards)
            mean_per_episode.append(mean)
            std_per_episode.append(std)
            print("Trial finished after {} timesteps and obtained {} Reward."
                  .format(timesteps, mean))

        #   estimate advantage for each step of a trial
            estimate_advantage(trajectories,
                               self.baseline, self.__gamma, self.__lambda)

        #   do NPG
            self.algorithm.do(trajectories, self.policy)

        #   Update NN of Valuefct
            estimate_value(trajectories, self.__gamma)
            self.baseline.train(trajectories)

        self.__plot(mean_per_episode, std_per_episode, int(200/10)) \
            if self.plot is True else None
        self.env.close()
        return False

    def __plot(self, mean, std, steps):
        x = np.arange(0, int(len(mean)-1), steps)
        y = np.asarray(mean)[x]
        e = np.asarray(std)[x]
        plt.errorbar(x, y, e, linestyle='None', marker='o')
        plt.plot(np.arange(len(mean)), mean, 'g')
        plt.show()
        return

    def benchmark_test(self):
        total_rewards = np.zeros(100)
        print("Starting Benchmark:")
        print("-------------------")
        for i_episode in range(100):
            print("Episode {}:".format(i_episode + 1))

            state = self.env.reset()
            state = state[None, :]
            # state = feature.featurize_state(state)
            for t in range(200):
                # env.render()
                action = self.policy.get_action(state, True)
                state, reward, done, info = self.env.step(action)
                state = state[None, :]
                # state = feature.featurize_state(state)
                total_rewards[i_episode] += reward
                if done:
                    print("Reward reached: ", total_rewards[i_episode])
                    print("Episode finished after {} timesteps.".format(
                        t + 1))
                    break
        average = np.sum(total_rewards) / 100
        print("Average Reward: ", average)
        if average >= 195:
            return True
        else:
            return False

