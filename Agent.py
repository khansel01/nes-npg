import numpy as np
import matplotlib.pyplot as plt
from Estimate_advantage import estimate_advantage
from Estimate_value import estimate_value
import time

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

    def train_policy(self, episodes, amount: int=1):
        mean_per_episode = []
        std_per_episode = []
        time_per_episode = []
        for i_episode in range(episodes):
            print("\nbegin episode: ", i_episode)
            T0 = time.time()

        #   roll out trajectories
        #     if i_episode == 3:
        #         self.render = True
            trajectories = self.env.roll_out(self.policy, amount=amount,
                                             render=self.render)

        #   log data
            timesteps = np.mean([len(t["rewards"]) for t in trajectories])
            rewards = [np.sum(t["rewards"]) for t in trajectories]
            mean = np.mean(rewards)
            std = np.std(rewards)
            mean_per_episode.append(mean)
            std_per_episode.append(std)
            time_per_episode.append(timesteps)
            print("Trial finished after {} timesteps and obtained {} Reward."
                  .format(timesteps, mean))

            t0 = time.time()
        #   estimate advantage for each step of a trial
            estimate_advantage(trajectories,
                               self.baseline, self.__gamma, self.__lambda)

        #   Update policy
            self.algorithm.do(trajectories, self.policy)
            t1 = time.time()
            print("Update policy : {}".format(t1 - t0))

        #   Update critic
            t0 = time.time()
            estimate_value(trajectories, self.__gamma)
            self.baseline.train(trajectories)
            t1 = time.time()
            print("Update baseline : {}".format(t1 - t0))

            T1 = time.time()
            print("Do Episode : {}".format(T1 - T0))

        self.__plot(mean_per_episode, std_per_episode,
                    time_per_episode, int(200/10)) \
            if self.plot is True else None
        self.env.close()
        return False

    def __plot(self, mean, std, time, steps):
        plt.subplot(2, 1, 1)
        x = np.arange(0, int(len(mean)-1), steps)
        y = np.asarray(mean)[x]
        e = np.asarray(std)[x]
        plt.errorbar(x, y, e, linestyle='None', marker='o')
        plt.plot(np.arange(len(mean)), mean, 'g')
        plt.subplot(2, 1, 2)
        plt.plot(np.arange(len(time)), time, 'g')
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
            for t in range(200):
                # env.render()
                action = self.policy.get_action(state[0], True)
                state, reward, done, info = self.env.step(int(action))
                state = state[None, :]
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

