import numpy as np
import matplotlib.pyplot as plt
import torch as tr
import torch.nn.functional as F
import torch.nn as nn


#######################################
# Baseline
#######################################


class Baseline(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Baseline, self).__init__()
        # Calling Super Class's constructor
        self.linear1 = nn.Linear(input_dim, int(input_dim * 2))
        self.linear2 = nn.Linear(int(input_dim * 2), output_dim)
        # nn.linear is defined in nn.Module
        self.criterion = nn.MSELoss()
        self.optimiser = tr.optim.SGD(self.parameters(), lr=0.1)
        self.loss = 1

    def forward(self, x):
        # Here the forward pass is simply a linear function
        out = F.relu(self.linear1(x))
        return F.relu(self.linear2(out))

    def train(self, x, y):
        while (self.loss > 0.00001):
            # increase the number of epochs by 1 every time
            inputs = tr.from_numpy(x).float()
            labels = tr.from_numpy(y).float()
            self.optimiser.zero_grad()
            outputs = self.forward(inputs)
            self.loss = self.criterion(outputs, labels)
            self.loss.backward()  # back props
            self.optimiser.step()  # update the parameter
        print("loss: ", self.loss)
        return


#######################################
# Agent
#######################################


class Agent:
    def __init__(self, env, policy, algorithm, feature=None,
                 _lambda=0.95, _gamma=0.98, _delta=0.0001):
        self.policy = policy
        self.env = env
        self.feature = feature
        self.algorithm = algorithm
        self.__lambda = _lambda
        self.__gamma = _gamma
        self.__delta = _delta
        self.__values = []
        self.__eps = 1e-6
        self.baseline = Baseline(4 + 2, 1)
        np.random.seed(1)

    def train_policy(self):
        episodes = 10000
        rewards_per_episode = []
        time_per_episode = []
        for i_episode in range(episodes):
            print("begin episode: ", i_episode)

        #   roll out trajectories
            if i_episode >= episodes*0.99:
                trajectories = self.env.roll_out(self.policy,
                                                 features=self.feature,
                                                 amount=1, render=True)
            else:
                trajectories = self.env.roll_out(self.policy,
                                                 features=self.feature,
                                                 amount=1)

        #   get observations, action and rewards out of trials
            observations = np.concatenate([t["observations"]
                                           for t in trajectories])
            actions = np.concatenate([t["actions"] for t in trajectories])
            rewards = np.concatenate([t["rewards"] for t in trajectories])

            print("Trial finished after {} timesteps and obtained {} Reward."
                  .format(len(rewards), np.sum(rewards)))
            rewards_per_episode.append(sum(rewards))
            time_per_episode.append(len(rewards))

        #   update baseline
            inputs = np.concatenate(
                (observations, actions.reshape(-1, 1), rewards.reshape(-1, 1)), axis=1)
            if self.__values == []:
                self.__values = np.zeros_like(rewards).reshape(-1, 1)

            self.baseline.train(inputs, self.__values)
        #   estimate advantage for each step of a trial
            advantage = np.concatenate([self.__estimate_advantage(t)
                                        for t in trajectories])

        #   Do NPG
            #   vanilla gradient for each step
            vpg = np.zeros((len(self.policy.weights), len(actions)))
            for i in range(len(actions)):
               vpg[:, i:i+1] = self.policy.get_log_grad(observations[i:i+1, :], actions[i:i+1]).reshape(-1, 1, order='F')
            #print("vpg: ", vpg)

        #   compute g
            g = vpg @ advantage
            g /= vpg.shape[1]

        #   Fisher matrix
            fisher = vpg @ vpg.T
            fisher /= vpg.shape[1]
            fisher = np.diagonal(fisher)[None, :]
            fisher = np.diagflat(fisher)
            inv_fisher = self.__compute_inverse(fisher)

        #   update step
            nominator = g.T @ (inv_fisher @ g)
            learning_rate = np.sqrt(self.__delta / nominator)
            step = np.multiply(learning_rate, (inv_fisher @ g))
            self.policy.weights += step
            self.__values = np.concatenate(
                [self.estimate_empirical_return(t)
                 for t in trajectories]).reshape(-1, 1)

            print("finish episode:", i_episode, "\n")

        plt.plot(np.arange(len(rewards_per_episode)), rewards_per_episode)
        plt.plot(np.arange(len(time_per_episode)), time_per_episode, 'g')
        plt.show()
        self.env.close()
        return False

    def __estimate_advantage(self, trajectory):
        observations = trajectory["observations"]
        actions = trajectory["actions"]
        rewards = trajectory["rewards"]
        inputs = np.concatenate(
            (observations, actions.reshape(-1, 1), rewards.reshape(-1, 1)), axis=1)
        values = self.baseline(tr.from_numpy(inputs).float())
        values = values.detach().numpy().squeeze()
        advantage = np.zeros_like(rewards)
        delta = np.zeros_like(rewards)
        delta[:-1] = rewards[:-1] - values[:-1] + self.__gamma * values[1:]
        if len(rewards) == 10000:
            delta[-1] = rewards[-1] - values[-1]\
                        + self.__gamma * values[-1]
        else:
            delta[-1] = rewards[-1] - values[-1]

        for i in range(len(delta) - 1, -1, -1):
            advantage[i] = delta[i] if i == len(delta) - 1 \
                else delta[i] + self.__gamma * self.__lambda * advantage[i + 1]
        # advantage = (advantage - np.mean(advantage)) / (
        #             np.std(advantage) + self.__eps)
        return advantage

    def __compute_inverse(self, matrix):
        u, s, v = np.linalg.svd(matrix)
        s = np.diag(s**-1)
        return v.T @ (s @ u.T)

    def benchmark_test(self):
        return False

    # TODO estiamte value and estimate advantage fct
    def estimate_empirical_return(self, trajectory):
        rewards = trajectory["rewards"]
        values = np.zeros_like(rewards)
        for i in range(len(values)-1, -1, -1):
            values[i] = rewards[i] if i == len(values)-1 \
                else rewards[i] + self.__gamma * values[i + 1]
            # values = (values - np.mean(values)) / (
            #           np.std(values) + self.__eps)
        return values

