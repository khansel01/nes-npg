import numpy as np
import gym
import matplotlib.pyplot as plt
import Features
import torch as tr
import torch.nn.functional as F
import torch.nn as nn
from cg_solve import cg_solve
from sklearn.gaussian_process import GaussianProcessRegressor

#######################################
# NPG using Softmax Policy
#######################################


class ValueFunction(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(ValueFunction, self).__init__()
        # Calling Super Class's constructor
        self.linear1 = nn.Linear(input_dim, int(input_dim*2))
        self.linear2 = nn.Linear(int(input_dim*2), output_dim)
        # nn.linear is defined in nn.Module
        self.criterion = nn.MSELoss()
        self.optimiser = tr.optim.SGD(self.parameters(), lr=0.1)
        self.loss = 1

    def forward(self, x):
        # Here the forward pass is simply a linear function
        out = F.relu(self.linear1(x))
        return F.relu(self.linear2(out))

    def train(self, x, y):
        while(self.loss>0.00001):
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


class NPG:
    # Define training setup
    # --------------------------------
    # gamma is the discount factor.
    # lambda is the bias-variance tradeoff for the advantage function.
    # T is the max number of steps in a single run of the simulation.
    # K is the number of episodes for training the algorithm.
    # delta is the normalized step size of the parameter update.

    def __init__(self, env, policy, episodes):
        np.random.seed(1)
        self.env = env
        self.policy = policy
        self.__n_Actions = 2  # env.action_space.n
        self.__K = episodes
        self.__lambda = 0.95
        self.__gamma = 0.98
        self.__delta = 0.05
        self.__eps = np.finfo(np.float32).eps.item()
        self.__values = []
        self.W = np.random.sample((200, 2))
        # self.feature = Features.RbfFeatures(env)
        # self.feature = Features.RBFs(5, 200)
        self.regressor = GaussianProcessRegressor()
        self.valuefunction = ValueFunction(202, 1)

    def train(self):
        rewards_per_episode = []
        timsteps_per_episode = []

        for i_episode in range(self.__K):
            print("Episode ", i_episode, ":")
            log_gradients = []
            transitions = []
            rewards = []
            states = []

            state = np.asarray(self.env.reset())
            # state = state[None, :]

            #TODO feature
            # state = self.feature.featurize_state(state)
            # state = self.feature.get_rbfs(_state)

            self.env.seed(0)
            timestep = 0
            flag = True
            while(True):
                if i_episode >= 0.99*self.__K:
                    self.env.render()
                # self.env.render()
                old_state = state

                action = self.policy.get_action(state)
                action = np.clip(action, -18, 18)

                state, reward, done, _ = self.env.step(np.asarray(action))
                state = np.asarray(state)
                # state = state[None, :]

                # TODO feature
                # state = self.feature.featurize_state(state)
                # self.feature.update_v(_state, _old_state)
                # state = self.feature.get_rbfs(_state)

                log_grad = self.policy.get_log_grad(old_state, action)

                log_gradients.append(log_grad.reshape((-1, 1), order='F'))
                rewards.append(reward)
                timestep -= 1
                transition = np.append(old_state, np.append(action, reward))
                transitions.append(transition)

                if done:
                    print("Trial finished after {} timesteps and obtained {} Reward."
                          .format(timestep, np.sum(rewards)))
                    flag = False
                    break
            print("Trial finished after {} timesteps and obtained {} Reward."
                  .format(timestep, np.sum(rewards))) if flag else None

            # TODO: Value Fct in continious action space
            # With NN
            self.__values = np.zeros(len(rewards)) if self.__values == [] \
                else self.__values

            # self.valuefunction.train(np.asarray(transitions),
            #                          self.__values.reshape(-1, 1))

            # temp_values = self.valuefunction.forward(
            #     tr.from_numpy(np.asarray(transitions)).float())
            # self.__values = temp_values.detach().numpy().squeeze()

            self.__update_parameters(log_gradients, rewards)

            self.__values = self.__estimate_value(rewards)
            rewards_per_episode.append(np.sum(rewards))
            timsteps_per_episode.append(timestep)

            # self.feature.featurize_fit(states)

            # With GP
            # self.__values = self.regressor.predict(
            #     np.asarray(transitions)).squeeze()
            # self.__update_parameters(log_gradients, rewards)
            #
            # self.__values = self.__estimate_value(rewards)
            # self.regressor.fit(np.asarray(transitions),
            #                    self.__values.reshape(-1, 1))
            # rewards_per_episode.append(np.sum(rewards))
        return self.W, rewards_per_episode, timsteps_per_episode

    def __update_parameters(self, log_gradients, rewards):
        g = self.__compute_gradient(log_gradients, rewards)
        fisher = self.__compute_fisher(log_gradients)
        try:
            # inv_fisher = np.linalg.inv(fisher)
            inv_fisher = self.__compute_inverse(fisher)
            nominator = (g.T @ inv_fisher) @ g
            # NPG = cg_solve(fisher, g)
            # nominator = g.T @ NPG
            if nominator <= 0:
                print("Nominator <= 0: ", nominator)
            else:
                learning_rate = np.sqrt(self.__delta/nominator)
                step = np.multiply(learning_rate, (inv_fisher @ g))
                # step = np.multiply(learning_rate, NPG)

                c = step.T@fisher
                c = c@step
                if c > (self.__delta*(1 + 0.0001)):
                    print("condition: ", c, " > ", self.__delta)
                else:
                    self.policy.weights += step
        except np.linalg.LinAlgError:
            print("Skipping parameter update due to singular matrix.")
            pass
        return

    def __compute_inverse(self, matrix):
        u, s, v = np.linalg.svd(matrix)
        s = np.diag(s**-1)
        return v.T @ (s @ u.T)

    def __compute_gradient(self, log_g, rewards):
        g = 0
        advantage = self.__estimate_advantage(rewards)
        for i in range(len(log_g)):
            g += log_g[i] * advantage[i]
        return g/len(log_g)

    def __compute_fisher(self, log_g):
        f = sum([(lg.reshape(-1, 1) @ lg.reshape(-1, 1).T) for lg in log_g])
        f = np.diagonal(f)[None, :]
        f = np.diagflat(f)
        return f/len(log_g)

    def __estimate_value(self, rewards):
        value = np.zeros(len(rewards))
        for i in range(len(rewards)):
            value[i] = sum([r * (self.__gamma ** t) for t, r in
                                enumerate(rewards[i:])])
        # value = (value - np.mean(value)) / (np.std(value) + self.__eps)
        return value

    def __estimate_advantage(self, rewards):
        index = len(rewards) if len(rewards) >= len(self.__values)\
            else len(self.__values)
        values = self.__values@np.eye(len(self.__values), index)
        rewards = rewards@np.eye(len(rewards), index)
        advantage = np.zeros(index)
        for i in range(index):
            for remainingsteps in range(index - i - 1):
                delta_func = rewards[i+remainingsteps] - \
                             values[i+remainingsteps] + \
                             self.__gamma*values[i+remainingsteps]
                advantage[i] += ((self.__gamma * self.__lambda) **
                                 remainingsteps) * delta_func
        # advantage = (advantage - np.mean(advantage)) / \
        #          (np.std(advantage) + self.__eps)
        return advantage




