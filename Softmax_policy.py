import argparse
import gym
import numpy as np

import torch as tr
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as agrad
from torch.distributions import Categorical
from torch.distributions import Normal


class Policy(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(obs_dim, act_dim)
        self.saved_log_probs = []
        self.rewards = []
        self.advantage = []

    def get_action(self, observation):
        obs = tr.from_numpy(observation).float().unsqueeze(0)
        probs = self(obs)  # call forward function and get the probabilities for each action
        m = Categorical(probs)  # create a categorical distribution of "probs"
        action = m.sample()  # choose an action based on the distribution "m"
        self.saved_log_probs.append(m.log_prob(action))
        return action.item()  # get the single value as python number if tensor contains only single value

    def forward(self, x):
        action_scores = self.affine1(x)
        return F.softmax(action_scores, dim=1)

    def clear(self):
        del self.rewards[:]
        del self.advantage[:]
        return


# # Choose an action based on higher probability using policy function
# # TODO: Implement for arbitrary amount of actions
# def choose_action(observation):
#     a1 = softmax_policy(observation, 1.0, W1)
#     a2 = softmax_policy(observation, 0.0, W2)
#     # print("Exp: ", a1, a2)
#     p1 = a1 / (a1+a2)
#     p2 = a2 / (a1+a2)
#     probs = np.matrix([p1, p2])
#     probs = tr.from_numpy(probs)
#     m = Categorical(probs)
#     action = m.sample()
#     return action.item()
#     # print("Prob: ", p1, p2)
#     if p1 >= p2:
#         return 1
#     else:
#         return 0

# def softmax_policy(observation, action, W):
#     obs = np.zeros(len(observation))
#     W = np.append(W, 1)
#     for i, ele in enumerate(observation):
#         obs[i] = ele
#     phi = np.append(obs, action)
#     return np.exp(np.dot(phi, np.transpose(W)))

def main():
    # Cart Pole Simulation
    # --------------------------------
    # Observations: 1. Cart Position   2. Cart Velocity    3. Pole Angle   4. Pole Velocity at Tip
    # Actions: 1. Left  2. Right
    env = gym.make('CartPole-v0')
    env.seed(1995)

    # Define Parameters [W, b, sigma]
    # --------------------------------
    # W consists of 4 weights - one for each observation
    # delta is the normalized step size of the parameter update
    W1 = np.zeros((1, 4)) * 0.0
    W2 = np.zeros((1, 4)) * 0.0
    delta = 0.05

    # Define training setup
    # --------------------------------
    # gamma is the discount factor
    # T is the max number of steps in a single run of the simulation
    # K is the number of iterations for training the algorithm
    T = 200
    K = 1000
    lambda_ = 0.95
    gamma_ = 0.98
    Reward = 0


    observation = env.reset()
    env.seed(1995)
    env.render()
    observation, reward, done, _ = env.step(1)
    for i in range(2):
        print("ob: ", np.transpose(observation))
        policy = Policy(4, 2)
        action = policy.get_action(observation)
        for p in policy.parameters():
            print(i, "parameter: ", p)
            print(":")
        env.render()
        observation, reward, done, _ = env.step(action)
    env.close()

if __name__ == '__main__':
    main()