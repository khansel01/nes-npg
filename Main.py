import gym
import numpy as np

import torch as tr
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class Policy(nn.Module):
    def __init__(self, in_layer, out_layer, hid_layer=0):
        super(Policy, self).__init__()
        if hid_layer==0:
            self.affine1 = nn.Linear(in_layer, hid_layer)
        else:
            self.affine1 = nn.Linear(in_layer, hid_layer)
            self.affine2 = nn.Linear(hid_layer, out_layer)

        self.saved_log_probs = []
        self.rewards = []

    def get_action(self, observation):
        # tr.from_numpy() converts np.array to tensor of type float()
        obs = tr.from_numpy(observation).float().unsqueeze(0)
        probs = self(obs)  # call forward function and get the probabilities for each action
        m = Categorical(probs)  # create a categorical distribution of "probs"
        action = m.sample()  # choose an action based on the distribution "m"
        self.saved_log_probs.append(m.log_prob(action))  # estimate the gradient of log p(a, pi(s))
        return action.item()  # get the single value as python number if tensor contains only single value

    def forward(self, x):
        x = F.relu(self.affine1(x))
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)


def main():
    env = gym.make('CartPole-v0')
    episode = 0
    # get a machine epsilon for the given float type of float32
    eps = np.finfo(np.float32).eps.item()
    # transitions = [Observations, actions, rewards]
    transitions = []
    rewardss = []
    gamma = 0.97
    for i_episode in range(1):
        observation = env.reset()
        policy = Policy(4, 2, 5)
        for trans in range(200):
            env.render()
            action = policy.get_action(observation)
            observation, reward, done, _ = env.step(action)
            transitions.append([observation, action, reward])
            rewardss.append(reward)
            policy.rewards.append(reward)
            if done:
                break
        # estimate the advantages for each step
        advantage = 0
        policy_loss=[]
        advantages = []
        for r in rewardss[::-1]:
            advantage = r + gamma*advantage
            advantages.insert(0, advantage)
        advantages = tr.tensor(advantages)
        advantages = (advantages - advantages.mean())/(advantages.std()+eps)  # TODO: ???? What the hell ?

        print(advantages)
        print(policy.saved_log_probs)
        transitions.clear()
    env.close()


if __name__ == '__main__':
    main()
