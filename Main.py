import argparse
import gym
import numpy as np

import torch as tr
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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
        # n = Normal(*probs)
        action = m.sample()  # choose an action based on the distribution "m"
        # print(n.rsample())
        self.saved_log_probs.append(m.log_prob(action))  # estimate the gradient of log p(a, pi(s))
        return action.item()  # get the single value as python number if tensor contains only single value

    def forward(self, x):
        x = F.relu(self.affine1(x))
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)


def main():
    # env = gym.make('CartPole-v0')
    # TODO: vvvvvvvvvvvvvvvvvv
    parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor (default: 0.99)')
    parser.add_argument('--seed', type=int, default=543, metavar='N',
                        help='random seed (default: 543)')
    parser.add_argument('--render', action='store_true',
                        help='render the environment')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='interval between training status logs (default: 10)')
    args = parser.parse_args()

    env = gym.make('CartPole-v0')
    env.seed(args.seed)
    tr.manual_seed(args.seed)
    # TODO: ^^^^^^^^^^^^^^^^^^
    episode = 0
    # get a machine epsilon for the given float type of float32
    eps = np.finfo(np.float32).eps.item()
    # transitions = [Observations, actions, rewards]
    # transitions = []
    gamma = 0.97
    policy = Policy(4, 2, 128)
    optimizer = optim.Adam(policy.parameters(), lr=1e-2)    # TODO
    for i_episode in range(100):
        print(i_episode)
        observation = env.reset()        # TODO
        for trans in range(200):
            env.render()
            action = policy.get_action(observation)
            observation, reward, done, _ = env.step(action)
            # transitions.append([observation, action, reward])
            policy.rewards.append(reward)
            if done:
                print("Episode finished after {} timesteps.".format(trans + 1))
                break
        # estimate the advantages for each step
        advantage = 0
        advantages = []
        for r in policy.rewards[::-1]:
            advantage = r + gamma*advantage
            advantages.insert(0, advantage)
        advantages = tr.tensor(advantages)
        advantages = (advantages - advantages.mean())/(advantages.std()+eps)  # TODO: ???? What the hell ?

        # TODO: vvvvvvvvvvvvvvvvvv
        # g = 0
        # F = 0
        # T = 0
        # delta = 0.05
        # policy_loss = []
        # for log_prob, a in zip(policy.saved_log_probs, advantages):
        #     T +=1
        #     F += log_prob.item()*log_prob.item()
        #     g += log_prob.item()*a.item()
        #     policy_loss.append(-log_prob * a)
        # F /= T
        # F = F**(-1)
        # alpha = float(np.sqrt(delta/(F*g*g))*F*g)
        # for param_group in optimizer.param_groups:
        #     param_group["lr"] = alpha
        # optimizer.zero_grad()
        # policy_loss = tr.cat(policy_loss).sum()
        # policy_loss.backward()
        # optimizer.step()
        # del policy.saved_log_probs[:]
        # del policy.rewards[:]
        # TODO -------------------
        policy_loss = []
        for log_prob, a in zip(policy.saved_log_probs, advantages):
            policy_loss.append(-log_prob * a)
        optimizer.zero_grad()
        policy_loss = tr.cat(policy_loss).sum()
        policy_loss.backward()
        optimizer.step()
        del policy.saved_log_probs[:]
        del policy.rewards[:]
        # TODO: ^^^^^^^^^^^^^^^^^
        # transitions.clear()
    env.close()




if __name__ == '__main__':
    main()

