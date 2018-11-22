import numpy as np
import gym
import matplotlib.pyplot as plt
import torch as tr
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from Softmax_policy import Policy

#######################################
# NPG using Softmax Policy
# Value is estimated with multiple regression
#######################################


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
W1 = np.zeros((1, 4))*0.0
W2 = np.zeros((1, 4))*0.0
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
policy = Policy(4, 2)

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
#
#
# def softmax_policy(observation, action, W):
#     obs = np.zeros(len(observation))
#     W = np.append(W, 1)
#     for i, ele in enumerate(observation):
#         obs[i] = ele
#     phi = np.append(obs, action)
#     return np.exp(np.dot(phi, np.transpose(W)))


def compute_advantage(transitions):
    advantage = np.zeros(len(transitions))
    for i, transition in enumerate(transitions):
        for remainingsteps in range(len(transitions)-i-1):
            advantage[i] += ((gamma_*lambda_)**remainingsteps) * transition[2]
    advantage = (advantage - np.mean(advantage))/(np.std(advantage) + 0.0001)
    return advantage


def log_policy_gradient_softmax(transitions):
    log_p_gradient = np.random.sample((8, len(transitions)))
    for i, transition in enumerate(transitions):
        observation = transition[0]
        action = transition[1]
        obs = np.zeros(len(observation))
        for i, ele in enumerate(observation):
            obs[i] = ele
        phi = np.transpose(np.append(obs, action))
        a1 = softmax_policy(observation, 1.0, W1)
        a2 = softmax_policy(observation, 0.0, W2)
        p1 = a1 / (a1 + a2)
        p2 = a2 / (a1 + a2)
        phi1 = np.transpose(np.append(obs, 1.0))
        phi2 = np.transpose(np.append(obs, 0.0))
        if action == 1:
            log_p_gradient[0:4, i] = (phi - phi1*p1 - phi2*p2)[0:4]
        else:
            log_p_gradient[4:8, i] = (phi - phi1*p1 - phi2 * p2)[0:4]
    return log_p_gradient


def compute_gradient(log_p_gradient, advantage):
    return np.dot(log_p_gradient, advantage)/np.size(log_p_gradient, 1)


def compute_fisher(log_p_gradient):
    return np.dot(log_p_gradient, np.transpose(log_p_gradient))/np.size(log_p_gradient, 1)


# Gradient ascent step -> Algorithm step 7
def update_parameters(log_p_gradient, advantage):
    g = np.asmatrix(compute_gradient(log_p_gradient, advantage))
    fisher = compute_fisher(log_p_gradient)
    nominator = np.dot(g, np.linalg.inv(fisher)) * np.transpose(g)
    if nominator > 0:
        d = np.multiply(np.sqrt(delta/nominator), np.dot(np.linalg.inv(fisher), np.transpose(g)))
        global W1, W2, sigma
        W1 += np.transpose(d[0:4])
        W2 += np.transpose(d[4:8])
    else:
        print("Nominator < 0: ", nominator)
    return


# Iteration of a single trajectory (basically run through a single game)
# Each run has up to T steps
# Termination conditions are pole angle +/- 12, cart position +/- 2.4 or steps >T (given by gym)
def run_trajectory():
    observation = env.reset()
    env.seed(1995)
    transitions = []
    for t in range(T):
        env.render()

        # Choose best suitable action based on current observation using gaussian distribution
        action = policy.get_action(observation)

        old_observation = observation

        # Execute action to get next state
        observation, reward, done, info = env.step(action)
        global Reward
        Reward += reward
        # Stop trajectory if termination condition is met
        if done:
            # print("Reward = ", np.sum(R[n, :]))
            if t < 199:
                transitions.append((old_observation, action, -10))
            else:
                transitions.append((old_observation, action, reward))
            print("Trial finished after {} timesteps.".format(t + 1))
            break
        else:
            transitions.append((old_observation, action, reward))
    return transitions


# Train the algorithm using the simulation
def train_algorithm():
    transitions = run_trajectory()
    reward_per_episode = np.empty(0)
    mean_until_episode = np.empty(0)

    optimal_counter = 0

    # do K iterations and update parameters in each iteration
    for i_episode in range(K):
        global Reward
        Reward = 0
        print("Episode {}:".format(i_episode+1))

        transitions = run_trajectory()
        advantage = compute_advantage(transitions)
        log_p_gradient = log_policy_gradient_softmax(transitions)

        reward_per_episode = np.append(reward_per_episode, Reward)
        mean_until_episode = np.append(mean_until_episode, np.mean(reward_per_episode))
        print("Mean reward so far: ", mean_until_episode[i_episode])

        if Reward == 200:
            optimal_counter += 1
        else:
            optimal_counter = 0

        if optimal_counter == 5:
            print("Assuming optimal parameters!")
            break;

        update_parameters(log_p_gradient, advantage)

        if i_episode % 10 == 0:
            print(W1)
            print(W2)

    plt.plot(np.arange(K), reward_per_episode, 'r')
    plt.plot(np.arange(K), mean_until_episode, 'b')
    plt.ylabel("Average Reward per Episode")
    plt.xlabel("Episodes")
    plt.show()
    return


def run_benchmark():
    total_rewards = np.zeros(100)
    print("Starting Benchmark:")
    print("-------------------")
    for i_episode in range(100):
        print("Episode {}:".format(i_episode+1))

        observation = env.reset()
        for t in range(T):
            env.render()
            action = policy.get_action(observation)
            observation, reward, done, info = env.step(action)
            total_rewards[i_episode] += reward
            if done:
                print("Reward reached: ", total_rewards[i_episode])
                print("Episode finished after {} timesteps.".format(t + 1))
                break
    average = np.sum(total_rewards)/100
    print("Average Reward: ", average)
    if average >= 195:
        return True
    else:
        return False


print(W1)
print(W2)
train_algorithm()
print(W1)
print(W2)
passed = run_benchmark()
print(passed)
env.close()
