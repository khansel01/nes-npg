import numpy as np
import gym
import matplotlib.pyplot as plt
# import torch as tr
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.distributions import Categorical
# from Softmax_policy import Policy

#######################################
# NPG using Softmax Policy
#######################################


# Cart Pole Simulation
# --------------------------------
# Observations: 1. Cart Position   2. Cart Velocity    3. Pole Angle   4. Pole Velocity at Tip
# Actions: 1. Left  2. Right
env = gym.make('CartPole-v0')
env.seed(0)
np.random.seed(0)

# Define Parameters [W, b, sigma]
# --------------------------------
# W consists of 4 weights - one for each observation
# delta is the normalized step size of the parameter update
W1 = np.ones((1, 4))*1.0
W2 = np.ones((1, 4))*-1.0
delta = 0.000025

# Define training setup
# --------------------------------
# gamma is the discount factor
# T is the max number of steps in a single run of the simulation
# K is the number of iterations for training the algorithm
T = 200
K = 5000
lambda_ = 0.95
gamma_ = 0.98
Reward = 0
eps = np.finfo(np.float32).eps.item()
Value = np.empty((1, T))

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

# Choose an action based on higher probability using policy function
# TODO: Implement for arbitrary amount of actions
def choose_action(observation):
    a1 = softmax_policy(observation, W1)
    a2 = softmax_policy(observation, W2)
    p1 = a1 / (a1+a2)
    p2 = a2 / (a1+a2)
    return p1, p2


def softmax_policy(observation, W):
    obs = np.zeros(len(observation))
    for i, ele in enumerate(observation):
        obs[i] = ele
    return np.sum(np.exp(np.dot(obs, np.transpose(W))))


def compute_last_ep_values(transitions):
    values = np.zeros((1, T))
    for i, transition in enumerate(transitions):
        for remainingsteps in range(len(transitions)-i-1):
            values[0, i] += (gamma_**remainingsteps) * transition[2]
    # values = (values - np.mean(values))/(np.std(values) + eps)
    return values

def compute_advantage(transitions, old_transitions):
    advantage = np.zeros(len(transitions))
    values = compute_last_ep_values(old_transitions)
    global Value
    Value = np.concatenate((Value, values), axis=0)
    values = np.mean(Value, axis=0)
    for i, transition in enumerate(transitions):
        for remainingsteps in range(len(transitions)-i-1):
            # delta_func = transitions[i+remainingsteps][2] - values[i+remainingsteps] + gamma_*values[i+remainingsteps+1]
            # advantage[i] += ((gamma_*lambda_)**remainingsteps) * delta_func
            advantage[i] += (gamma_**remainingsteps)*transitions[i+remainingsteps][2]
    advantage = (advantage - np.mean(advantage))/(np.std(advantage) + eps)
    return advantage


def log_policy_gradient_softmax(transitions):
    log_p_gradient = np.random.sample((8, len(transitions)))
    for i, transition in enumerate(transitions):
        observation = transition[0]
        action = transition[1]
        obs = np.zeros(len(observation))
        for i, ele in enumerate(observation):
            obs[i] = ele
        phi = np.transpose(obs)
        a1 = softmax_policy(observation, W1)
        a2 = softmax_policy(observation, W2)
        p1 = a1 / (a1 + a2)     # Action == 1
        p2 = a2 / (a1 + a2)     # Action == 0
        log_p_gradient[0:4, i] = (phi*action - phi*p1)
        log_p_gradient[4:8, i] = (phi*(1-action) - phi*p2)
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
    env.seed(0)
    transitions = []
    for t in range(T):
        env.render()

        # Choose best suitable action based on current observation using gaussian distribution
        p1, p2 = choose_action(observation)
        action = np.random.choice([1, 0], p=[p1, p2])

        old_observation = observation

        # Execute action to get next state
        observation, reward, done, info = env.step(action)
        transitions.append((old_observation, action, reward))

        global Reward
        Reward += reward
        # Stop trajectory if termination condition is met
        if done:
            print("Trial finished after {} timesteps.".format(t + 1))
            break

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

        old_transitions = transitions

        transitions = run_trajectory()
        advantage = compute_advantage(transitions, old_transitions)
        log_p_gradient = log_policy_gradient_softmax(transitions)

        reward_per_episode = np.append(reward_per_episode, Reward)
        mean_until_episode = np.append(mean_until_episode, np.mean(reward_per_episode))
        print("Mean reward so far: ", mean_until_episode[i_episode])

        if Reward == T:
            optimal_counter += 1
        else:
            optimal_counter = 0

        if optimal_counter == 10:
            print("Assuming optimal parameters!")
            break;

        update_parameters(log_p_gradient, advantage)

        if i_episode % 10 == 0:
            print(W1)
            print(W2)

    plt.plot(np.arange(len(reward_per_episode)), reward_per_episode, 'r')
    plt.plot(np.arange(len(mean_until_episode)), mean_until_episode, 'b')
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
            p1, p2 = choose_action(observation)
            if p1 >= p2:
                action = 1
            else:
                action = 0
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
