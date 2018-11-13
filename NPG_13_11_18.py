import gym
import numpy as np
from Multiple_Regression import MultipleRegression
import matplotlib.pyplot as plt

# Cart Pole Simulation
# --------------------------------
# Observations: 1. Cart Position   2. Cart Velocity    3. Pole Angle   4. Pole Velocity at Tip
# Actions: 1. Left  2. Right
env = gym.make('CartPole-v0')


# Define Parameters [W, b, sigma]
# --------------------------------
# W consists of 4 weights - one for each observation
# b is the belief that action a leads to state s' with observation o
# sigma is the variance in the gaussian distribution
# delta is the normalized step size of the parameter update
W = np.multiply(np.ones((1, 4)), np.random.sample(1))
b = 0
sigma = np.sqrt(0.02)
delta = 0.001

# Define training setup
# --------------------------------
# gamma is the discount factor
# T is the max number of steps in a single run of the simulation
# K is the number of iterations for training the algorithm
# V are the values for iteration k-1 (updated to V_k at the end iteration k)
# R are the Rewards for each state t in each trajectory n in iteration k
# A are the values of the advantage in iteration k (each row represents a trajectory with max T steps)
# Log_PI are the derivations for the policy towards the parameters in iteration k for each state, action
# lda is lambda for bias-variance tradeoff of generalized advantage estimation (GAE)
T = 200
K = 50
lambda_ = 0.95
gamma_ = 0.98

Reward = 0


# Choose an action based on higher probability using policy function
# TODO: Implement for arbitrary amount of actions
def choose_action(observation):
    # mu = observation * np.transpose(W) + sigma
    # if mu >= 0.5:
    if policy(observation, 1) >= policy(observation, 0):
        return 1
    else:
        return 0


# Returns a probability for an action in the current state
# The expectation is based on the observation weighted by W plus the belief b
def policy(observation, action):
    mu = np.dot(np.asmatrix(observation), np.transpose(W))
    return (1/np.sqrt(2*np.pi*sigma**2))*np.exp(-((action-mu)**2)/(2*(sigma**2)))


def compute_delta_function(estimator, curr_transition, next_transition):
    curr_inputs, curr_reward = curr_transition
    next_inputs = next_transition[0]
    return curr_reward - estimator.predictValue(curr_inputs) + estimator.predictValue(next_inputs)


def compute_advantage(estimator, transitions):
    advantage = np.zeros(len(transitions))
    for i, transition in enumerate(transitions):
        for remainingsteps in range(len(transitions)-i-1):
            curr_transition = transitions[i+remainingsteps]
            next_transution = transitions[i+remainingsteps+1]
            delta = compute_delta_function(estimator, curr_transition, next_transution)
            advantage[i] += ((gamma_*lambda_)**remainingsteps) * delta
    return advantage


def log_policy_gradient(transitions):
    log_p_gradient = np.zeros((4, len(transitions)))
    for i, transition in enumerate(transitions):
        observation = transition[0][0:4]
        action = transition[0][4]
        mu = np.dot(observation, np.transpose(W))
        log_p_gradient[:, i] = np.multiply((action-mu)/(sigma**2), observation)
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
        global W, b, sigma
        W += np.transpose(d[0:4])
    else:
        print("Nominator < 0: ", nominator)
    return


def findValueEstimator(transitions):
    x = np.zeros((len(transitions), 5))
    y = np.zeros((len(transitions), 1))
    for i, transition in enumerate(transitions):
        inputs, reward = transition
        for ii, element in enumerate(inputs):
            x[i, ii] = element
        global Reward
        Reward += reward
        for remainingsteps in range(len(transitions) - i):
            y[i, 0] += (gamma_**remainingsteps) * transitions[i+remainingsteps][1]
    estimator = MultipleRegression(x, y)
    c = estimator.multiple_reg()
    # plt.plot(np.arange(1000), c, 'r')
    # plt.show()
    return estimator


# Iteration of a single trajectory (basically run through a single game)
# Each run has up to T steps
# Termination conditions are pole angle +/- 12, cart position +/- 2.4 or steps >T (given by gym)
def run_trajectory():
    observation = env.reset()
    transitions = []
    for t in range(T):
        env.render()

        # Choose best suitable action based on current observation using gaussian distribution
        action = choose_action(observation)

        old_observation = observation

        # Execute action to get next state
        observation, reward, done, info = env.step(action)
        transitions.append((np.append(old_observation, action), reward))

        # Stop trajectory if termination condition is met
        if done:
            # print("Reward = ", np.sum(R[n, :]))
            transitions.append((np.append(observation, choose_action(observation)), 0))
            print("Episode finished after {} timesteps.".format(t + 1))
            break
    return transitions


# Train the algorithm using the simulation
def train_algorithm():
    transitions = run_trajectory()
    estimator = findValueEstimator(transitions)
    reward_per_Episode = np.zeros(K)
    # do K iterations and update parameters in each iteration
    for i_episode in range(K):
        print("Episode {}:".format(i_episode+1))

        transitions = run_trajectory()

        advantage = compute_advantage(estimator, transitions)

        log_p_gradient = log_policy_gradient(transitions)

        update_parameters(log_p_gradient, advantage)

        estimator = findValueEstimator(transitions)

        global Reward
        reward_per_Episode[i_episode] = Reward
        Reward = 0

    plt.plot(np.arange(K), reward_per_Episode, 'r')
    plt.show()
    return


train_algorithm()
env.close()

