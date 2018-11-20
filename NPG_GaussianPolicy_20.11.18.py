import gym
import numpy as np
from Polynomial_Regression import PolynomialRegression
import matplotlib.pyplot as plt

#######################################
# NPG using Gaussian Policy
# Value is estimated with multiple regression
#######################################


# Cart Pole Simulation
# --------------------------------
# Observations: 1. Cart Position   2. Cart Velocity    3. Pole Angle   4. Pole Velocity at Tip
# Actions: 1. Left  2. Right
env = gym.make('CartPole-v0')


# Define Parameters [W, b, sigma]
# --------------------------------
# W consists of 4 weights - one for each observation
# sigma is the variance in the gaussian distribution
# delta is the normalized step size of the parameter update
# W1 = np.multiply(np.ones((1, 4)), np.random.sample(1))
# W2 = np.multiply(np.ones((1, 4)), np.random.sample(1))
W1 = np.random.randint(1, 10, (1, 4))*1.0
W2 = np.random.randint(1, 10, (1, 4))*1.0
sigma = np.sqrt(0.02)
delta = 0.05

# Define training setup
# --------------------------------
# gamma is the discount factor
# T is the max number of steps in a single run of the simulation
# K is the number of iterations for training the algorithm
# Log_PI are the derivations for the policy towards the parameters in iteration k for each state, action
# lda is lambda for bias-variance tradeoff of generalized advantage estimation (GAE)
T = 1000
K = 400
lambda_ = 0.95
gamma_ = 0.98
baseline = 1
Reward = 0


# Choose an action based on higher probability using policy function
# TODO: Implement for arbitrary amount of actions
def choose_action(observation):
    a1 = policy(observation, W1, sigma, 1)  # +policy(observation, W2, sigma, 1)
    a2 = policy(observation, W2, sigma, 0)  # +policy(observation, W1, sigma, -1)
    # print(a1, a2)
    if a1 >= a2:
        return 1
    else:
        return 0


# Returns a probability for an action in the current state
# The expectation is based on the observation weighted by W plus the belief b
def policy(observation, W, sigma, action):
    mu = np.dot(np.asmatrix(observation), np.transpose(W))
    return (1/np.sqrt(2*np.pi*sigma**2))*np.exp(-((action-mu)**2)/(2*(sigma**2)))


def compute_delta_function(estimator, curr_transition, next_transition):
    curr_obs, curr_action, curr_reward = curr_transition
    next_obs = next_transition[0]
    return curr_reward - estimator.predict_value(curr_obs) + estimator.predict_value(next_obs)


def compute_advantage(estimator, transitions):
    advantage = np.zeros(len(transitions))
    for i, transition in enumerate(transitions):
        for remainingsteps in range(len(transitions)-i-1):
            curr_transition = transitions[i+remainingsteps]
            next_transition = transitions[i+remainingsteps+1]
            delta_function = compute_delta_function(estimator, curr_transition, next_transition)
            advantage[i] += ((gamma_*lambda_)**remainingsteps) * delta_function
    return advantage


def log_policy_gradient(transitions):
    log_p_gradient = np.random.sample((8, len(transitions)))
    for i, transition in enumerate(transitions):
        observation = transition[0]
        action = transition[1]
        if action == 1:
            mu = np.dot(observation, np.transpose(W1))
            log_p_gradient[0:4, i] = np.multiply((action - mu) / (sigma ** 2), observation)
        else:
            mu = np.dot(observation, np.transpose(W2))
            log_p_gradient[4:8, i] = np.multiply((action - mu) / (sigma ** 2), observation)
        # log_p_gradient[0:4, i] = np.multiply((action - mu) / (sigma ** 2), observation)
        # log_p_gradient[4, i] = ((action - mu) ** 2 - sigma ** 2) / (sigma ** 3)

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


def findValueEstimator(transitions):
    x = np.zeros((len(transitions), 4))
    y = np.zeros((len(transitions), 1))
    for i, transition in enumerate(transitions):
        obs, actions, reward = transition
        for ii, element in enumerate(obs):
            x[i, ii] = element
        global Reward
        Reward += reward
        for remainingsteps in range(len(transitions) - i):
            y[i, 0] += (gamma_**remainingsteps) * transitions[i+remainingsteps][2]
    print("True y: ", y[0, 0])
    estimator = PolynomialRegression(x, y, pow(10, -6), 2)
    c = estimator.train()
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
        transitions.append((old_observation, action, reward))

        # Stop trajectory if termination condition is met
        if done & (t < 200):
            # print("Reward = ", np.sum(R[n, :]))
            # transitions.append((np.append(observation, choose_action(observation)), 0))
            print("Trial finished after {} timesteps.".format(t + 1))
            break
    return transitions


# Train the algorithm using the simulation
def train_algorithm():
    transitions = run_trajectory()
    estimator = findValueEstimator(transitions)
    reward_per_episode = np.empty(0)
    mean_until_episode = np.empty(0)
    global Reward
    Reward = 0
    # do K iterations and update parameters in each iteration
    for i_episode in range(K):
        print("Episode {}:".format(i_episode+1))

        transitions = run_trajectory()
        advantage = compute_advantage(estimator, transitions)
        log_p_gradient = log_policy_gradient(transitions)

        update_parameters(log_p_gradient, advantage)

        print("Estimation: ", estimator.predict_value(transitions[0][0]))
        estimator = findValueEstimator(transitions)

        reward_per_episode = np.append(reward_per_episode, Reward)
        mean_until_episode = np.append(mean_until_episode, np.mean(reward_per_episode))
        print("Mean reward so far: ", mean_until_episode[i_episode])
        Reward = 0

    plt.plot(np.arange(K), reward_per_episode, 'r')
    plt.plot(np.arange(K), mean_until_episode, 'b')
    plt.ylabel("Average Reward per Episode")
    plt.xlabel("Episodes")
    plt.show()
    return


print(W1)
print(W2)
train_algorithm()
print(W1)
print(W2)
env.close()