import gym
import numpy as np

# Cart Pole Simulation
# Observations: 1. Cart Position   2. Cart Velocity    3. Pole Angle   4. Pole Velocity at Tip
# Actions: 1. Left  2. Right

# Define Parameters [W, b, sigma]
# --------------------------------
# W consists of 4 weights - one for each observation
# b is the belief that action a leads to state s' with observation o
W = np.asmatrix([0.25, 0.25, 0.25, 0.25])
b = 0.5
sigma = 0.25

# Define training setup
# --------------------------------
# N is the number of trajectories per iteration
# gamma is the discount factor
# T is the max number of steps in a single run of the simulation
# K is the number of iterations for training the algorithm
# V are the values for iteration k-1 (updated to V_k at the end iteration k)
# A are the values of the advantage in iteration k (each row represents a trajectory with max T steps)
N = 20
gamma = 0.7
T = 200
K = 1
V = np.zeros(N)
A = np.zeros((N, T))


# Choose an action based on higher probability using policy function
def choose_action(observation):
    if policy(observation, 1) >= policy(observation, 0):
        return 1
    else:
        return 0


# Returns a probability for an action in                audibl777the current state
# The expectation is based on the observation weighted by W plus the belief b
def policy(observation, action):
    mu = observation*np.transpose(W)+b
    return (1/np.sqrt(2*np.pi*sigma**2))*np.exp(-((action-mu)**2)/2*sigma)


# gradient of log(policy) -> Algorithm step 4
def compute_policy_gradient():
    # TODO: implement gradient
    return


def compute_value():
    # TODO: implement calculation for value
    return


def compute_q_value():
    # TODO: implement Q-Value computation
    # Expected Reward as sum of reward for each action weighted by the probability of choosing it?
    # Q = E(R) + E( V(s',t+1) ) -> V(s',t+1) = compute_value(...)
    return


def compute_advantage():
    # is the total reward needed?
    # TODO: implement calculation for advantage
    # something like A = Q - V, where V is given from last iteration (see definitions at the top)
    # maybe as Vector, where each element contains advantage of a single step in trajectory
    # Q = compute_q_value(...)
    return


def compute_fisher():
    # TODO: implement fisher computation
    return


# Gradient ascent step -> Algorithm step 7
def update_parameters():
    # TODO: implement parameter update
    return


# Iteration of a single trajectory (basically run through a single game)
# Each run has up to T steps
# Termination conditions are pole angle +/- 12, cart position +/- 2.4 or steps >T (given by gym)
def run_trajectory(env):
    total_reward = 0    # Accumulated reward over all steps in a single trajectory
    observation = env.reset()
    for t in range(T):
        env.render()
        action = choose_action(observation)
        observation, reward, done, info = env.step(action)
        compute_policy_gradient()
        total_reward += reward
        if done:
            print("Reward = ", total_reward)
            print("Episode finished after {} timesteps.".format(t + 1))
            break
    return
    # TODO: is the order of action and states relevant for parameter update?
    # TODO: Return gradient of log policy for each state (or use global variable?)


# Train the algorithm using the simulation
def train_algorithm():
    env = gym.make('CartPole-v0')

    # do K iterations and update parameters in each iteration
    for i_episode in range(K):
        print("Episode {}:".format(i_episode+1))

        # run through n trajectories
        for n in range(20):
            run_trajectory(env)
        # At this point Advantage matrix and gradient of log policy should be done
        # TODO: algorithm steps 6-8

    return


def run_benchmark():
    env = gym.make('CartPole-v0')
    total_rewards = np.zeros(100)
    for i_episode in range(100):
        print("Episode {}:".format(i_episode+1))

        observation = env.reset()
        for t in range(T):
            env.render()
            action = choose_action(observation)
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


train_algorithm()
#result = run_benchmark()
#print(result)
