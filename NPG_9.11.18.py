import gym
import numpy as np

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
W = np.asmatrix([1.0, 1.0, 1.0, 1.0])
b = 0
sigma = 0.01
delta = 0.01

# Define training setup
# --------------------------------
# N is the number of trajectories per iteration
# gamma is the discount factor
# T is the max number of steps in a single run of the simulation
# K is the number of iterations for training the algorithm
# V are the values for iteration k-1 (updated to V_k at the end iteration k)
# R are the Rewards for each state t in each trajectory n in iteration k
# A are the values of the advantage in iteration k (each row represents a trajectory with max T steps)
# Log_PI are the derivations for the policy towards the parameters in iteration k for each state, action
# lda is lambda for bias-variance tradeoff of generalized advantage estimation (GAE)
N = 10
gamma = 0.98
T = 200
K = 1000000
V = np.zeros(T)
R = np.zeros((N, T))
A = np.zeros((N, T))
Log_PI = np.zeros((N, 4, T))
lda = 0.95


# Choose an action based on higher probability using policy function
# TODO: Implement for arbitrary amount of actions
def choose_action(observation):
    # mu = observation * np.transpose(W) + sigma
    # if mu >= 0.5:
    if policy(observation, 1) >= policy(observation, 0):
        return 1
    else:
        return 0



def phi(observation, exp):
    print(observation)
    return

# Returns a probability for an action in the current state
# The expectation is based on the observation weighted by W plus the belief b
def policy(observation, action):
    # phi(observation, 2)
    mu = observation*np.transpose(W)
    return (1/np.sqrt(2*np.pi*sigma**2))*np.exp(-((action-mu)**2)/(2*(sigma**2)))


# Gradient of log(policy) -> Algorithm step 4
# Returns gradient vector in 1x6 for log(policy)
# Partial derivation towards each parameter
def log_policy_gradient(observation, action):
    log_p_g = np.zeros(6)
    mu = np.dot(observation, np.transpose(W))+b
    log_p_g[0:4] = np.multiply((np.transpose(observation)), (action-mu)/(sigma**2))
    log_p_g[4] = (1/(sigma**2))*(action-mu)
    log_p_g[5] = ((action-mu)**2 - sigma**2)/(sigma**3)
    return log_p_g


def log_policy_gradient2(observation, action):
    mu = np.dot(observation, np.transpose(W))
    log_p_g = np.multiply((action-mu)/(sigma**2), observation)
    return log_p_g


def compute_value(index, t):
    value = 0
    for i in range(T-t):
        value += (gamma**i)*R[index, i+t]
    return value


def compute_delta_function(reward, t):
    if t+1 < T:
        return reward + gamma*V[t+1] - V[t]
    else:
        return reward + 1 - V[t]


def compute_advantage(n):
    for t in range(T):
        for e in range(T-t):
            A[n, t] += ((gamma*lda)**e)*compute_delta_function(R[n, t+e], t+e)

    return


def compute_gradient(n, T):
    return np.dot(Log_PI[n, :, :], (A[n, :]))/T


def compute_fisher(n, T):
    return np.dot(Log_PI[n, :, :], np.transpose(Log_PI[n, :, :]))/T


# Gradient ascent step -> Algorithm step 7
def update_parameters(n, T):
    g = np.asmatrix(compute_gradient(n, T))
    fisher = compute_fisher(n, T)
    nominator = np.dot(g, np.linalg.inv(fisher)) * np.transpose(g)
    if nominator > 0:
        d = np.multiply(np.sqrt(delta/nominator), np.dot(np.linalg.inv(fisher), np.transpose(g)))
        global W, b, sigma
        W += np.transpose(d[0:4])
        # b += d[4]
        # sigma += d[5]
    else:
        print("Nominator < 0: ", nominator)
    return


# Iteration of a single trajectory (basically run through a single game)
# Each run has up to T steps
# Termination conditions are pole angle +/- 12, cart position +/- 2.4 or steps >T (given by gym)
def run_trajectory(n):
    observation = env.reset()
    for t in range(T):
        env.render()

        # Choose best suitable action based on current observation using gaussian distribution
        action = choose_action(observation)

        # Calculate derivation for log(Policy) in current state for chosen action
        Log_PI[n, :, t] = log_policy_gradient2(observation, action)

        # Execute action to get next state
        observation, reward, done, info = env.step(action)
        R[n, t] = reward

        # Stop trajectory if termination condition is met
        if done:
            print("Reward = ", np.sum(R[n, :]))
            print("Episode finished after {} timesteps.".format(t + 1))
            break
    return


def init_reward_value():
    observation = env.reset()
    for t in range(T):
        env.render()

        # Choose best suitable action based on current observation using gaussian distribution
        action = choose_action(observation)

        # Execute action to get next state
        observation, reward, done, info = env.step(action)
        R[0, t] = reward

        if done:
            break

        t_reached = int(np.sum(R[0, :]))
        for t in range(t_reached):
            V[t] = compute_value(0, t)
    return


# Train the algorithm using the simulation
def train_algorithm():
    init_reward_value()

    # do K iterations and update parameters in each iteration
    for i_episode in range(K):
        print("Episode {}:".format(i_episode+1))

        # reset Rewards and Advantages
        global R, A
        R = np.zeros((N, T))
        A = np.zeros((N, T))

        # run through n trajectories
        for n in range(N):
            run_trajectory(n)
            compute_advantage(n)

        # Find best trajectory based on Advantage
        A[0, :] = np.mean(A, 0)
        t_reached = int(np.sum(np.mean(R, 0)))

        # Compute policy gradient
        update_parameters(n, t_reached)

        print(W, ", ", b, ", ", sigma)
        # Approximate values for each state -> Algorithm step 8
        for t in range(T):
            V[t] = compute_value(n, t)

    return


def run_benchmark():
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
# result = run_benchmark()
# print(result)
env.close()

sigma = np.sqrt(0.2)
mu = 0
action = 0
print( (1/np.sqrt(2*np.pi*sigma**2))*np.exp(-((action-mu)**2)/(2*(sigma**2))) )
