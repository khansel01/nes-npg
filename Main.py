import torch as tr
import numpy as np
import gym
import quanser_robots
import matplotlib.pyplot as plt
from NES import *
from utilities.Environment import Environment
from Agent import Agent

#######################################
# Environment
#######################################


# np.random.seed(1)
# gym_env = 'CartPole-v0'
# env = Environment(gym_env)
# policy = SoftmaxPolicy(env)
# feature = None
# # gym_env = 'CartpoleSwingShort-v0'
# # env = Environment(gym_env)
# # feature = RbfFeatures(env, SoftmaxPolicy(env))
# # policy = GaussianPolicy(env)
# algorithm = NPG(0.001)
# agent = Agent(env, policy, algorithm, feature=feature)
# agent.train_policy(200)
# # agent.benchmark_test()
# # env.roll_out(policy, feature, 1, True)

# np.random.seed(1)
# env = gym.make('CartpoleSwingShort-v0')
#env = gym.make('Pendulum-v0')
gym_env = 'Pendulum-v0'
env = Environment(gym_env)
episodes = 50
# dim = 17
obs_space = env.obs_dim


def get_action(state, w):
    # state = np.reshape(np.append(state, 1), (-1, 1))
    state = np.reshape(state, (-1, 1))
    x = np.dot(np.reshape(state*np.transpose(state), (1, -1)), w)
    return [np.sum(x)]


def fitness(policy, env, w, n_roll_outs: int = 1):

    samples = np.size(w, 0)
    f = np.zeros(samples)

    seed = env.get_seed()

    for s in range(samples):

        policy.set_parameters(w[s])

        env.seed(seed)

        trajectories: dict = env.roll_out(policy, n_roll_outs=n_roll_outs)
        rewards = np.concatenate([t["rewards"]
                                  for t in trajectories]).reshape(-1, 1)

        f[s] = rewards.sum() / n_roll_outs

    return f, f


def run_benchmark(w, env, episodes=100):
    total_rewards = np.zeros(episodes)
    print("Starting Benchmark:")
    print("-------------------")
    for i_episode in range(episodes):
        # print("Episode {}:".format(i_episode + 1))

        state = env.reset()
        done = False
        while not done:
            action = get_action(state, w)
            # print(action)
            state, reward, done, info = env.step(action)
            total_rewards[i_episode] += reward
        print("Reward reached: ", total_rewards[i_episode])

    average = np.sum(total_rewards) / episodes
    print("-------------------")
    print("Average reward: ", average)
    print("Min reward:", min(total_rewards))
    print("Max reward:", max(total_rewards))
    return average


def render(w, env, seed=False):
    if seed:
        env.seed(0)

    state = env.reset()
    done = False
    r = 0
    while not done:
        env.render()
        action = get_action(state, w)
        state, reward, done, info = env.step(action)
        r += reward
    return r


# random_policy = PolicyNN(env, hidden_dim=(8, 8))
random_policy = PolicySquare(env)
nes = NES(env, 1, 0.3, 50, max_iter=episodes)
mu = np.zeros(random_policy.length)
sigma = np.ones(random_policy.length)
w, s, r, e = nes.do(fitness, mu, sigma, random_policy)


print(w, s)

x = np.arange(episodes)

print(np.shape(x), np.shape(r), np.shape(e))
plt.errorbar(x, r, e, linestyle='-', marker='x', markeredgecolor='red')

plt.show()

run_benchmark(w, env)
# print(render(w, env))

