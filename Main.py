import numpy as np
import gym
import quanser_robots
import matplotlib.pyplot as plt
from NES import NES
from Features import RbfFeatures, RBFs
from Environment import Environment
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


def get_action(state, w):
    x = np.dot(state, w)
    return 1 if x > 0.5 else 0


def fitness(env, w):

    eps = 1

    s = np.size(w, 0)
    f = np.zeros(s)

    seed = env.seed()[0]

    for k in range(s):

        env.seed(seed)
        total_reward = 0
        for i in range(eps):

            done = False
            obs = env.reset()

            while not done:
                # env.render()
                obs, reward, done, info = env.step(get_action(obs, w[k]))
                total_reward += reward

        f[k] = total_reward / eps
    # a = np.argmax(f)
    # print(f[a], w[a])

    return f / np.max(f), f


def run_benchmark(w, env, episodes=1000):
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


np.random.seed(1)
env = gym.make('CartPole-v0')
nes = NES(env, 1, 0.35, 50, max_iter=200)
obs_space = len(env.observation_space.low)
# w, s = nes.do(fitness, np.zeros(obs_space), np.ones(obs_space))
w, s = nes.optimize(fitness, np.zeros(obs_space), np.ones(obs_space))
print(w, s)

# w = [2.518688, 0.29741, 21.013169, 39.44757]
# w = [-0.174842, 1.420497, 0.941925, 1.6674]
# w = [2.97329, 1.058639, 20.500426, 40.637915]

run_benchmark(w, env)
render(w, env)
