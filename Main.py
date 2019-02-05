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
# env = gym.make('CartpoleSwingShort-v0')
#env = gym.make('Pendulum-v0')
# gym_env = 'Pendulum-v0'
# gym_env = 'CartpoleSwingShort-v0'
gym_env = 'Qube-v0'
env = Environment(gym_env)
episodes = 1000
# dim = 17
obs_space = env.obs_dim()


def run_benchmark(policy, w, env, episodes=100):
    total_rewards = np.zeros(episodes)
    print("Starting Benchmark:")
    print("-------------------")
    for i_episode in range(episodes):
        # print("Episode {}:".format(i_episode + 1))

        state = env.reset()
        done = False
        policy.set_parameters(w)
        while not done:
            action = policy.get_action(state)
            # print(action)
            state, reward, done, info = env.step(np.asarray([action]))
            total_rewards[i_episode] += reward

        print(i_episode + 1,
              "Reward reached: ", total_rewards[i_episode])

    average = np.sum(total_rewards) / episodes
    print("-------------------")
    print("Average reward: ", average)
    print("Min reward:", min(total_rewards))
    print("Max reward:", max(total_rewards))
    return average


def render(policy, w, env, seed=False, step_size=1):
    if seed:
        env.seed(0)

    state = env.reset()
    done = False
    r = 0
    policy.set_parameters(w)
    step = 0
    while not done:
        env.render() if step % step_size == 0 else None
        action = policy.get_action(state)
        state, reward, done, info = env.step(np.asarray([action]))
        r += reward
        # print(action, state[4])
        step += 1
    return r


random_policy = PolicyNN(env, hidden_dim=(obs_space, obs_space))
# random_policy = PolicySquare(env)
nes = NES(env, random_policy.length, max_iter=episodes, population_size=50)
mu = np.zeros(random_policy.length)
sigma = np.ones(random_policy.length) * 100
w, s, r, e = nes.do(mu, sigma, random_policy)

print(w, s)

x = np.arange(episodes)
plt.errorbar(x, r, e, linestyle='-', marker='x', markeredgecolor='red')

plt.show()

run_benchmark(random_policy, w, env)
for i in range(20):
    print(render(random_policy, w, env, step_size=1))





# # Qube
# w = np.array([-31785.660418, -2227.473459, 7.347786, -0.299539, 77.704061,
#      -25.306262, 7.626246, 0.681779, -56.638355, 67.61963,
#      6.314224, 0.93536, 215.328449, 17.814674, -33.518828,
#      -181.456118, -30.605811, -35.840817, -157.349857, 463.711905,
#      259.252819, 11075.521799, -131.53329, 41.189572, 41.635023,
#      -52.245608, 1.559508, -26.364992, -8.095538, -2.098233,
#      -10.220149, -4.249671, 33.044315, 2.144991, -8.673519,
#      7.507008, -8.117554, -17.950644, -19.738507, -2.519171,
#      -2.158746, 1.870617, 19.724263, 125.531829, 2.536407,
#      2768.216542, -4.465866, -157.431832, 3.140001, 22.118608,
#      -3.080627, 13.861909, 26.300828, 6.859524, -29.677852,
#      149.554352, 0.437819, -6.615781, 2.946169, 1.796828,
#      4.736637, -5.141543, 8.657232, 1.534183, 3.924998])

# CartpoleSwingupShort
# w = np.array(
# [14.825309, 51.494195, 21.555105, -2.729548, -5.635839,
#  66.424593, -0.815661, 8.209522, -11.516751, -8.28176,
#  -17.49363, -431.367915, 14.601292, -32.020926, 15.835356,
#  -13.066291, -94.766865, 2.582798, -19.842735, 2.388979,
#  -15.096263, -213.414783, -210.294485, 72.537768, 23.706841,
#  49.604795, 7.46807, 1.646445, 2.099388, 12.016627,
#  -423.458024, 29.570824, -1.570491, 78.579185, -4.474687,
#  -70.808208, -18.595336, 91.755613, -26.340203, -4.936742,
#  9.44686, 7.502085, -0.27206, 3.667115, 29.933978, 0.415163,
#  -3.615755, -24.248179, 0.834203, -0.748019, -10.027531,
#  -4.539246, -2.154302, -3.628191, -0.108677, 2.820117, 0.82556])
