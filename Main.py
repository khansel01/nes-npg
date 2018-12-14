import numpy as np
import gym
import quanser_robots
import matplotlib.pyplot as plt
from NPG import NPG
from Policies import SoftmaxPolicy, GaussianPolicy
from Features import RbfFeatures
from Environment import Environment
from Agent import Agent

#######################################
# Environment
#######################################


np.random.seed(1)
gym_env = 'CartPole-v0'
# gym_env = 'CartpoleSwingShort-v0'
env = Environment(gym_env)
# feature = RbfFeatures(env)
feature = None
policy = SoftmaxPolicy(env)
algorithm = NPG(0.001)
# policy = GaussianPolicy(env)
agent = Agent(env, policy, algorithm, feature=feature)
agent.train_policy(200)
agent.benchmark_test()


# env = Environment('CartPole-v0')
# # env = gym.make(gym_env)
# env.seed(0)
# policy = SoftmaxPolicy(env)
# # policy = GaussianPolicy()
# algorithm = NPG(env, policy, 500)
# w, r, t = algorithm.train()
# print(w)
# plt.plot(np.arange(len(r)), r)
# plt.plot(np.arange(len(t)), t, 'g')
# plt.show()
# # passed = run_benchmark(policy, w)
# # print(passed)
# env.close()

