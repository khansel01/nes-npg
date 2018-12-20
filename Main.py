import numpy as np
import gym
import quanser_robots
import matplotlib.pyplot as plt
from NPG import NPG
from Policies import SoftmaxPolicy, GaussianPolicy
from Features import RbfFeatures, RBFs
from Environment import Environment
from Agent import Agent

#######################################
# Environment
#######################################


np.random.seed(1)
gym_env = 'CartPole-v0'
env = Environment(gym_env)
policy = SoftmaxPolicy(env)
feature = None
# gym_env = 'CartpoleSwingShort-v0'
# env = Environment(gym_env)
# feature = RbfFeatures(env, SoftmaxPolicy(env))
# policy = GaussianPolicy(env)
algorithm = NPG(0.001)
agent = Agent(env, policy, algorithm, feature=feature)
agent.train_policy(200)
# agent.benchmark_test()


