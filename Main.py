import torch as tr
import numpy as np
from NPG import NPG
from NN_GaussianPolicy import Policy
from Environment import Environment
from Agent import Agent

#######################################
# Environment
#######################################


## Softmaxpolicy on cartpole
# np.random.seed(0)
# tr.manual_seed(0)
# gym_env = 'CartPole-v0'
# env = Environment(gym_env)
# policy = Policy(env, hidden_dim=(168, 168), activation=tr.nn.Tanh())
# algorithm = NPG(0.01)
# agent = Agent(env, policy, algorithm)
# agent.train_policy(200, 10)

# Gausspolicy on cartpoleswingup
np.random.seed(0)
tr.manual_seed(0)
gym_env = 'CartpoleSwingShort-v0'
env = Environment(gym_env)
policy = Policy(env, hidden_dim=(150, 150, 50), activation=tr.nn.Tanh())
algorithm = NPG(0.05)
agent = Agent(env, policy, algorithm)
agent.train_policy(2000, 1)
#agent.benchmark_test()


