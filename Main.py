import torch as tr
import numpy as np
from NPG import NPG
from NN_GaussianPolicy import Policy
#from policies.NN_SoftmaxPolicy import Policy
from Environment import Environment
from Agent import Agent
from Baseline import Baseline

#######################################
# Environment
#######################################


## Softmaxpolicy on cartpole
# np.random.seed(0)
# tr.manual_seed(0)
# gym_env = 'CartPole-v0'
# env = Environment(gym_env)
# policy = Policy(env, hidden_dim=(100, 100, 100), activation=tr.nn.Tanh())
# algorithm = NPG(0.01)
# agent = Agent(env, policy, algorithm)
# agent.train_policy(200, 10)

# Gausspolicy on cartpoleswingup
np.random.seed(0)
tr.manual_seed(0)
gym_env = 'CartpoleSwingShort-v0'
env = Environment(gym_env)
policy = Policy(env, hidden_dim=(100, 100))
baseline = Baseline(env, hidden_dim=(150, 150))
algorithm = NPG(0.001)
agent = Agent(env, policy, algorithm, baseline, _gamma=0.99) # , render=True)
agent.train_policy(1500, 10)
agent.benchmark_test()


