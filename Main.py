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

## Gausspolicy on cartpoleswingup
np.random.seed(0)
tr.manual_seed(0)
gym_env = 'CartpoleSwingShort-v0'
env = Environment(gym_env)
policy = Policy(env, hidden_dim=(32, 32))
baseline = Baseline(env, hidden_dim=(128, 128))
algorithm = NPG(0.01)
agent = Agent(env, policy, algorithm, baseline, _gamma=0.99)
agent.train_policy(1000, 10)

agent.benchmark_test(episodes=2, render=True)


