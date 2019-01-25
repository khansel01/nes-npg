import torch as tr
import numpy as np
from NPG import NPG
from NN_GaussianPolicy import Policy
#from policies.NN_SoftmaxPolicy import Policy
from Environment import Environment
from Agent import Agent
from Baseline import Baseline
from Normalizer import Normalizer

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


""" Gausspolicy on cartpoleswingup """
print("================== Start Cartpoleswingup ==================")

""" set seed """
np.random.seed(0)
tr.manual_seed(0)

""" define the environment """
gym_env = 'DoublePendulum-v0'  # 'Pendulum-v0'  # 'CartpoleSwingShort-v0'
env = Environment(gym_env)

""" create policy """
policy = Policy(env, hidden_dim=(32, 32))

""" create baseline """
baseline = Baseline(env, hidden_dim=(32, 32))

""" create Normalizer to scale the states/observations """
normalizer = Normalizer(env)

""" create NPG-algorithm """
algorithm = NPG(0.05)

""" create agent """
agent = Agent(env, policy, algorithm, baseline, _lambda=0.96, _gamma=0.98, render=True)

""" train the policy """
agent.train_policy(100, 20, normalizer=normalizer)

print("====================== DO Benchmark ======================")
""" check the results """
#   TODO benchmark has a bug
agent.benchmark_test(episodes=2, render=True)

