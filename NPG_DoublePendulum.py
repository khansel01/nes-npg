import torch as tr
import numpy as np
from Agent import Agent
from NPG import NPG
from models.NN_GaussianPolicy import Policy
from utilities.Environment import Environment
from models.Baseline import Baseline
from utilities.Normalizer import Normalizer

#######################################
# Environment
#######################################

""" set seed """
np.random.seed(0)
tr.manual_seed(0)

""" define the environment """
gym_env = 'DoublePendulum-v0'
print("===================== Start {} =====================".format(gym_env))
env = Environment(gym_env)

""" create policy """
policy = Policy(env, hidden_dim=(8,))

""" create baseline """
baseline = Baseline(env, hidden_dim=(6, 6))

""" create Normalizer to scale the states/observations """
normalizer = Normalizer(env)

""" create NPG-algorithm """
algorithm = NPG(0.005)

""" create agent """
agent = Agent(env, policy, algorithm, baseline, render=True)

""" train the policy """
agent.train_policy(1000, 100, normalizer=normalizer)

print("====================== DO Benchmark ======================")
""" check the results """
#   TODO benchmark has a bug
agent.benchmark_test(episodes=2, render=True)

env.reset()
env.close()