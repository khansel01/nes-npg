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
gym_env = 'CartpoleSwingShort-v0'
print("===================== Start {} =====================".format(gym_env))
env = Environment(gym_env, horizon=2000, clip=10)

""" create policy """
policy = Policy(env, hidden_dim=(8,), log_std=np.log(3.5))

""" create baseline """
baseline = Baseline(env, hidden_dim=(8, 8), epochs=50, lr=1e-3)

""" create Normalizer to scale the states/observations """
normalizer = Normalizer(env)

""" create NPG-algorithm """
algorithm = NPG(0.0001)

""" create agent """
agent = Agent(env, policy, algorithm, baseline, _gamma=0.997, _lambda=0.945)

""" train the policy """
agent.train_policy(500, 20, normalizer=normalizer)

print("====================== DO Benchmark ======================")
""" check the results """
#   TODO benchmark has a bug
agent.benchmark_test(episodes=5, render=True)

env.reset()
env.close()


