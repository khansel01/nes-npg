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


""" Gausspolicy on cartpoleswingup """
print("================== Start Cartpoleswingup ==================")

""" set seed """
np.random.seed(0)
tr.manual_seed(0)

""" define the environment """
# gym_env = 'CartpoleSwingShort-v0'
gym_env = 'Pendulum-v0'
env = Environment(gym_env)

""" create policy """
policy = Policy(env, hidden_dim=(32, 32), log_std=-1)

""" create baseline """
baseline = Baseline(env, hidden_dim=(32, 32))

""" create Normalizer to scale the states/observations """
normalizer = Normalizer(env)

""" create NPG-algorithm """
algorithm = NPG(0.0005)

""" create agent """
agent = Agent(env, policy, algorithm, baseline)

""" train the policy """
agent.train_policy(100, 20, normalizer=normalizer)

print("====================== DO Benchmark ======================")
""" check the results """
#   TODO benchmark has a bug
agent.benchmark_test(episodes=2, render=True)


