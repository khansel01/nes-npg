import torch as tr
import numpy as np
from Agent import Agent
from NPG import NPG
from NES import NES
from models.NN_GaussianPolicy import Policy
from utilities.Environment import Environment
from utilities import Helper
from models.Baseline import Baseline
from utilities.Normalizer import Normalizer

#######################################
# Environment
#######################################

""" set seed """
np.random.seed(0)
tr.manual_seed(0)

""" define the environment """
gym_env = 'Pendulum-v0'
# gym_env = 'Qube-v0'
# gym_env = 'Levitation-v0'
# gym_env = 'Walker2d-v2'
# gym_env = 'DoublePendulum-v0'
# gym_env = 'Cartpole-v0'
# gym_env = 'CartpoleSwingShort-v0'
# gym_env = 'CartpoleSwingLong-v0'

env = Environment(gym_env)

print("===================== Start {} =====================".format(gym_env))


""" create policy """
policy = Policy(env, hidden_dim=(4, 4), log_std=0)

""" create baseline """
baseline = Baseline(env, hidden_dim=(4, 4))

""" create Normalizer to scale the states/observations """
normalizer = Normalizer(env)

""" create NPG-algorithm """
# algorithm = NES(policy.length)
algorithm = NPG(baseline, 0.005, _gamma=0.99, normalizer=normalizer)

""" create agent """
agent = Agent(env, policy, algorithm)

""" train the policy """
agent.train_policy(episodes=5, n_roll_outs=1)

print("====================== DO Benchmark ======================")
""" check the results """
Helper.run_benchmark(policy, env)

""" render one episode"""
Helper.render(policy, env, step_size=1)
