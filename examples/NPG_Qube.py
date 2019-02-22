import torch as tr
import numpy as np
from Agent import Agent
from NPG import NPG
from models.NN_GaussianPolicy import Policy
from utilities.Environment import Environment
from utilities import Helper
from models.Baseline import Baseline
from utilities.Normalizer import Normalizer
import pickle

#######################################
# Environment
#######################################

""" set seed """
np.random.seed(0)
tr.manual_seed(0)

""" define the environment """
gym_env = 'Qube-v0'
env = Environment(gym_env)

print("===================== Start {} =====================".format(gym_env))

# """ load pretrained data """
# path = "{}_npg.p".format(gym_env)
# pickle_in = open(path, "rb")
# policy, baseline, normalizer = pickle.load(pickle_in)

""" create policy """
policy = Policy(env, hidden_dim=(6, 6))

""" create baseline """
baseline = Baseline(env, hidden_dim=(4, 4), epochs=10)

""" create Normalizer to scale the states/observations """
normalizer = Normalizer(env)

""" create NPG-algorithm """
algorithm = NPG(baseline, 0.05, _gamma=0.999, normalizer=normalizer)

""" create agent """
agent = Agent(env, policy, algorithm)

""" train the policy """
agent.train_policy(600, 100)

print("====================== DO Benchmark ======================")
""" check the results """
Helper.run_benchmark(policy, env, normalizer=normalizer)

""" render one episode"""
Helper.render(policy, env, step_size=1)

""" Save trained data """
pickle_out = open("{}_npg.p".format(gym_env), "wb")
pickle.dump((policy, baseline, normalizer), pickle_out)
pickle_out.close()
