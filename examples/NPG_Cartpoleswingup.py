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
gym_env = 'CartpoleSwingShort-v0'
env = Environment(gym_env, horizon=2000, clip=3)

print("===================== Start {} =====================".format(gym_env))

# """ load pretrained data """
# path = "{}_npg.p".format(gym_env)
# pickle_in = open(path, "rb")
# policy, baseline, normalizer = pickle.load(pickle_in)

""" create policy """
policy = Policy(env, hidden_dim=(10,))

""" create baseline """
baseline = Baseline(env, hidden_dim=(50, 50), epochs=10)

""" create Normalizer to scale the states/observations """
normalizer = Normalizer(env)


""" create NPG-algorithm """
algorithm = NPG(baseline, 0.05, _gamma=0.999, normalizer=normalizer)

""" create agent """
agent = Agent(env, policy, algorithm)

""" train the policy """
agent.train_policy(500, 20)

print("====================== DO Benchmark ======================")
""" check the results """
Helper.run_benchmark(policy, env, normalizer=normalizer)

""" render one episode"""
Helper.render(policy, env, step_size=10)

""" Save trained data """
pickle_out = open("{}_npg.p".format(gym_env), "wb")
pickle.dump((policy, baseline, normalizer), pickle_out)
pickle_out.close()
