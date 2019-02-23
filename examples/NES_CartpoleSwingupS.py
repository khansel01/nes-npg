from NES import *
from utilities.Environment import Environment
from utilities import Helper
from models.NN_GaussianPolicy import Policy
from Agent import Agent
import pickle

#######################################
# Environment
#######################################

""" define the environment """
gym_env = 'CartpoleSwingShort-v0'
env = Environment(gym_env, clip=3)

print("================== Start {} ==================".format(gym_env))

""" load pretrained data """

path = "{}_3clipped100_nes.p".format(gym_env)
pickle_in = open(path, "rb")
policy = pickle.load(pickle_in)

""" create policy """
# policy = Policy(env, hidden_dim=(10,))
#
# """ create NES-algorithm """
algorithm = NES(policy.length, sigma_init=1.0)

""" create agent """
agent = Agent(env, policy, algorithm)

""" train the policy """
agent.train_policy(episodes=10, n_roll_outs=5)

""" check the results """
Helper.run_benchmark(policy, env)

""" render one episode"""
Helper.render(policy, env, step_size=5)

""" Save trained data """
path = "{}_3clipped.2_nes.p".format(gym_env)
pickle_out = open(path, "wb")
pickle.dump(policy, pickle_out)
pickle_out.close()