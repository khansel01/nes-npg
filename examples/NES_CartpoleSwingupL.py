from NES import *
from utilities.Environment import Environment
from utilities import Helper
from models.SquaredFeaturePolicy import PolicySquare
from Agent import Agent

#######################################
# Environment
#######################################

""" define the environment """
gym_env = 'CartpoleSwingLong-v0'
env = Environment(gym_env)

print("================== Start {} ==================".format(gym_env))

""" create policy """
policy = PolicySquare(env)

""" create NES-algorithm """
algorithm = NES(policy.length)

""" create agent """
agent = Agent(env, policy, algorithm)

""" train the policy """
agent.train_policy(episodes=500, n_roll_outs=1)

""" check the results """
Helper.run_benchmark(policy, env)

""" render one episode"""
Helper.render(policy, env, step_size=1)
