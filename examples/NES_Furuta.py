from NES import *
from utilities.Environment import Environment
from utilities import Helper
from models.NN_GaussianPolicy import Policy
from Agent import Agent

#######################################
# Environment
#######################################

""" define the environment """
gym_env = 'Qube-v0'
env = Environment(gym_env)

print("================== Start {} ==================".format(gym_env))


""" create policy """
policy = Policy(env, hidden_dim=(6, 6))

""" create NES-algorithm """
algorithm = NES(policy.length, population_size=50, sigma_init=2.0)

""" create agent """
agent = Agent(env, policy, algorithm)

""" train the policy """
agent.train_policy(episodes=1000, n_roll_outs=10)

""" check the results """
Helper.run_benchmark(policy, env)

""" render one episode"""
Helper.render(policy, env, step_size=1)
