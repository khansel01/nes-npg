from NES import *
from utilities.Environment import Environment
from utilities import Helper
from models.NN_GaussianPolicy import Policy
from Agent import Agent

#######################################
# Environment
#######################################

""" define the environment """
gym_env = 'Pendulum-v0'
env = Environment(gym_env)

print("================== Start {} ==================".format(gym_env))


""" create policy """
policy = Policy(env, hidden_dim=(8,))

""" create NES-algorithm """
algorithm = NES(policy.length)

""" create agent """
agent = Agent(env, policy, algorithm)

""" train the policy """
agent.train_policy(episodes=250, n_roll_outs=1)

""" check the results """
Helper.run_benchmark(policy, env)

""" render one episode"""
Helper.render(policy, env, step_size=1)
