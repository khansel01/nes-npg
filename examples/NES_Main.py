from NES import *
from utilities.Environment import Environment
from utilities import Helper
from models.NN_GaussianPolicy import Policy
from Agent import Agent

#######################################
# Environment
#######################################

""" define the environment """
# gym_env = 'Pendulum-v0'
gym_env = 'Qube-v0'
# gym_env = 'Levitation-v0'
# gym_env = 'Walker2d-v2'
# gym_env = 'DoublePendulum-v0'
# gym_env = 'Cartpole-v0'
# gym_env = 'CartpoleSwingShort-v0'
# gym_env = 'CartpoleSwingLong-v0'

env = Environment(gym_env)

print("================== Start {} ==================".format(gym_env))


""" create policy """
policy = Policy(env, hidden_dim=(8,))

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
