from NES import *
from utilities.Environment import Environment
import matplotlib.pyplot as plt
import Helper

#######################################
# Environment
#######################################

""" define the environment """
gym_env = 'DoublePendulum-v0'

env = Environment(gym_env)

print("================== Start {} ==================".format(gym_env))


""" create NES-algorithm """
episodes = 500
algorithm = NES(env, policy='square', episodes=episodes)

""" train the policy """
policy, sigma, means, stds = algorithm.do()

""" plot learning curve"""
x = np.arange(episodes)
Helper.plot(gym_env, x, means, stds)

""" check the results """
Helper.run_benchmark(policy, env)

""" render one episode"""
Helper.render(policy, env, step_size=1)

""" close environment"""
env.close()
