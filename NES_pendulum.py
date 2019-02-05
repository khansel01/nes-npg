from NES import *
from utilities.Environment import Environment
import matplotlib.pyplot as plt
import Helper

#######################################
# Environment
#######################################

""" define the environment """
gym_env = 'Pendulum-v0'
env = Environment(gym_env)

print("================== Start {} ==================".format(gym_env))


""" create NES-algorithm """
episodes = 500
algorithm = NES(env, policy='square', episodes=episodes, population_size=15)

""" train the policy """
policy, sigma, means, stds = algorithm.do()

""" plot learning curve"""
x = np.arange(episodes)
plt.errorbar(x, means, stds, linestyle='-', marker='x', markeredgecolor='red')

plt.show()

""" check the results """
Helper.run_benchmark(policy, env)

""" render one episode"""
Helper.render(policy, env, step_size=1)

""" close environment"""
env.close()
