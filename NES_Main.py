from NES import *
from utilities.Environment import Environment
import Helper

#######################################
# Environment
#######################################

""" define the environment """
gym_env = 'Pendulum-v0'
# gym_env = 'DoublePendulum-v0'
# gym_env = 'Cartpole-v0'
# gym_env = 'CartpoleSwingShort-v0'
# gym_env = 'CartpoleSwingLong-v0'

env = Environment(gym_env)

print("================== Start {} ==================".format(gym_env))


""" create NES-algorithm """
episodes = 100
algorithm = NES(env, policy='square', episodes=episodes, population_size=15)

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
