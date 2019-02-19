from NES import *
from utilities.Environment import Environment
import Helper

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


""" create NES-algorithm """
episodes = 10000
algorithm = NES(env, policy='square', episodes=episodes)

""" train the policy """
policy, sigma, means, stds = algorithm.do(sigma_init=25)

""" plot learning curve"""
x = np.arange(episodes)
Helper.plot(gym_env, x, means, stds)

""" check the results """
Helper.run_benchmark(policy, env)

""" render one episode"""
Helper.render(policy, env, step_size=1)

""" close environment"""
env.close()
