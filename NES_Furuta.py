from NES import *
from utilities.Environment import Environment
import Helper

#######################################
# Environment
#######################################

""" define the environment """
gym_env = 'Qube-v0'

env = Environment(gym_env)

print("================== Start {} ==================".format(gym_env))


""" create NES-algorithm """
episodes = 1000
hidden_dim = (6, 6)
print(hidden_dim)
algorithm = NES(env, policy='nn', hidden_dim=hidden_dim,
                episodes=episodes)

""" train the policy """
policy, sigma, means, stds = algorithm.do(sigma_init=1)

""" plot learning curve"""
x = np.arange(episodes)
Helper.plot(gym_env, x, means, stds)

""" check the results """
Helper.run_benchmark(policy, env)

""" render one episode"""
# Helper.render(policy, env, step_size=1)

""" close environment"""
env.close()

print(policy.get_parameters())
