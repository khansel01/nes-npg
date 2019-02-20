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
hidden_dim = (8, 6)
print(hidden_dim)
algorithm = NES(env, policy='gaussian', hidden_dim=hidden_dim,
                episodes=episodes)

""" train the policy """
policy, sigma, means, stds = algorithm.do(sigma_init=.2, seed=987654321)

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


# w = [2.693147, -0.335921, 1.956266, -5.812498, 1.388454, -0.545044, 0.11036,
#      -0.519145, 3.659652, 3.437301, -0.146759, -2.039053, -1.009923, 2.475141,
#      1.058118, -4.392096, -1.360432, 1.596232, -0.321535, -0.95179, 1.14977,
#      -0.416808, -0.576681, 0.216488, 0.519639, -2.494178, 0.930256, -1.602918,
#      -0.087524, -0.041899, -0.192339, 0.509974, -1.280399, 3.322551, 0.544384,
#      0.073667, -3.012741, -0.06783, 0.369093, -0.246433, -1.616315, 0.537478,
#      0.726007, -0.634872, 0.953565, 1.233518, -2.130079, -0.264191, -0.429705,
#      -0.76407, -1.136345, 0.754616, 0.18606, -0.138678, 0.746534, -0.087938,
#      1.773041, 0.695874, -0.868047, 2.747215, -0.517416, 2.357787, 4.744955,
#      -1.147905, -1.326092, 0.559475, -0.187068, 3.236455, 0.180119, 0.765992,
#      2.362155, 0.414358, 0.430997, -1.238772, 1.333316, -0.060441, 2.044225,
#      -2.742436, -0.437364, -0.977538, -1.289457, -0.168034, -0.920484,
#      -1.782632, -2.083966, 0.789738, -2.749243, 0.685473, 0.494868, 2.786625,
#      2.322999]