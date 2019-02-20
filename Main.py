import torch as tr
import numpy as np
import gym
import quanser_robots
import matplotlib.pyplot as plt
from NES import *
from utilities.Environment import Environment
from Agent import Agent

#######################################
# Environment
#######################################

""" set seed """
np.random.seed(0)
tr.manual_seed(0)

""" define the environment """
gym_env = 'CartpoleSwingShort-v0'
print("===================== Start {} =====================".format(gym_env))
env = Environment(gym_env)

""" create policy """
policy = Policy(env, hidden_dim=(5, 5), log_std=0)

""" create baseline """
baseline = Baseline(env, hidden_dim=(5, 5), epochs=10)

    state = env.reset()
    done = False
    r = 0
    policy.set_parameters(w)
    step = 0
    while not done:
        env.render() if step % step_size == 0 else None
        action = policy.get_action(state)
        state, reward, done, info = env.step(np.asarray([action]))
        r += reward
        # print(action, state[4])
        step += 1
    return r

""" create NPG-algorithm """
algorithm = NPG(0.005)

""" create agent """
agent = Agent(env, policy, algorithm, baseline, _gamma=0.99)

""" train the policy """
agent.train_policy(200, 20, normalizer=normalizer)

x = np.arange(episodes)
plt.errorbar(x, r, e, linestyle='-', marker='x', markeredgecolor='red')

env.reset()
env.close()