import numpy as np
import gym
import quanser_robots
import matplotlib.pyplot as plt
from NPG import NPG
from Policies import SoftmaxPolicy, GaussianPolicy
from Features import RbfFeatures
from environment import Environment
from agent import Agent


def run_benchmark(policy, w):
    total_rewards = np.zeros(100)
    print("Starting Benchmark:")
    print("-------------------")
    for i_episode in range(100):
        print("Episode {}:".format(i_episode+1))

        state = env.reset()
        state = state[None, :]
        # state = feature.featurize_state(state)
        for t in range(200):
            # env.render()
            action = policy.get_action(state, w, True)
            state, reward, done, info = env.step(action)
            state = state[None, :]
            # state = feature.featurize_state(state)
            total_rewards[i_episode] += reward
            if done:
                print("Reward reached: ", total_rewards[i_episode])
                print("Episode finished after {} timesteps.".format(t + 1))
                break
    average = np.sum(total_rewards)/100
    print("Average Reward: ", average)
    if average >= 195:
        return True
    else:
        return False



# gym_env = 'CartPole-v0'
# # gym_env = 'CartpoleSwingShort-v0'
# env = Environment(gym_env)
# # feature = RbfFeatures(env)
# feature = None
# policy = SoftmaxPolicy(env)
# # policy = GaussianPolicy(env)
# agent = Agent(env, policy, None, feature=feature)
# agent.train_policy()

env = Environment('CartPole-v0')
# env = gym.make(gym_env)
env.seed(0)
policy = SoftmaxPolicy(env)
# policy = GaussianPolicy()
algorithm = NPG(env, policy, 500)
w, r, t = algorithm.train()
print(w)
plt.plot(np.arange(len(r)), r)
# plt.plot(np.arange(len(t)), t, 'g')
# plt.show()
# # passed = run_benchmark(policy, w)
# # print(passed)
# env.close()
env.close()
