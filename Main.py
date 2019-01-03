import numpy as np
import gym
import quanser_robots.qube
import matplotlib.pyplot as plt
from NPG import NPG
from Policies import SoftmaxPolicy
import Features


def run_benchmark(policy, w):
    total_rewards = np.zeros(1000)
    print("Starting Benchmark:")
    print("-------------------")
    for i_episode in range(1000):
        print("Episode {}:".format(i_episode+1))

        state = env.reset()
        state = np.asarray(state)
        state = state[None, :]
        # state = feature.featurize_state(state)
        t = 0
        while(True):
            env.render()
            probs = policy.get_action_prob(state, w)
            action = np.argmax(probs)
            state, reward, done, info = env.step(np.asarray(action))
            state = np.asarray(state)
            state = state[None, :]
            # state = feature.featurize_state(state)
            total_rewards[i_episode] += reward
            t += 1
            if done:
                print("Reward reached: ", total_rewards[i_episode])
                print("Episode finished after {} timesteps.".format(t))
                break
    average = np.sum(total_rewards)/1000
    print("Average Reward: ", average)
    if average >= 195:
        return True
    else:
        return False


env = gym.make('CartpoleSwingShort-v0')
env.seed(0)
policy = SoftmaxPolicy()
algorithm = NPG(env, policy, 2000)
w, r = algorithm.train()
print(w)
plt.plot(np.arange(len(r)), r)
plt.show()
passed = run_benchmark(policy, w)
print(passed)
env.close()