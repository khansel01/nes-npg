import numpy as np
import gym
import quanser_robots
import matplotlib.pyplot as plt
from NPG import NPG
from Policies import SoftmaxPolicy
import Features


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
            probs = policy.get_action_prob(state, w)
            action = np.argmax(probs)
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


# env = gym.make('CartPole-v0')
env = gym.make('CartpoleSwingLong-v0')
env.seed(0)
policy = SoftmaxPolicy()
algorithm = NPG(env, policy, 200)
w, r = algorithm.train()
print(w)
plt.plot(np.arange(len(r)), r)
plt.show()
passed = run_benchmark(policy, w)
print(passed)
env.close()