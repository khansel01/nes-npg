import numpy as np
import gym
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

        observation = env.reset()
        for t in range(200):
            env.render()
            probs = policy.get_action_prob(observation[None, :], w)
            action = np.argmax(probs)
            observation, reward, done, info = env.step(action)
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


env = gym.make('CartPole-v0')
env.seed(1)
np.random.seed(1)
policy = SoftmaxPolicy()
algorithm = NPG(env, policy, 1000)
w, r = algorithm.train()
plt.plot(np.arange(len(r)), r)
plt.show()
passed = run_benchmark(policy, w)
print(passed)
env.close()