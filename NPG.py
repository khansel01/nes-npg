import numpy as np
import gym
import matplotlib.pyplot as plt
from Softmax_policy import SoftmaxPolicy

#######################################
# NPG using Softmax Policy
#######################################


class NPG:
    # Define training setup
    # --------------------------------
    # gamma is the discount factor.
    # lambda is the bias-variance tradeoff for the advantage function.
    # T is the max number of steps in a single run of the simulation.
    # K is the number of episodes for training the algorithm.
    # delta is the normalized step size of the parameter update.

    def __init__(self, env, policy, episodes):
        self.env = env
        self.policy = policy
        self.__n_Actions = env.action_space.n
        self.__K = episodes
        self.__lambda = 0.95
        self.__gamma = 0.98
        self.__delta = 0.000025
        self.__eps = np.finfo(np.float32).eps.item()
        self.W = np.random.sample((4, 2))
        # self.W = np.ones((4, 2)) * 1.0
        # self.W[:, 0] *= -1

    def train(self):
        rewards_per_episode = []
        for i_episode in range(self.__K):
            print("Episode ", i_episode, ":")
            log_gradients = [[], []]
            rewards = []

            state = self.env.reset()[None, :]
            self.env.seed(1)
            while(True):
                self.env.render()

                old_state = state
                prob = self.policy.get_action_prob(state, self.W)
                action = np.random.choice(self.__n_Actions, p=prob[0])
                state, reward, done, _ = self.env.step(action)
                state = state[None, :]

                p_grad = self.policy.get_p_grad(old_state, self.W)[action, :]
                log_grad = p_grad / prob[0, action]
                log_grad = np.dot(old_state.T, log_grad[None, :])

                for i in range(self.__n_Actions):
                    log_gradients[i].append(log_grad[:, i])
                rewards.append(reward)

                if done:
                    print("Trial finished after {} timesteps."
                          .format(np.sum(rewards)))
                    break
            self.__update_parameters(log_gradients, rewards)
            print(self.W)
            rewards_per_episode.append(np.sum(rewards))
        return self.W, rewards_per_episode

    def __update_parameters(self, log_gradients, rewards):
        for n in range(self.__n_Actions):
            g = self.__compute_gradient(log_gradients[n], rewards)
            fisher = self.__compute_fisher(log_gradients[n])
            try:
                inv_fisher = np.linalg.inv(fisher)
                nominator = (g.T @ inv_fisher) @ g
                if nominator <= 0:
                    print("Nominator <= 0: ", nominator)
                else:
                    learning_rate = np.sqrt(self.__delta/nominator)
                    step = np.multiply(learning_rate, (inv_fisher @ g))

                    c = np.dot(step.T, fisher)
                    c = np.dot(c, step)
                    if c > self.__delta:
                        print("condition: ", c, " > ", self.__delta)


                    self.W[:, n] += step
            except np.linalg.LinAlgError:
                print("Skipping parameter update due to singular matrix.")
                pass
        return

    def __compute_gradient(self, log_g, rewards):
        g = 0
        advantage = np.zeros(len(log_g))
        for i in range(len(log_g)):
            advantage[i] = sum([r * (self.__gamma ** t) for t, r in
                                enumerate(rewards[i:])])
        advantage = (advantage - np.mean(advantage)) / (np.std(advantage) +
                                                        self.__eps)
        for i in range(len(log_g)):
            g += log_g[i] * advantage[i]

        return g / len(log_g)

    def __compute_fisher(self, log_g):
        f = sum([(lg.reshape(-1, 1) @ lg.reshape(-1, 1).T) for lg in log_g])
        return f / len(log_g)


class SoftmaxPolicy:

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
