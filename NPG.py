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
        self.__delta = 0.001
        self.__eps = np.finfo(np.float32).eps.item()
        self.__values = []
        self.W = np.random.sample((4, 2))

    def get_param_values(self):
        return self.__lambda, self.__gamma, self.__delta

    def set_param_values(self, parameter):
        self.__lambda = parameter[0]
        self.__gamma = parameter[1]
        self.__delta = parameter[2]
        return

    def train(self):
        rewards_per_episode = []
        for i_episode in range(self.__K):
            print("Episode ", i_episode, ":")
            log_gradients = []
            rewards = []

            state = self.env.reset()[None, :]
            self.env.seed(0)
            while(True):
                # self.env.render()

                old_state = state
                prob = self.policy.get_action_prob(state, self.W)
                action = np.random.choice(self.__n_Actions, p=prob[0])
                state, reward, done, _ = self.env.step(action)
                state = state[None, :]

                p_grad = self.policy.get_p_grad(old_state, self.W)[action, :]
                log_grad = p_grad / prob[0, action]
                log_grad = np.dot(old_state.T, log_grad[None, :])

                log_gradients.append(log_grad.reshape((-1, 1), order='F'))
                rewards.append(reward)

                if done:
                    print("Trial finished after {} timesteps."
                          .format(np.sum(rewards)))
                    break

            if self.__values==[]:
                self.__values = np.zeros(len(rewards))
            self.__update_parameters(log_gradients, rewards)
            print(self.W)
            self.__values = self.__estimate_value(rewards)
            rewards_per_episode.append(np.sum(rewards))
        return self.W, rewards_per_episode

    def __update_parameters(self, log_gradients, rewards):
        g = self.__compute_gradient(log_gradients, rewards)
        fisher = self.__compute_fisher(log_gradients)
        try:
            inv_fisher = self.__compute_inverse(fisher)
            nominator = (g.T @ inv_fisher) @ g
            if nominator <= 0:
                print("Nominator <= 0: ", nominator)
            else:
                learning_rate = np.sqrt(self.__delta/nominator)
                step = np.multiply(learning_rate, (inv_fisher @ g))

                c = step.T@fisher
                c = c@step
                if c > self.__delta:
                    print("condition: ", c, " > ", self.__delta)

                self.W += step.reshape((4, 2), order='F')
        except np.linalg.LinAlgError:
            print("Skipping parameter update due to singular matrix.")
            pass
        return

    def __compute_gradient(self, log_g, rewards):
        g = 0
        advantage = self.__estimate_advantage(rewards)
        for i in range(len(log_g)):
            g += log_g[i] * advantage[i]
        return g/len(log_g)

    def __compute_fisher(self, log_g):
        f = sum([(lg.reshape(-1, 1) @ lg.reshape(-1, 1).T) for lg in log_g])
        f = np.diagonal(f)[None, :]
        f = np.diagflat(f)
        return f/len(log_g)

    def __compute_inverse(self, matrix):
        u, s, v = np.linalg.svd(matrix)
        s = np.diag(s**-1)
        return v.T@(s@u.T)

    def __estimate_value(self, rewards):
        value = np.zeros(len(rewards))
        for i in range(len(rewards)):
            value[i] = sum([r * (self.__gamma ** t) for t, r in
                                enumerate(rewards[i:])])
        # value = (value - np.mean(value)) / (np.std(value) + self.__eps)
        return value

    def __estimate_advantage(self, rewards):
        index = len(rewards) if len(rewards) >= len(self.__values) else len(self.__values)
        values = self.__values@np.eye(len(self.__values), index)
        rewards = rewards@np.eye(len(rewards), index)
        advantage = np.zeros(index)
        for i in range(index):
            for remainingsteps in range(index - i - 1):
                delta_func = rewards[i+remainingsteps] - values[i+remainingsteps] + self.__gamma*values[i+remainingsteps]
                advantage[i] += ((self.__gamma * self.__lambda) ** remainingsteps) * delta_func
        # advantage = (advantage - np.mean(advantage)) / (np.std(advantage) + self.__eps)
        return advantage


def run_benchmark(policy, w):
    total_rewards = np.zeros(100)
    print("Starting Benchmark:")
    print("-------------------")
    for i_episode in range(100):
        print("Episode {}:".format(i_episode+1))

        observation = env.reset()
        for t in range(200):
            # env.render()
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
env.seed(0)
np.random.seed(1)
policy = SoftmaxPolicy()
algorithm = NPG(env, policy, 200)
w, r = algorithm.train()
_lambda, _gamma, _ = algorithm.get_param_values()
# algorithm = NPG(env, policy, 1000)
# algorithm.set_param_values([_lambda, _gamma, 0.004])
# _, r1 = algorithm.train()
# algorithm = NPG(env, policy, 1000)
# algorithm.set_param_values([_lambda, _gamma, 0.003])
# _, r2 = algorithm.train()
# algorithm = NPG(env, policy, 1000)
# algorithm.set_param_values([_lambda, _gamma, 0.002])
# _, r3 = algorithm.train()
# algorithm = NPG(env, policy, 1000)
# algorithm.set_param_values([_lambda, _gamma, 0.001])
# _, r4 = algorithm.train()

plt.plot(np.arange(len(r)), r, 'g')
# plt.plot(np.arange(len(r1)), r1, 'r')
# plt.plot(np.arange(len(r2)), r2, 'b')
# plt.plot(np.arange(len(r3)), r3, 'y')
# plt.plot(np.arange(len(r4)), r4)
plt.title(["lambda: ", _lambda, "gamma: ", _gamma, "delta: ", 0.005, "to", 0.001])
plt.show()
#passed = run_benchmark(policy, w)
#print(passed)
env.close()
