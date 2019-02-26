import numpy as np
import matplotlib.pyplot as plt


def run_benchmark(policy, env, episodes=100, normalizer=None):
    total_rewards = np.zeros(episodes)
    print("Starting Benchmark:")
    print("-------------------")
    for i_episode in range(episodes):
        # print("Episode {}:".format(i_episode + 1))

        env.seed(i_episode)
        state = env.reset()
        state = normalizer.transform(state) \
            if normalizer is not None else state
        done = False
        while not done:
            # env.render()
            action = policy.get_action(state, greedy=True)
            # print(action)
            state, reward, done, info = env.step(action)
            state = normalizer.transform(state) \
                if normalizer is not None else state
            total_rewards[i_episode] += reward

        print(i_episode + 1,
              "Reward reached: ", total_rewards[i_episode])

    average = np.sum(total_rewards) / episodes
    print("-------------------")
    print("Average reward: ", average)
    print("Min reward:", min(total_rewards))
    print("Max reward:", max(total_rewards))
    return average


def render(policy, env, seed=False, step_size=1):
    if seed:
        env.seed(0)

    state = env.reset()
    done = False
    r = 0
    step = 0
    while not done:
        env.render() if step % step_size == 0 else None
        action = policy.get_action(state, greedy=True)
        state, reward, done, info = env.step(action)
        r += reward
        step += 1
    return r

