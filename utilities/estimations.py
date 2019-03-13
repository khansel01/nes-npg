""" Module containing an estimator to estimate the empirical return for
the baseline updates as well as an estimator to estimate the
baseline's generalized advantage for the natural policy gradient.
Further it defines the fitness function used in the natural evolution
strategies to evaluate the samples' fitness.

:Date: 2019-03-11
:Version: 1
:Authors:
    - Kay Hansel
    - Cedric Derstroff
    - Janosch Moos
"""

import numpy as np


def estimate_value(trajectories, _gamma: float):
    """ Calculates the state-value function based on empirical
    monte-carlo returns

    :param trajectories: Contains a set of trajectories each being a
        dictionary with information about every transition performed in
        the trajectory simulation
    :type trajectories: list of dictionaries

    :param _gamma: Represents the discount factor
    :type _gamma: float
    """

    for t in trajectories:
        rewards = t["rewards"]
        values = np.zeros_like(rewards)
        for i in range(len(values) - 1, -1, -1):
            if i == len(values) - 1:
                values[i] = rewards[i]
            else:
                values[i] = rewards[i] + _gamma * values[i + 1]
        t["values"] = values


def estimate_advantage(trajectories, baseline, _gamma: float = 0.98,
                       _lambda: float = 0.95):
    """This function calculates the advantage function for each state as
    generalized advantage estimator, GAE.

    John Schulman, Philipp Moritz, Sergey Levine, Michael I. Jordan and
    Pieter Abbeel, High-Dimensional Continuous Control Using Generalized
    Advantage Estimation, International Conference on Learning
    Representations, 2016

    :param trajectories: Contains a set of trajectories each being a
        dictionary with information about every transition performed in
        the trajectory simulation
    :type trajectories: list of dict

    :param baseline: The baseline represents an estimator for estimating
        the value function
    :type baseline: Baseline

    :param _gamma: Determines the scale of the value function
    :type _gamma: float

    :param _lambda: Controls the bias and variance trade-off
    :type _lambda: float
    """

    for t in trajectories:
        values = baseline.predict(t).reshape(-1, 1)
        rewards = t["rewards"].reshape(-1, 1)
        advantage = np.zeros_like(rewards)
        delta = np.zeros_like(rewards)
        delta[:-1] = rewards[:-1] - values[:-1] + _gamma * values[1:]
        delta[-1] = rewards[-1] - values[-1]

        for i in range(len(delta) - 1, -1, -1):
            if i == len(delta) - 1:
                advantage[i] = delta[i]
            else:
                advantage[i] = delta[i] + _gamma * _lambda * advantage[i + 1]
        t["advantages"] = advantage

    # Normalize all the advantages
    advantages = np.concatenate([t["advantages"] for t in trajectories])
    mean = advantages.mean()
    std = advantages.std()
    for t in trajectories:
        t["advantages"] = (t["advantages"] - mean)/(std + 1e-6)


def estimate_fitness(policy, env, w, n_roll_outs: int = 1):
    """Evaluates the fitness of each sample in a set of samples.
    This is done by running a simulation on the environment using
    the same seed for all trials.

    :param policy: The policy
    :type policy: Policy

    :param env: Contains the gym environment the simulations are
        performed on
    :type env: Environment

    :param w: The weights for the policy
    :type w: ndarray of floats

    :param n_roll_outs: Number of roll outs per policy
    :type n_roll_outs: int

    :returns the fitness of the policy, i.e. the total reward of the
        episode on the given environment
    :rtype: float
    """

    samples = np.size(w, 0)
    f = np.zeros(samples)
    steps = np.zeros(samples)

    # define seed to be the same for each sample
    # numpy is used to get deterministic outcomes during testing
    # seed = env.get_seed()
    seed = np.random.randint(0, 90000)

    for s in range(samples):
        # set sample as policy parameters
        policy.set_parameters(w[s])

        env.seed(seed)

        trajectories: dict = env.roll_out(policy, n_roll_outs=n_roll_outs,
                                          greedy=True)

        t_steps = []
        t_reward = []
        for t in trajectories:
            t_steps.append(t["time_steps"])
            t_reward.append(t["total_reward"])

        steps[s] = np.sum(t_steps) / n_roll_outs
        f[s] = np.sum(t_reward) / n_roll_outs

    return f, steps
