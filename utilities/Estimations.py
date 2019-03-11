""" Module containing an estimator to estimate the empirical return for
    the baseline updates as well as an estimator to estimate the
    baseline's generalized advantage for the natural policy gradient.
"""

import numpy as np


def estimate_value(trajectories, _gamma):
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
            values[i] = rewards[i] if i == len(values) - 1 \
                else rewards[i] + _gamma * values[i + 1]
        t["values"] = values


def estimate_advantage(trajectories, baseline, _gamma=0.98, _lambda=0.95):
    """This function calculates the advantage function for each state as
    generalized advantage estimator, GAE.

    John Schulman, Philipp Moritz, Sergey Levine, Michael I. Jordan and
    Pieter Abbeel, High-Dimensional Continuous Control Using Generalized
    Advantage Estimation, International Conference on Learning
    Representations, 2016

    :param trajectories: Contains a set of trajectories each being a
        dictionary with information about every transition performed in
        the trajectory simulation
    :type trajectories: list of dictionaries

    :param baseline: The baseline represents an estimator for estimating
        the value function
    :type baseline: baseline

    :param _gamma: Determines the scale of the value function
    :type _gamma: float

    :param _lambda: Controls the bias and variance trade-off
    :type _lambda: float
    """

    for t in trajectories:
        values = baseline.predict(t)
        rewards = t["rewards"]
        advantage = np.zeros_like(rewards)
        delta = np.zeros_like(rewards)
        delta[:-1] = \
            rewards[:-1] - values[:-1] + _gamma * values[1:]
        delta[-1] = rewards[-1] - values[-1]
        for i in range(len(delta) - 1, -1, -1):
            advantage[i] = delta[i] if i == len(delta) - 1 else \
                delta[i] + _gamma * _lambda * advantage[i + 1]
        t["advantages"] = advantage

    # Normalize all the advantages
    advantages = np.concatenate([t["advantages"] for t in trajectories])
    mean = advantages.mean()
    std = advantages.std()
    for t in trajectories:
        t["advantages"] = (t["advantages"] - mean)/(std + 1e-6)
