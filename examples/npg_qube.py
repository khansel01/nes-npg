"""Main module for running the algorithms

:Date: 2019-03-11
:Version: 1
:Authors:
    - Janosch Moos
    - Kay Hansel
    - Cedric Derstroff
"""

import torch as tr
import numpy as np
import pickle
import os

from npg import NPG
from agent import Agent
from models.nn_policy import Policy
from utilities.environment import Environment
from models.baseline import Baseline
from utilities.normalizer import Normalizer


def main(load: bool = False, train: bool = False, benchmark: bool = False,
         save: bool = False, render: bool = True):
    """Main function for running this program. At the bottom the
    settings can be chosen.

    All relevant parameters used for NPG and NES can be set in the code
    below as well as environment settings and training, benchmark
    settings

    :param load: Defines whether or not an existing policy and algorithm
     will be loaded or new ones are created
    :type load: bool

    :param train: Defines whether training will be done or not
    :type train: bool

    :param benchmark: Decides whether or not a benchmark evaluation is
        included
    :type benchmark: bool

    :param save: Defines whether or not the policy and algorithm will be
        saved
    :type save: bool

    :param render: Defines whether or not there is a final rendered run
    :type render: bool
    """

    # set seeds for numpy and pytorch
    np.random.seed(0)
    tr.manual_seed(0)

    # define the environment
    gym_env = 'Qube-v0'

    # create environment using Environment wrapper
    env = Environment(gym_env)
    print("{:-^50s}".format(' Start {} '.format(gym_env)))

    if load:
        # load pre trained policy and algorithm from data
        print("{:-^50s}".format(' Load '))
        path = os.getcwd() + "/trained_data/{}_NPG.p".format(env.to_string())

        pickle_in = open(path, "rb")

        policy, algorithm = pickle.load(pickle_in)
    else:
        # create new policy
        print("{:-^50s}".format(' Init '))
        policy = Policy(env, hidden_dim=(6, 6))

        print(policy.get_parameters())
        # create NPG-algorithm, baseline and normalizer
        # NPG needs a baseline, however normalizer can be used at own
        # will
        baseline = Baseline(env, hidden_dim=(6, 6), epochs=10)
        normalizer = Normalizer(env)
        algorithm = NPG(baseline, 0.05, _gamma=0.9999, normalizer=normalizer)

    # create agent for controlling the training and benchmark process
    agent = Agent(env, policy, algorithm)

    if train:
        # train the policy
        print("{:-^50s}".format(' Train '))
        agent.train_policy(episodes=500, n_roll_outs=100, save=save)

    if benchmark:
        # check the results in a benchmark test
        # Unchanged, 100 trials will be run on the environment and
        # plotted for
        # evaluation
        print("{:-^50s}".format(' Benchmark '))
        agent.run_benchmark(episodes=10)

    if render:
        # Runs a single rendered trial for visual performance check
        print("{:-^50s}".format(' Render '))
        agent.run_benchmark(episodes=2, render=True)


if __name__ == '__main__':
    main(load=True, train=False, benchmark=True, save=False, render=True)
