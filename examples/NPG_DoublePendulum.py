import torch as tr
import numpy as np
from agent import Agent
from npg import NPG
from models.nn_policy import Policy
from utilities.environment import Environment
from models.baseline import Baseline
from utilities.normalizer import Normalizer
import pickle

"""Module containing an example set up using NPG on the Double Pendulum
Platform
"""


def main(load: bool = False, train: bool = False, benchmark: bool = False,
         save: bool = False, render: bool = True):
    """Main function for running this program. At the bottom the settings can
    be chosen.
     -  "load" defines whether an existing policy and algorithm will be loaded
        or new ones are created
     -  "train" defines whether training will be done
     -  "benchmark" decides whether a benchmark evaluation is included
     -  "save" defines whether the policy and algorithm should be saved
     -  "render" defines if there is a final rendered run
     All relevant parameters used for NPG and NES can be set in the code below
     as well as environment settings and training, benchmark settings
     """

    # set seeds for numpy and pytorch
    np.random.seed(0)
    tr.manual_seed(0)

    # define the environment
    gym_env = 'DoublePendulum-v0'
    env = Environment(gym_env)
    print("{:=^50s}".format(' Start {} '.format(gym_env)))

    if load:
        # load pre trained policy and algorithm from data
        print("{:=^50s}".format(' Load '))
        path = "trained_data/{}_300_5.0_NPG.p".format(gym_env)

        pickle_in = open(path, "rb")

        policy, algorithm = pickle.load(pickle_in)
    else:
        # create new policy, baseline, Normalizer as necessary
        print("{:=^50s}".format(' Init '))
        policy = Policy(env, hidden_dim=(10, 10))

        baseline = Baseline(env, hidden_dim=(10, 10), epochs=20)

        normalizer = Normalizer(env)

        # create NPG-algorithm, baseline and normalizer
        # NPG needs a baseline, however normalizer can be used at own will
        algorithm = NPG(baseline, 0.005, _gamma=0.9999, _lambda=0.2, normalizer=normalizer)

    # create agent for controlling the training and benchmark process
    agent = Agent(env, policy, algorithm)

    if train:
        # train the policy
        print("{:=^50s}".format(' Train '))
        agent.train_policy(episodes=1000, n_roll_outs=10, save=save)

    if benchmark:
        # check the results in a benchmark test
        # Unchanged, 100 trials will be run on the environment and plotted for
        # evaluation
        print("{:=^50s}".format(' Benchmark '))
        agent.run_benchmark()

    if render:
        # Runs a single rendered trial for visual performance check
        print("{:=^50s}".format(' Render '))
        agent.run_benchmark(episodes=1, render=True)

    return


if __name__ == '__main__':
    main(load=False, train=True, benchmark=True, save=False, render=True)

