import torch as tr
import numpy as np
from NPG import NPG
from Agent import Agent
from models.NN_GaussianPolicy import Policy
from utilities.environment import Environment
from models.Baseline import Baseline
from utilities.normalizer import Normalizer
import pickle
import os

#######################################
# Environment
#######################################


def main(load: bool = False, train: bool = False, benchmark: bool = False,
         save: bool = False, render: bool = True):
    """ set seed """
    np.random.seed(0)
    tr.manual_seed(0)

    """ define the environment """
    gym_env = 'BallBalancerSim-v0'
    env = Environment(gym_env)
    print("{:-^50s}".format(' Start {} '.format(gym_env)))

    if load:
        """ load pretrained policy, algorithm from data """
        print("{:-^50s}".format(' Load '))
        path = os.getcwd() + "/trained_data/{}_NPG.p".format(env.to_string())

        pickle_in = open(path, "rb")

        policy, algorithm = pickle.load(pickle_in)
    else:
        """ create policy, algorithm """
        print("{:-^50s}".format(' Init '))
        policy = Policy(env, hidden_dim=(8, 8), log_std=0)

        """ create baseline """
        baseline = Baseline(env, hidden_dim=(8, 8))

        """ create Normalizer to scale the states/observations """
        normalizer = Normalizer(env)

        algorithm = NPG(baseline, 0.005, _gamma=0.99999, normalizer=normalizer)

    """ create agent """
    agent = Agent(env, policy, algorithm)

    if train:
        """ train the policy """
        print("{:-^50s}".format(' Train '))
        agent.train_policy(episodes=200, n_roll_outs=10, save=save)

    if benchmark:
        """ check the results """
        print("{:-^50s}".format(' Benchmark '))
        agent.run_benchmark(episodes=25)

    if render:
        """ render one episode"""
        print("{:-^50s}".format(' Render '))
        agent.run_benchmark(episodes=2, render=True)

    return


if __name__ == '__main__':
    main(load=True, train=True, benchmark=True, save=True, render=False)
