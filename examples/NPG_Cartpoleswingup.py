import torch as tr
import numpy as np
from agent import Agent
from Npg import NPG
from models.nn_policy import Policy
from utilities.Environment import Environment
from models.baseline import Baseline
from utilities.Normalizer import Normalizer
import pickle

#######################################
# Environment
#######################################


def main(load: bool = False, train: bool = False, benchmark: bool = False,
         save: bool = False, render: bool = True):
    """ set seed """
    np.random.seed(0)
    tr.manual_seed(0)

    """ define the environment """
    gym_env = 'CartpoleSwingShort-v0'
    env = Environment(gym_env, horizon=2000)
    print("{:=^50s}".format(' Start {} '.format(gym_env)))

    if load:
        """ load pretrained policy, baseline, Normalizer from data """
        print("{:=^50s}".format(' Load '))
        path = "trained_data/{}_2000_24.0_NPG.p".format(gym_env)

        pickle_in = open(path, "rb")

        policy, algorithm = pickle.load(pickle_in)
    else:
        """ create new policy, baseline, Normalizer """
        print("{:=^50s}".format(' Init '))
        policy = Policy(env, hidden_dim=(6, 6), log_std=1)

        baseline = Baseline(env, hidden_dim=(6, 6), epochs=10)

        normalizer = Normalizer(env)

        """ create NPG-algorithm """
        gamma = 0.999999
        algorithm = NPG(baseline, 0.01, _gamma=gamma, normalizer=normalizer)

    """ create agent """
    agent = Agent(env, policy, algorithm)

    if train:
        """ train the policy """
        print("{:=^50s}".format(' Train '))
        agent.train_policy(episodes=1000, n_roll_outs=50, save=save)

    if benchmark:
        """ check the results """
        print("{:=^50s}".format(' Benchmark '))
        agent.run_benchmark()

    if render:
        """ render one episode """
        print("{:=^50s}".format(' Render '))
        agent.run_benchmark(episodes=1, render=True)

    return


if __name__ == '__main__':
    main(load=False, train=True, benchmark=True, save=True, render=True)
