from REINFORCE import REINFORCE
from NES import NES
from utilities.Environment import Environment
from models.NN_GaussianPolicy import Policy
from models.Baseline import Baseline
from utilities.Normalizer import Normalizer
from Agent import Agent
import pickle
import numpy as np
import torch as tr
import os

#######################################
# Environment
#######################################


def main(load: bool = False, train: bool = False, benchmark: bool = False,
         save: bool = False, render: bool = True, horizon=5000):
    """ set seed """
    np.random.seed(10)
    tr.manual_seed(23)

    """ define the environment """
    gym_env = 'Levitation-v1'
    env = Environment(gym_env, horizon=horizon)
    print("{:-^50s}".format(' Start {} '.format(gym_env)))

    if load:
        """ load pretrained policy, algorithm from data """
        print("{:-^50s}".format(' Load '))
        path = os.getcwd() + "/trained_data/Levitation-v1_4_24.0_REINFORCE.p"
        # path = os.getcwd() + "/trained_data/{}_REINFORCE.p".format(env.to_string())

        pickle_in = open(path, "rb")

        policy, algorithm = pickle.load(pickle_in)
    else:
        """ create policy, algorithm """
        print("{:-^50s}".format(' Init '))
        policy = Policy(env, hidden_dim=(1,), log_std=10.0)

        # init_params = np.ones(policy.length) * np.random.randint(0, 44)
        # policy.set_parameters(init_params)

        """ create baseline """
        baseline = Baseline(env, hidden_dim=(1,), epochs=10)

        """ create Normalizer to scale the states/observations """
        normalizer = Normalizer(env)

        # algorithm = NES(policy.length)
        algorithm = REINFORCE(baseline, 0.999, _gamma=0.9, _lambda=0)

    """ create agent """
    agent = Agent(env, policy, algorithm)

    if train:
        """ train the policy """
        print("{:-^50s}".format(' Train '))
        agent.train_policy(episodes=1000, n_roll_outs=1, save=save)

    if benchmark:
        """ check the results """
        print("{:-^50s}".format(' Benchmark '))

        agent.run_benchmark(episodes=1)

    if render:
        """ render one episode"""
        print("{:-^50s}".format(' Render '))
        agent.run_benchmark(episodes=1, render=True)

    return


if __name__ == '__main__':
    # main(load=False, train=True, benchmark=True, save=True, render=False, horizon=4)
    main(load=True, train=False, benchmark=True, save=False, render=False)
