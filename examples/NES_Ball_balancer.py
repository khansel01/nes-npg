import torch as tr
import numpy as np
from agent import Agent
from nes import NES
from models.nn_policy import Policy
from utilities.Environment import Environment
import pickle
import os

#######################################
# Environment
#######################################


def main(load: bool = False, train: bool = False, benchmark: bool = False,
         save: bool = False, render: bool = True):
    """ set seed """
    # np.random.seed(0)
    # tr.manual_seed(0)

    """ define the environment """
    gym_env = 'BallBalancerSim-v0'
    env = Environment(gym_env)
    print("{:-^50s}".format(' Start {} '.format(gym_env)))

    if load:
        """ load pretrained policy, algorithm from data """
        print("{:-^50s}".format(' Load '))
        path = os.getcwd() + "/trained_data/{}_NES.p".format(env.to_string())

        pickle_in = open(path, "rb")

        policy, algorithm = pickle.load(pickle_in)
    else:
        """ create policy, algorithm """
        print("{:-^50s}".format(' Init '))
        policy = Policy(env, hidden_dim=(10,))

        """ create NPG-algorithm """
        algorithm = NES(policy.length)

    """ create agent """
    agent = Agent(env, policy, algorithm)

    if train:
        """ train the policy """
        print("{:=^50s}".format(' Train '))
        agent.train_policy(episodes=500, n_roll_outs=1, save=save)

    if benchmark:
        """ check the results """
        print("{:-^50s}".format(' Benchmark '))
        agent.run_benchmark(episodes=100)

    if render:
        """ render one episode"""
        print("{:-^50s}".format(' Render '))
        agent.run_benchmark(episodes=2, render=True)

    return


if __name__ == '__main__':
    main(load=True, train=False, benchmark=True, save=True, render=False)
