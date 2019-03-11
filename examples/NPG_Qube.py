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
    gym_env = 'Qube-v0'
    env = Environment(gym_env)
    print("{:-^50s}".format(' Start {} '.format(gym_env)))

    if load:
        """ load pretrained policy, baseline, Normalizer from data """
        print("{:-^50s}".format(' Load '))
        path = "trained_data/{}_300_5.0_NPG.p".format(gym_env)
        # path = "trained_data/{}_npg.p".format(gym_env)
        pickle_in = open(path, "rb")

        policy, algorithm = pickle.load(pickle_in)
    else:
        """ create new policy, baseline, Normalizer """
        print("{:-^50s}".format(' Init '))
        policy = Policy(env, hidden_dim=(6, 6))

        baseline = Baseline(env, hidden_dim=(6, 6), epochs=10)

        normalizer = Normalizer(env)

        """ create NPG-algorithm """
        algorithm = NPG(baseline, 0.05, _gamma=0.9999, normalizer=normalizer)

    """ create agent """
    agent = Agent(env, policy, algorithm)

    if train:
        """ train the policy """
        print("{:-^50s}".format(' Train '))
        agent.train_policy(episodes=500, n_roll_outs=100, save=save)

    if benchmark:
        """ check the results """
        print("{:-^50s}".format(' Benchmark '))
        agent.run_benchmark(episodes=5)

    if render:
        """ render one episode """
        print("{:-^50s}".format(' Render '))
        agent.run_benchmark(episodes=2, render=True)

    return


if __name__ == '__main__':
    main(load=True, train=False, benchmark=True, save=False, render=False)
