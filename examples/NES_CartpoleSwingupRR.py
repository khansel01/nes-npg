from NES import *
from utilities.Environment import Environment
from models.NN_GaussianPolicy import Policy
from Agent import Agent
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
    gym_env = 'CartpoleSwingRR-v0'
    env = Environment(gym_env, clip=6)
    print("{:=^50s}".format(' Start {} '.format(gym_env)))

    if load:
        """ load pretrained policy, algorithm from data """
        print("{:=^50s}".format(' Load '))
        # path = "trained_data/CartpoleSwingShort-v0_10000_6.0_NES.p"
        path = os.getcwd() + "/trained_data/{}_NES.p".format(env.to_string())

        pickle_in = open(path, "rb")

        policy, algorithm = pickle.load(pickle_in)
    else:
        """ create policy, algorithm """
        print("{:=^50s}".format(' Init '))
        policy = Policy(env, hidden_dim=(10,))

        algorithm = NES(policy.length, sigma_init=1.0)

    """ create agent """
    agent = Agent(env, policy, algorithm)

    if train:
        """ train the policy """
        print("{:=^50s}".format(' Train '))
        agent.train_policy(episodes=30, n_roll_outs=1, save=save)

    if benchmark:
        """ check the results """
        print("{:=^50s}".format(' Benchmark '))
        agent.run_benchmark(episodes=10)

    if render:
        """ render one episode"""
        print("{:=^50s}".format(' Render '))
        agent.run_benchmark(episodes=1, render=True)

    return


if __name__ == '__main__':
    main(load=True, train=True, benchmark=True, save=True, render=False)
