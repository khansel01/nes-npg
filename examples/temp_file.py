import torch as tr
import numpy as np
from Agent import Agent
from NPG import NPG
from models.NN_GaussianPolicy import Policy
from utilities.Environment import Environment
from utilities import Helper
from models.Baseline import Baseline
from utilities.Normalizer import Normalizer
import pickle

gym_env = 'Qube-v0'
env = Environment(gym_env)
print("=================== Start {} ===================".format(gym_env))


""" load pretrained policy, baseline, Normalizer from data """
print("====================== Load ======================")
path = "{}_npg.p".format(gym_env)

pickle_in = open(path, "rb")

policy, baseline, normalizer = pickle.load(pickle_in)

""" create NPG-algorithm """
algorithm = NPG(baseline, 0.05, _gamma=0.996, normalizer=normalizer)


pickle_out = open("{}_npg.p".format(gym_env), "wb")
pickle.dump((policy, algorithm), pickle_out)
pickle_out.close()