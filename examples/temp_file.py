import torch as tr
import numpy as np
from Agent import Agent
from NPG import NPG
from NES import NES
from models.NN_GaussianPolicy import Policy
from utilities.Environment import Environment
from utilities import Helper
from models.Baseline import Baseline
from utilities.Normalizer import Normalizer
import pickle

# TODO Delete me!!

title = "Pendulum-v0_npg.p"

""" load pretrained policy, baseline, Normalizer from data """
print("====================== Load ======================")
path = "{}".format(title)

pickle_in = open(path, "rb")

policy, baseline, normalizer = pickle.load(pickle_in)
# policy = pickle.load(pickle_in)

""" create NPG-algorithm """
algorithm = NPG(baseline, 0.05, _gamma=0.996, normalizer=normalizer)
# algorithm = NES(policy.length, sigma_init=1.0)

pickle_out = open("{}".format(title), "wb")
pickle.dump((policy, algorithm), pickle_out)
pickle_out.close()
