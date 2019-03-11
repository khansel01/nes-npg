import torch as tr
import numpy as np
from agent import Agent
from npg import NPG
from nes import NES
from models.nn_policy import Policy
from utilities.environment import Environment
from utilities import Helper
from models.baseline import Baseline
from utilities.normalizer import Normalizer
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
