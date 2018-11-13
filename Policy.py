import gym
import torch as tr
import numpy as np

class Policy:
    # Initialize class Policy
    def __init__(self, input, output):
       self.input = input
       self.output = output


    def get_action(self, observation):
