import numpy as np
import gym
import sklearn.pipeline
from sklearn.kernel_approximation import RBFSampler

######################################
# This feature calculation is taken from:
# https://medium.com/samkirkiles/
# reinforce-policy-gradients-from-scratch-in-numpy-6a09ae0dfe12

# Just for test purposes for now
######################################


class RbfFeatures:

    def __init__(self, env, policy):
        trajectories = env.roll_out(policy, amount=1)
        observations = np.concatenate([t["observations"]
                                       for t in trajectories])
        # Create radial basis function sampler to convert states
        # to features for nonlinear function approx
        self.featurizer = sklearn.pipeline.FeatureUnion([
            ("rbf1", RBFSampler(gamma=1.0, n_components=300)),
            #("rbf2", RBFSampler(gamma=2.0, n_components=10)),
            #("rbf3", RBFSampler(gamma=1.0, n_components=10)),
            #("rbf4", RBFSampler(gamma=0.5, n_components=10)),
            #("rbf5", RBFSampler(gamma=0.1, n_components=10))
            # ("rbf18", RBFSampler(gamma=75.0, n_components=10)),
            # ("rbf19", RBFSampler(gamma=100.0, n_components=10)),
            # ("rbf20", RBFSampler(gamma=200.0, n_components=10))
        ])
        # Fit featurizer to our samples
        self.featurizer.fit(np.array(observations))

    def featurize_state(self, state):
        # Transform states
        return self.featurizer.transform([state])


class RBFs:

    def __init__(self, obs_dim, size):
        self.P = np.random.randn(size, obs_dim)
        self.v = []
        self.phi = np.linspace(-np.pi, np.pi, size)
        self.eps = np.finfo(np.float32).eps.item()

    def get_rbfs(self, x):
        v = 1 if self.v == [] else sum(self.v[0])/(len(self.v[0])+self.eps)
        y = np.sin((self.P @ x)/v + self.phi.T)
        return y[None, :]

    def update_v(self, x1, x2):
        self.v.append(x1 - x2)
        return
