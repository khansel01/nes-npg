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

    def __init__(self, env):
        env.reset()
        self.observation_examples = []
        for i in range(300):
            s, r, d, _ = env.step(np.asarray(10))
            self.observation_examples.append(s)

        # Create radial basis function sampler to convert states to features for nonlinear function approx
        self.featurizer = sklearn.pipeline.FeatureUnion([
            ("rbf1", RBFSampler(gamma=0.1, n_components=40)),
            ("rbf2", RBFSampler(gamma=0.5, n_components=40)),
            ("rbf3", RBFSampler(gamma=1.0, n_components=40)),
            ("rbf4", RBFSampler(gamma=1.5, n_components=40)),
            ("rbf5", RBFSampler(gamma=2.0, n_components=40)),
            # ("rbf6", RBFSampler(gamma=2.5, n_components=10)),
            # ("rbf7", RBFSampler(gamma=3.0, n_components=10)),
            # ("rbf8", RBFSampler(gamma=3.5, n_components=10)),
            # ("rbf9", RBFSampler(gamma=4.0, n_components=10)),
            # ("rbf10", RBFSampler(gamma=4.5, n_components=10)),
            # ("rbf11", RBFSampler(gamma=5.0, n_components=10)),
            # ("rbf12", RBFSampler(gamma=7.5, n_components=10)),
            # ("rbf13", RBFSampler(gamma=10.0, n_components=10)),
            # ("rbf14", RBFSampler(gamma=15.0, n_components=10)),
            # ("rbf15", RBFSampler(gamma=20.0, n_components=10)),
            # ("rbf16", RBFSampler(gamma=25.0, n_components=10)),
            # ("rbf17", RBFSampler(gamma=50.0, n_components=10)),
            # ("rbf18", RBFSampler(gamma=75.0, n_components=10)),
            # ("rbf19", RBFSampler(gamma=100.0, n_components=10)),
            # ("rbf20", RBFSampler(gamma=200.0, n_components=10))
        ])
        # Fit featurizer to our samples
        self.featurizer.fit(np.array(self.observation_examples))

    def featurize_fit(self, observations):
        self.featurizer.fit(np.array(observations))
        return

    def featurize_state(self, state):
        # Transform states
        featurized = self.featurizer.transform([state])
        return featurized


class RBFs:

    def __init__(self, obs_dim, size):
        self.P = np.random.randn(size, obs_dim)
        self.v = []
        self.phi = np.linspace(-np.pi, np.pi, size)
        self.eps = np.finfo(np.float32).eps.item()

    def get_rbfs(self, x):
        v = self.eps #  if self.v == [] else sum(self.v)/(len(self.v)+self.eps)
        y = np.sin((self.P @ x)/v + self.phi.T)
        return y[None, :]

    def update_v(self, x1, x2):
        self.v.append(x1 - x2)
        return

