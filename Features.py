import numpy as np
import gym
import sklearn.pipeline
from sklearn.kernel_approximation import RBFSampler



class RbfFeatures:

    def __init__(self, env):
        env.reset()
        self.observation_examples = []
        for i in range(300):
            s, r, d, _ = env.step(1)
            self.observation_examples.append(s)

        # Create radial basis function sampler to convert states to features for nonlinear function approx
        self.featurizer = sklearn.pipeline.FeatureUnion([
            ("rbf1", RBFSampler(gamma=5.0, n_components=50)),
            ("rbf2", RBFSampler(gamma=2.0, n_components=50)),
            ("rbf3", RBFSampler(gamma=1.0, n_components=50)),
            ("rbf4", RBFSampler(gamma=0.5, n_components=50))
        ])
        # Fit featurizer to our samples
        self.featurizer.fit(np.array(self.observation_examples))

    def featurize_state(self, state):
        # Transform states
        featurized = self.featurizer.transform([state])
        return featurized
