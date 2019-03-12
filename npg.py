"""Module containing the core class of the Natural Policy Gradient
algorithm. However due to performance purposes the NPG has been changed
towards the Trust Region Policy Optimization algorithm.

:Date: 2019-03-11
:Version: 1
:Authors:
    - Cedric Derstroff
    - Janosch Moos
    - Kay Hansel
"""

import numpy as np
import torch as tr
import copy
from utilities.conjugate_gradient import conjugate_gradient as cg
from utilities.estimations import *


class NPG:
    """This class implements the natural policy gradient method TRPO.
    Compared to the traditional NPG algorithm as suggested by
    Rajeswara, Lowrey, Todorov and Kakade [1, 2] the TRPO has faster
    convergernce and a stronger constrain by adding a line search
    rechecking whether the KL-divergence constrain is satisfied. If the
    constrain is not satisfied, the line search reduces the update step
    to satisfy the constrain. Further TRPO introduces importance
    sampling to replace the sum over all actions.
    As such the TRPO is effective in optimizing large nonlinear policies
    such as neural networks [3].


    [1] Aravind Rajeswaran, Kendall Lowrey, Emanuel Todorov and
    Sham Kakade, Towards Generalization and Simplicity in Continuous
    Control, CoRR, 1703 (2017)

    [2] Sham Kakade, A Natural Policy Gradient, NIPS, 01, 1531-1538
    (2001)

    [3] John Schulman, Sergey Levine, Philipp Moritz, Michael I. Jordan
    and Pieter Abbeel, Trust Region Policy Optimization, CoRR, 1502
    (2015)

    Attributes
    -----------
    normalizer
        The normalizer to normalize the inputs to zero mean

    Methods
    -------
    do(env, policy, n_roll_outs)
        Performs a single update step of the algorithm
        1. draw a set of parameter samples
        2. Calculates policy gradients using weighted samples
        3. Updates parameters


    get_title()
        Generates a title containing all relevant parameters

    get_name()
        Returns the algorithms' name
    """

    def __init__(self, baseline, _delta=0.05, damping=1e-4,
                 _lambda=0.95, _gamma=0.98, normalizer=None):
        """
        :param baseline: The baseline represents an estimator for
            estimating the state-value function
        :type baseline: Baseline

        :param _delta: Learning rate
        :type _delta: float

        :param damping: Damping factor to increase stability of the
            conjugate gradient
        :type damping: float

        :param _lambda: Controls the bias and variance trade-off
        :type _lambda: float

        :param _gamma: Determines the scale of the value function
        :type _gamma: float

        :param normalizer: Normalizer for zero mean normalization of the
            observations
        :type normalizer: Normalizer
        """

        self.__delta = 2 * _delta
        self.__damping = damping
        self.__lambda = _lambda
        self.__gamma = _gamma
        self.__baseline = baseline
        self.normalizer = normalizer

    # Utility Functions
    # ===============================================================
    def __line_search(self, old_policy, new_policy, observations):
        obs = tr.from_numpy(observations).float()
        old_mean, old_log_std = old_policy.network(obs)
        old_std = tr.exp(old_log_std)

        new_mean, new_log_std = new_policy.network(obs)
        new_std = tr.exp(new_log_std)
        kl = (old_std ** 2 + (old_mean - new_mean) ** 2)
        kl /= (2.0 * new_std ** 2 + 1e-10)
        kl += new_log_std - old_log_std - 0.5
        kl_mean = tr.mean(kl.sum(1, keepdim=True)).detach().numpy()
        return kl_mean <= self.__delta

    # Main Functions
    # ===============================================================
    def do(self, env, policy, n_roll_outs):
        """Performs a single update step of the algorithm by first
        simulating n roll outs on the given environment. Afterwards the
        vanilla and natural gradient of the policy are calculated and
        applied under the constrained of the KL-divergence. In case the
        constrain is violated, a line search is performed to satisfy
        the constrain. For rare cases, where the line search is unable
        to satisfy the constrain in a set amount of iterations, the old
        policy is kept.

        :param env: The environment the simulations are run on
        :type env: Environment

        :param policy: The decision making policy
        :type policy: Policy

        :param n_roll_outs: The number of roll outs to perform
        :type n_roll_outs: int

        :return: Returns the episodic returns and the time steps of the
            simulations
        :rtype: array of float, array of float
        """

        print("log_std:", policy.network.log_std)

        trajectories = env.roll_out(policy,
                                    n_roll_outs=n_roll_outs,
                                    render=False,
                                    normalizer=self.normalizer)

        estimate_advantage(trajectories,
                           self.__baseline, self.__gamma, self.__lambda)

        # TODO do in one loop and check reshape
        observations = np.concatenate([t["observations"]
                                       for t in trajectories])
        actions = np.concatenate([t["actions"]
                                  for t in trajectories])
        advantages = np.concatenate([t["advantages"]
                                    for t in trajectories])

        # vanilla gradient
        with tr.no_grad():
            fixed_log_probs = policy.get_log_prob(observations, actions)
            fixed_policy = copy.deepcopy(policy)

        log_probs = policy.get_log_prob(observations, actions)
        action_losses = tr.from_numpy(advantages).float() * tr.exp(
            log_probs - fixed_log_probs)
        action_loss = action_losses.mean()

        vpg = tr.autograd.grad(action_loss,
                               policy.network.parameters(), retain_graph=True)
        vpg_grad = np.concatenate([v.contiguous().detach().view(-1).numpy()
                                   for v in vpg])

        # product inv(fisher) times vanilla gradient via conjugate grad
        # Inspired by:
        #
        def get_npg(v):
            damping = self.__damping
            kl = tr.mean(policy.get_kl(observations))
            grads = tr.autograd.grad(kl, policy.network.parameters(),
                                     create_graph=True)
            grads_flat = tr.cat([grad.view(-1) for grad in grads])
            kl_v = tr.sum(grads_flat * tr.from_numpy(v).float())
            grads_kl_v = tr.autograd.grad(kl_v, policy.network.parameters())
            flat_grad_grad_v = np.concatenate(
                [g.contiguous().view(-1).data.numpy() for g in grads_kl_v])
            return flat_grad_grad_v + v * damping

        npg_grad = cg(get_npg, vpg_grad)

        # update policy
        nominator = npg_grad.dot(get_npg(npg_grad))
        learning_rate = np.sqrt(self.__delta / nominator)
        current = policy.get_parameters()
        for i in range(100):
            new = current + 0.9 ** i * learning_rate * npg_grad
            policy.set_parameters(new)
            if self.__line_search(fixed_policy, policy, observations):
                break
            elif i == 99:
                policy.set_parameters(current)

        # update baseline
        estimate_value(trajectories, self.__gamma)
        self.__baseline.train(trajectories)

        # update normalizer
        if self.normalizer is not None:
            self.normalizer.update(trajectories)

        # calculate return values
        returns = np.asarray([np.sum(t["rewards"]) for t in trajectories])
        time_steps = np.array([t["time_steps"]
                               for t in trajectories]).sum() / n_roll_outs

        return returns, time_steps

    def get_title(self):
        """Generates a title containing all relevant parameters for
        plotting purposes.

        :return: the title for the plots
        :rtype str
        """

        return r"NPG $\gamma = {}, \lambda = {}, \delta = {} $" \
               "\nBaseline: {} with {} epochs"\
               .format(self.__gamma,
                       self.__lambda,
                       self.__delta/2,
                       self.__baseline.get_hidden_dim(),
                       self.__baseline.get_epochs())

    @staticmethod
    def get_name():
        """Returns the algorithms' name

        :return: 'NPG'
        :rtype str
        """
        return 'NPG'
