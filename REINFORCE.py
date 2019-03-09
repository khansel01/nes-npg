import numpy as np
import torch as tr
import copy
from utilities.Estimations import *

#######################################
# REINFROCE
#######################################


class REINFORCE:

    """ Init """
    """==============================================================="""
    def __init__(self, baseline, _delta=0.05, damping=1e-4,
                 _lambda=0.95, _gamma=0.98, normalizer=None):

        self.learning_rate = _delta
        self.__delta = 2 * _delta
        self.damping = damping
        self.__lambda = _lambda
        self.__gamma = _gamma
        self.baseline = baseline
        self.normalizer = normalizer

    """ Utility Functions """
    """==============================================================="""
    @staticmethod
    def get_name():
        return 'REINFORCE'

    """ Main Functions """
    """==============================================================="""
    def do(self, env, policy, n_roll_outs):

        print("log_std:", policy.network.log_std)

        trajectories = env.roll_out(policy,
                                    n_roll_outs=n_roll_outs,
                                    render=False,
                                    normalizer=self.normalizer)

        estimate_advantage(trajectories,
                           self.baseline, self.__gamma, self.__lambda)

        observations = np.concatenate([t["observations"]
                                       for t in trajectories])
        actions = np.concatenate([t["actions"]
                                  for t in trajectories]).reshape(-1, 1)
        advantages = np.concatenate([t["advantages"]
                                    for t in trajectories]).reshape(-1, 1)

        """ vanilla gradient """
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

        """ update policy """
        current = policy.get_parameters()
        new = current + self.learning_rate * vpg_grad
        policy.set_parameters(new)

        """ update baseline """
        estimate_value(trajectories, self.__gamma)
        self.baseline.train(trajectories)

        """ update normalizer """
        if self.normalizer is not None:
            self.normalizer.update(trajectories)

        """ calculate return values """
        returns = np.asarray([np.sum(t["rewards"]) for t in trajectories])
        time_steps = np.array([t["time_steps"]
                               for t in trajectories]).sum() / n_roll_outs

        return returns, time_steps

    def get_title(self):
        return "NPG \u03B3 = {}, \u03BB = {}, \u03B4 = {} \n" \
               "Baseline: {} with {} epochs" .format(self.__gamma,
                                                     self.__lambda,
                                                     self.__delta,
                                                     self.baseline.hidden_dim,
                                                     self.baseline.epochs)






