import numpy as np
import torch as tr
import copy
from utilities.Conjugate_gradient import conjugate_gradient as cg

#######################################
# NPG
#######################################


class NPG:

    """ Init """
    """==============================================================="""
    def __init__(self, _delta=0.05, damping=1e-4):
        self.__delta = 2 * _delta
        self.damping = damping

    """ Utility Functions """
    """==============================================================="""
    def line_search(self, old_policy, new_policy, observations):
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

    """ Main Functions """
    """==============================================================="""
    def do(self, trajectories, policy):

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

        """ product inv(fisher) times vanilla gradient via conjugate grad """
        def get_npg(v):
            damping = self.damping
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

        """ update policy """
        #nominator = vpg_grad.T @ npg_grad + 1e-20
        nominator = npg_grad.dot(get_npg(npg_grad))
        learning_rate = np.sqrt(self.__delta / nominator)
        current = policy.get_parameters()
        for i in range(100):
            new = current + 0.9 ** i * learning_rate * npg_grad
            policy.set_parameters(new)
            if self.line_search(fixed_policy, policy, observations):
                break
            elif i == 99:
                policy.set_parameters(current)
        return







