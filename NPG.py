import numpy as np
import torch as tr
from utilities.Conjugate_gradient import conjugate_gradient as cg

#######################################
# NPG
#######################################


class NPG:

    """ Init """
    """==============================================================="""
    def __init__(self, _delta=0.05, damping=1e-4):
        self.__delta = 2*_delta
        self.damping = damping

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
        nominator = vpg_grad.T @ npg_grad + 1e-20
        learning_rate = np.sqrt(self.__delta / nominator)
        current = policy.get_parameters()
        new = current + learning_rate * npg_grad
        policy.set_parameters(new)

        new_log_prob = policy.get_log_prob(observations, actions)
        kl = tr.exp(new_log_prob) * (new_log_prob - fixed_log_probs)
        if tr.mean(kl.sum(1, keepdim=True)) >= self.__delta:
            print(True)
            print(tr.mean(kl.sum(1, keepdim=True)) - self.__delta)
            # for i in range(10):
            #     new = current + (0.9 ** i )*learning_rate * npg_grad
            #     policy.set_parameters(new)
            #
            #     new_log_prob = policy.get_log_prob(observations, actions)
            #     kl = tr.exp(new_log_prob) * (new_log_prob - fixed_log_probs)
            #     if tr.mean(kl.sum(1, keepdim=True)) <= self.__delta:
            #         print(i)
            #         break
        return







