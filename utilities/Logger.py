import numpy as np

#######################################
# Log data
#######################################


class Logger:

    """ Init """
    """==============================================================="""
    def __init__(self):
        self.logger = []

    """ Main Functions """
    """==============================================================="""
    def log_data(self, trajectories, policy_parameters,
                 delta_t_c, delta_t_p, delta_t_e):

        """ get rewards """
        rewards = np.asarray([np.sum(t["rewards"]) for t in trajectories])
        r_mean = rewards.mean()
        r_std = rewards.std()
        r_max = rewards.max()
        r_min = rewards.min()

        """ get time steps """
        time_steps = np.zeros(len(trajectories))
        for i, t in enumerate(trajectories):
            time_steps[i] = len(t["rewards"])
        t_mean = time_steps.mean()
        t_std = time_steps.std()
        t_max = time_steps.max()
        t_min = time_steps.min()

        episode = dict(
            roll_outs=np.array(len(trajectories)),
            reward_mean=np.array(r_mean).reshape(-1, 1),
            reward_std=np.array(r_std).reshape(-1, 1),
            reward_max=np.array(r_max).reshape(-1, 1),
            reward_min=np.array(r_min).reshape(-1, 1),
            time_mean=np.array(t_mean).reshape(-1, 1),
            time_std=np.array(t_std).reshape(-1, 1),
            time_max=np.array(t_max).reshape(-1, 1),
            time_min=np.array(t_min).reshape(-1, 1),
            time_episode=np.array(delta_t_e).reshape(-1, 1),
            time_critic=np.array(delta_t_c).reshape(-1, 1),
            time_policy=np.array(delta_t_p).reshape(-1, 1),
            policy_parameters=policy_parameters
            )

        self.logger.append(episode)
        return

