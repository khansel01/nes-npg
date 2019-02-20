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
    def log_data(self, returns, time_steps, roll_outs, policy_parameters):

        """ get rewards """
        r_mean = returns.mean()
        r_std = returns.std()
        r_max = returns.max()
        r_min = returns.min()

        """ get time steps """
        t_mean = time_steps.mean()
        t_std = time_steps.std()
        t_max = time_steps.max()
        t_min = time_steps.min()

        episode = dict(
            roll_outs=roll_outs,
            reward_mean=np.array(r_mean).reshape(-1, 1),
            reward_std=np.array(r_std).reshape(-1, 1),
            reward_max=np.array(r_max).reshape(-1, 1),
            reward_min=np.array(r_min).reshape(-1, 1),
            time_mean=np.array(t_mean).reshape(-1, 1),
            time_std=np.array(t_std).reshape(-1, 1),
            time_max=np.array(t_max).reshape(-1, 1),
            time_min=np.array(t_min).reshape(-1, 1),
            policy_parameters=policy_parameters
            )

        self.logger.append(episode)
        return

