import numpy as np

"""Module containing Logger class for documenting episodic data"""


class Logger:
    """ The logger is used to document each episode for later evaluation
    (e.g. plotting)
    """

    def __init__(self):
        self.logger = []

    # Main Function
    # ===============================================================
    def log_data(self, returns, time_steps, roll_outs, policy_parameters):
        """Extracts relevant data from episodic returns, time steps etc. and
        saves them in a dictionary for later use
        """

        # get rewards
        r_mean = returns.mean()
        r_std = returns.std()
        r_max = returns.max()
        r_min = returns.min()

        # get time steps
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

