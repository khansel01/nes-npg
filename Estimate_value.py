import numpy as np

#######################################
# Estimate Values
#######################################


# TODO estiamte value and estimate advantage fct
def estimate_value(trajectories, _gamma):
    for t in trajectories:
        rewards = t["rewards"]
        values = np.zeros_like(rewards)
        for i in range(len(values) - 1, -1, -1):
            values[i] = rewards[i] if i == len(values) - 1 \
                else rewards[i] + _gamma * values[i + 1]
        # values = (values - np.mean(values)) / (
        #           np.std(values) + self.__eps)
        t["values"] = values
    return
