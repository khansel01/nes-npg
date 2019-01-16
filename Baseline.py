import numpy as np
import torch as tr
import torch.nn as nn

#######################################
# Baseline
#######################################


class Baseline:
    def __init__(self, env, hidden_dim: tuple=(128, 128),
                 activation: nn=nn.Tanh, lr: float=0.1):

        #   init
        self.input_dim = env.obs_dim() + 2
        self.output_dim = env.act_dim()
        self.hidden_dim = hidden_dim
        self.lr = lr

        #   create nn
        self.act = activation()
        self.network = nn.Sequential()
        hidden_dim = self.input_dim
        i = 0
        for i, next_hidden_dim in enumerate(self.hidden_dim):
            self.network.add_module('linear' + i.__str__(),
                                    nn.Linear(hidden_dim, next_hidden_dim))
            self.network.add_module('activation' + i.__str__(), self.act)
            hidden_dim = next_hidden_dim
        self.network.add_module('linear' + (i+1).__str__(),
                                nn.Linear(hidden_dim, self.output_dim))

        #   set last layer weights and bias small
        for p in list(self.network.parameters())[-2:]:
            p.data *= 1e-2

        # Create Loss function and SGD Optimizer
        self.loss_fct = nn.MSELoss()
        self.optimizer = tr.optim.SGD(self.network.parameters(), lr=self.lr)
        self.loss = 1

    def train(self, trajectories, eps=1e-6):
        data, values = self.__get_data(trajectories)
        while self.loss > eps:
            # increase the number of epochs by 1 every time
            inputs = tr.from_numpy(data).float()
            labels = tr.from_numpy(values).float()
            self.optimizer.zero_grad()
            predicted = self.network(inputs)
            self.loss = self.loss_fct(predicted, labels)
            self.loss.backward()  # back props
            self.optimizer.step()  # update the parameter
        return

    def predict(self, trajectories):
        x, _ = self.__get_data(trajectories)
        x = tr.from_numpy(x).float()
        return self.network(x).detach().numpy().squeeze()

    @staticmethod
    def __get_data(trajectories):
        if isinstance(trajectories, list):
            obs = np.concatenate([t["observations"]
                                  for t in trajectories])
            act = np.concatenate([t["actions"]
                                  for t in trajectories]).reshape(-1, 1)
            rew = np.concatenate([t["rewards"]
                                  for t in trajectories]).reshape(-1, 1)
            if "values" in trajectories:
                val = np.concatenate([t["values"]
                                      for t in trajectories]).reshape(-1, 1)
            else:
                val = np.zeros_like(rew).reshape(-1, 1)
        else:
            obs = trajectories["observations"]
            act = trajectories["actions"].reshape(-1, 1)
            rew = trajectories["rewards"].reshape(-1, 1)
            if "values" in trajectories:
                val = trajectories["values"].reshape(-1, 1)
            else:
                val = np.zeros_like(rew).reshape(-1, 1)

        # return obs, val
        return np.concatenate((obs, act, rew), axis=1), val
