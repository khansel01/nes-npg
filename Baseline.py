import numpy as np
import torch as tr
import torch.nn as nn

#######################################
# Baseline
#######################################


class Baseline:
    def __init__(self, input_dim, output_dim, lr=0.1):

        # Calling Super Class's constructor
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = int(100) if input_dim*2 <= 100 else int(input_dim*5)
        self.lr = lr

        # create nn
        self.model = nn.Sequential()
        self.model.add_module('linear1',
                              nn.Linear(self.input_dim, self.hidden_dim))
        self.model.add_module('relu0', nn.ReLU())
        self.model.add_module('linear2',
                              nn.Linear(self.hidden_dim, self.hidden_dim))
        self.model.add_module('relu1', nn.ReLU())
        self.model.add_module('linear3',
                              nn.Linear(self.hidden_dim, self.output_dim))

        # Create Loss function and SGD Optimizer
        self.loss_fct = nn.MSELoss()
        self.optimizer = tr.optim.SGD(self.model.parameters(), lr=self.lr)
        self.loss = 1

    def train(self, trajectories, eps=1e-6):
        data, values = self.__get_data(trajectories)
        while self.loss > eps:
            # increase the number of epochs by 1 every time
            inputs = tr.from_numpy(data).float()
            labels = tr.from_numpy(values).float()
            self.optimizer.zero_grad()
            predicted = self.model(inputs)
            self.loss = self.loss_fct(predicted, labels)
            self.loss.backward()  # back props
            self.optimizer.step()  # update the parameter
        return

    def predict(self, trajectories):
        x, _ = self.__get_data(trajectories)
        x = tr.from_numpy(x).float()
        return self.model(x).detach().numpy().squeeze()

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

        #return np.concatenate((obs, act, rew), axis=1), val
        return obs, val
