import numpy as np
import torch as tr
import torch.nn as nn

#######################################
# Baseline
#######################################


class Baseline:

    """ Init """
    """==============================================================="""
    def __init__(self, env, hidden_dim: tuple=(128, 128),
                 activation: nn=nn.Tanh, batch_size: int = 64,
                 epochs: int = 10, lr: float=0.1):

        """ init """
        self.input_dim = env.obs_dim()
        self.output_dim = env.act_dim()
        self.hidden_dim = hidden_dim
        self.act = activation
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr

        """ create nn """
        self.network = Network(self.input_dim, self.output_dim,
                               self.hidden_dim, self.act)

        """ Create Loss function and Adam Optimizer """
        self.loss_fct = nn.MSELoss()
        self.optimizer = tr.optim.SGD(self.network.parameters(), lr=self.lr)

    """ Utility Functions """
    """==============================================================="""
    @staticmethod
    def __get_data(trajectories):
        if isinstance(trajectories, list):
            obs = np.concatenate([t["observations"]
                                  for t in trajectories])
            rew = np.concatenate([t["rewards"]
                                  for t in trajectories]).reshape(-1, 1)
            if "values" in trajectories:
                val = np.concatenate([t["values"]
                                      for t in trajectories]).reshape(-1, 1)
            else:
                val = np.zeros_like(rew).reshape(-1, 1)
        else:
            obs = trajectories["observations"]
            rew = trajectories["rewards"].reshape(-1, 1)
            if "values" in trajectories:
                val = trajectories["values"].reshape(-1, 1)
            else:
                val = np.zeros_like(rew).reshape(-1, 1)

        return obs, val

    """ Main Functions """
    """==============================================================="""
    def train(self, trajectories):
        data, values = self.__get_data(trajectories)

        # values = (values - values.mean())/(values.std() + 1e-10)
        for e in range(self.epochs):
            permuted_idx = np.random.permutation(len(values))
            for batch in range(int(len(values)/self.batch_size)-1):
                idx = tr.LongTensor(permuted_idx[batch*64:(batch+1)*64])
                inputs = tr.from_numpy(data).float()[idx]
                labels = tr.from_numpy(values).float()[idx]

                self.optimizer.zero_grad()
                predicted = self.network(inputs)
                loss = self.loss_fct(predicted, labels)

                """ back propagation"""
                loss.backward()

                """ update parameters"""
                self.optimizer.step()
        return

    def predict(self, trajectories):
        obs, _ = self.__get_data(trajectories)
        return self.network(tr.from_numpy(obs).float())\
            .detach().numpy().squeeze()
        # val = self.network(tr.from_numpy(obs).float())\
        #     .detach().numpy().squeeze()
        # return val * val.std() + val.mean()



class Network(nn.Module):
    def __init__(self, input_dim: int = 1, output_dim: int=1,
                 hidden_dim: tuple=(128, 128), activation: nn=nn.Tanh):

        """ init """
        super(Network, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.act = activation()
        self.net = nn.Sequential()

        """ create NN """
        hidden_dim = self.input_dim
        i = 0
        for i, next_hidden_dim in enumerate(self.hidden_dim):
            self.net.add_module('linear' + i.__str__(),
                                nn.Linear(hidden_dim,
                                          next_hidden_dim))
            self.net.add_module('Batch' + i.__str__(),
                                nn.BatchNorm1d(next_hidden_dim))
            self.net.add_module('activation' + i.__str__(), self.act)
            hidden_dim = next_hidden_dim
        self.net.add_module('linear' + (i + 1).__str__(),
                            nn.Linear(hidden_dim, self.output_dim))

        """ set last layer weights and bias small """
        for p in list(self.net.parameters())[-2:]:
            p.data *= 1e-2

    def forward(self, x):
        value = self.net(x)
        return value
