import numpy as np
import matplotlib.pyplot as plt
import torch as tr
import torch.nn.functional as F
import torch.nn as nn

#######################################
# baselina
#######################################

class baseline(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(baseline, self).__init__()
        # Calling Super Class's constructor
        self.linear1 = nn.Linear(input_dim, int(input_dim*2))
        self.linear2 = nn.Linear(int(input_dim*2), output_dim)
        # nn.linear is defined in nn.Module
        self.criterion = nn.MSELoss()
        self.optimiser = tr.optim.SGD(self.parameters(), lr=0.1)
        self.loss = 1

    def forward(self, x):
        # Here the forward pass is simply a linear function
        out = F.relu(self.linear1(x))
        return F.relu(self.linear2(out))

    def train(self, x):
        values = self.__estimate_value(self, x)
        while(self.loss>0.00001):
            # increase the number of epochs by 1 every time
            inputs = tr.from_numpy(x).float()
            labels = tr.from_numpy(values).float()
            self.optimiser.zero_grad()
            outputs = self.forward(inputs)
            self.loss = self.criterion(outputs, labels)
            self.loss.backward()  # back props
            self.optimiser.step()  # update the parameter
        print("loss: ", self.loss)
        return

    def __estimate_value(self, values):
        for i in range(len(values)-2, -1, -1):
            values[i] = values[i] + self.__gamma * values[i + 1]
        return values
