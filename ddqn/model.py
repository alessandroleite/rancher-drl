import torch
import torch.nn as nn

import torch.nn.functional as F
from torch.nn import Linear

import numpy as np

class CriticNetwork(nn.Module):

    """Critic value model"""

    def __init__(self, input_size:int, output_size:int, seed:int, hidden_units=[400,300]):
        """
          Creates and initializes the critic network
        Args:
          input_size (int): dimension of the input layer (i.e., state size)
          output_size (int): dimension of the output layer (i.e., actions size)
          seed (int): random seed
          hidden_units (array): number of units of the hidden layers
        """
        super(CriticNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.fc1 = Linear(input_size, hidden_units[0])
        self.fc2 = Linear(output_size + hidden_units[0], hidden_units[1])
        self.fc3 = Linear(hidden_units[1], 1)
        # self.initialize_parameters()
        initialize_parameters([self.fc1, self.fc2], self.fc3)

    # def initialize_parameters(self):
    #     self.fc1.weight.data.uniform_(*initialize_hidden_layer(self.fc1))
    #     self.fc2.weight.data.uniform_(*initialize_hidden_layer(self.fc2))
    #     self.fc3.weight.data.uniform(-3e-3, 3e-3)
    #     self.fc3.bias.data.uniform(-3e-4, 3e-4)
    
    def forward (self, state, action):
        """
        Build a critic value network that maps state and action pairs to Q-values
        """
        xs = F.relu(self.fc1(state))
        x = torch.cat((xs, action.float()), dim=1)
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class ActorNetwork(nn.Module):

    def __init__(self, input_size, output_size, seed:int, hidden_units=[400,300]):
        
        super(ActorNetwork, self).__init__()

        self.seed = torch.manual_seed(seed)
        self.fc1 = Linear(input_size, hidden_units[0])
        self.fc2 = Linear(hidden_units[0], hidden_units[1])
        self.fc3 = Linear(hidden_units[1], output_size)
        initialize_parameters([self.fc1, self.fc2], self.fc3)
    
    def forward (self, state):
        """
        Build a actor policy network that maps states to actions
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        yhat = torch.tanh(self.fc3(x))

        return yhat


def initialize_hidden_layer(layer):
    """
    Initializes the given layer following a uniform distribution [-1/sqrt(f), 1/sqrt(f)], where 
    f if the fan-in of the given layer 
    """
    fan_in = layer.weight.data.size()[0]
    value = 1.0 / np.sqrt(fan_in)
    return (-value, value)


def initialize_parameters(hidden_layers, output_layer):

    for layer in hidden_layers:
        layer.weight.data.uniform_(*initialize_hidden_layer(self.fc1))
        
    output_layer.weight.data.uniform(-3e-3, 3e-3)
    output_layer.bias.data.uniform(-3e-4, 3e-4)