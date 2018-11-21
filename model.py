"""
This file is for defining neural networks for
actor and critic models.

"""
# import libraries
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def hidden_init(layer):
    """
    This function is used for initialization of weights.

    (input)
    - layer of neural network
    (output)
    - [-lim, lim] where lim = (1/N)**0.5 where N is the number of units in the layer
    """
    fan_in = layer.weight.data.size()[0]
    lim = np.sqrt(1.0/fan_in)

    return [-lim, lim]

class Actor(nn.Module):
    """
    This module defines the neural network for the actor model.
    This network takes a state as an input and returns unnormalized probability
    to take possible actions.

    MEMO: modify the input shapes
    """

    def __init__(self, state_size, action_size, seed):
        """
        For initialization.
        (input)
        - state_size (int): size of a state for a DDPG agent (=24)
        - action_size (int): size of an action for a DDPG agent (=2)
        - num_agents (int): total number of agents to play with (=2)
        - seed (int): random seed

        """

        super(Actor, self).__init__()

        self.seed = torch.manual_seed(seed) # random seed
        self.state_size = state_size # state size for each agent
        self.action_size = action_size # action size for each agent
        self.hidden_units = [128,128] # the number of units for the hidden layers

        # layers
        self.fc1 = nn.Linear(self.state_size, self.hidden_units[0])
        self.fc2 = nn.Linear(self.hidden_units[0], self.hidden_units[1])
        self.fc3 = nn.Linear(self.hidden_units[1], self.action_size)

        self.reset_parameters()

    def reset_parameters(self):
        """
        This method is for initializing weights of the neural network.

        """

        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """
        Forwarding.

        (input)
        - state (tensor, float, dim=[batch size, state size]):
            a state tensor for an agent
        (output)
        - x (tensor, float, dim=[batch size, action size]):
            a action tensor for an agent
        """

        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.tanh(self.fc3(x))

        return x



class Critic(nn.Module):
    """
    This module defines the neural network for the critic model.
    This network takes state and action as an input and then returns
    the Q-value.

    MEMO: modify the input shapes
    """

    def __init__(self, state_size, action_size, num_agents, seed):
        """
        For initialization.
        (input)
        - state_size (int): size of a state for a DDPG agent (=24)
        - action_size (int): size of an action for a DDPG agent (=2)
        - num_agents (int): total number of agents to play the game with (=2)
        - seed (int): random seed

        """

        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed) # set random seed
        self.state_size = state_size
        self.action_size = action_size
        self.full_state_size = state_size * num_agents # size of the states for all the agents
        self.full_action_size = action_size * num_agents # size of the actions for all the agents
        self.hidden_units = [256, 128] # the number of units for the hidden layers

        # layers
        self.fc1 = nn.Linear(self.full_state_size + self.full_action_size, self.hidden_units[0])
        self.fc2 = nn.Linear(self.hidden_units[0], self.hidden_units[1])
        self.fc3 = nn.Linear(self.hidden_units[1], 1)

        self.reset_parameters()

    def reset_parameters(self):
        """
        This method is for initializing weights of the neural network.

        """

        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, full_state, full_action):
        """
        Forwarding.

        (input)
        - full_state_fl (tensor, float, dim=[batch size, (state size)*(number of agents)]):
          a tensor storing (flattened) states of all the agents.
        - full_action_fl (tensor, float, dim=[batch size, (action size)*(number of agents)]):
          a tensor storing (flattened) actions of all the agents.
        (output)
        - x (tensor, float, dim=[batch size, 1]): a tensor for Q-value

        """

        fs_flat = full_state.view(-1, self.full_state_size)
        fa_flat = full_action.view(-1, self.full_action_size)
        x = torch.cat([fs_flat, fa_flat], dim = 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x
