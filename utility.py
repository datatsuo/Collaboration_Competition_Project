"""
This Python file defines the module OUNoise
used for generating noises for actions and
the module Replaybuffer which is used for store and sampling
experiences for the learning process.

"""

import numpy as np
import copy
import random
from collections import deque, namedtuple
import torch


class OUNoise():
    """
    This module is for generationg noises for the actions
    based on OU process.

    """

    def __init__(self, size, seed, mu = 0.0, theta = 0.15, sigma = 0.2): # 0.0, 0.15, 0.4
        """
        Initialization.
        (input)
        - size (int): size of noise
        - seed (int): random seed
        - mu, theta, sigma (float): parameters for the OU process
        """

        self.seed = random.seed(seed)
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma

        self.reset()

    def reset(self):
        """
        This method is for resetting the initial state for generating noises
        """
        self.state = copy.copy(self.mu)

    def sample(self):
        """
        This method is for generating noises with the OU process.
        (output)
        - noise (float, dim = size)
        """
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx

        return self.state

class ReplayBuffer():
    """
    This module defines the (standard) replay buffer
    for storing and sampling experiences

    """

    def __init__(self, buffer_size, batch_size, seed = 31):
        """
        For initilization.
        (input)
        - buffer_size (int): size of the replay buffer
        - batach_size (int): batch size for sampling
        - seed (int): random seed
        """

        self.seed = random.seed(seed) # set random seed
        self.buffer_size = buffer_size # buffer size
        self.batch_size = batch_size # batch size
        self.memory = deque(maxlen = self.buffer_size) # memory for storing experiences
        # define named tuple for storing an experience
        # each experience tuple contains info of all the agents.
        # For example, full_state contains states of all the agents
        field_names = ['full_state', 'full_action', 'full_reward', 'full_next_state', 'full_done']
        self.experience = namedtuple("Experience", field_names = field_names)

    def add(self, full_state, full_action, full_reward, full_next_state, full_done):
        """
        This method is for adding a new experience tuple to the replay buffer
        (input):
        - full_state: a list of states for all the agents
        - full_action: a list of actions for all the agents
        - full_reward: a list of rewards for all the agents
        - full_next_state: a list of next states for all the agents
        - full_done: a list of if an episode is done or not for all the agents
        """
        e = self.experience(full_state, full_action, full_reward, full_next_state, full_done)
        self.memory.append(e)

    def sample(self, agent_id):
        """
        This method is for sampling from the replay buffer.
        The sampled experiences are reshaped such that they can be used
        for training the agent_id-th DDPG agent.

        (input)
        - agent_id (int): id of the agent (= 0 or 1)
        (output)
        - states (tensor, float, dim =[batch_size, size of a state]): a tensor storing states for the agent_id-th agent
        - full_states (tensor, float, dim =[batch_size, number of agents, size of a state]): a tensor storing states for all the agents
        - actions (tensor, float, dim =[batch_size, size of an action]): a tensor storing actions for the agent_id-th agent
        - full_states (tensor, float, dim =[batch_size, number of agents, size of an action]): a tensor storing actions for all the agents
        - rewards (tensor, float, dim =[batch_size]): a tensor storing rewards for the agent_id-th agent
        - next_states (tensor, float, dim =[batch_size, size of a state]): a tensor storing next states for the agent_id-th agent
        - full_next_states (tensor, float, dim =[batch_size, number of agents, size of a state]): a tensor storing next states for all the agents
        - rewards (tensor, float, dim =[batch_size]): a tensor storing if an episode is done or not for the agent_id-th agent
        """
        experiences = random.sample(self.memory, k = self.batch_size)

        states = torch.from_numpy(np.vstack([e.full_state[agent_id] for e in experiences if e is not None])).float()
        actions = torch.from_numpy(np.vstack([e.full_action[agent_id] for e in experiences if e is not None])).float()
        rewards = torch.from_numpy(np.vstack([e.full_reward[agent_id] for e in experiences if e is not None])).float()
        next_states = torch.from_numpy(np.vstack([e.full_next_state[agent_id] for e in experiences if e is not None])).float()
        dones = torch.from_numpy(np.vstack([e.full_done[agent_id] for e in experiences if e is not None]).astype(np.uint8)).float()
        full_states = torch.from_numpy(np.vstack([[e.full_state] for e in experiences if e is not None])).float()
        full_actions = torch.from_numpy(np.vstack([[e.full_action] for e in experiences if e is not None])).float()
        full_next_states = torch.from_numpy(np.vstack([[e.full_next_state] for e in experiences if e is not None])).float()

        return (states, full_states, actions, full_actions, rewards, next_states, full_next_states, dones)

    def __len__(self):
        """
        This method lets len() to return the number of experiences stored in this replay buffer

        """
        return len(self.memory)
