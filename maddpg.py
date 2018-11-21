"""
This file defines the module for MADDPG agent.
The modules for OU noise and replay buffer are
also defined.

"""

# import libraries
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# import modules defining actor/critic neural networks
# as well as replay buffer and OU noise generator
from model import Actor, Critic
from utility import ReplayBuffer, OUNoise

# hyper parameters
GAMMA = 0.998 # discount factor
TAU =  0.1 # parameter for soft-update
LR_ACTOR = 0.0001 # learning rate for the actor model
LR_CRITIC = 0.001 # learning rate for the critic model
BUFFER_SIZE = 100000 # the buffer size for the replay buffer
BATCH_SIZE = 256 # batch size for sampling from the buffer
UPDATE_EVERY = 1 # how often the learning process runs
NUM_LEARNING = 1 # how many time the updates are done for each learning process
WEIGHT_DECAY_ACTOR = 0.0 # weight decay for the optimizer of the actor
WEIGHT_DECAY_CRITIC = 0.0 # weight decay for the optimizer of the critic

EPSILON = 1.0 # initial scale factor for noise
EPSILON_DECAY = 0.999 # decay rate of the scale factor for noise
EPSILON_MIN = 0.3 # minimum value of the scale factor for noise

# choose the device for evaluation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

class DDPG_Agent():
    """
    This module defines a DDPG agent (both actor and critic models).

    """

    def __init__(self, state_size, action_size, num_agents, seed):
        """
        For initialization.
        (input)
        - state_size (int): size of a state for an agent (=24)
        - size_action (int): size of an action for an agent (=2)
        - num_agents (int): the number of the agents to play with (=2).

        """

        self.seed = torch.manual_seed(seed) # random seed
        self.action_size = action_size # action size
        self.state_size = state_size # state size
        self.num_agents = num_agents # number of agents

        # actor model (local and target networks) and optimizer
        self.actor_local = Actor(state_size, action_size, seed).to(device)
        self.actor_target = Actor(state_size, action_size, seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr = LR_ACTOR, weight_decay = WEIGHT_DECAY_ACTOR)

        # critic model (local and target networks) and optimizer
        self.critic_local = Critic(state_size, action_size, num_agents, seed).to(device)
        self.critic_target = Critic(state_size, action_size, num_agents, seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr = LR_CRITIC, weight_decay = WEIGHT_DECAY_CRITIC)

        # noise for action
        self.noise = OUNoise(action_size, seed)

        # scale factor for the noise
        self.epsilon = EPSILON

        # set equal the weights of the local models and those of the target ones
        self.soft_update(self.actor_local, self.actor_target, 1.0)
        self.soft_update(self.critic_local, self.critic_target, 1.0)

    def act(self, state, add_noise = True):
        """
        This method is for selecting an action based on the policy
        determined by the local actor model (with a noise generated
        by the OU process added).

        (input)
        - state (tensor, dim = 24): a state tensor for an agent
        - add_noise (bool): if add a noise or not

        (output)
        - action (tensor, dim = 2): an action tensor for an agent

        """

        # move the state tensor to the device
        state = torch.from_numpy(state).float().to(device)

        # select an action based on the policy defined by the local actor model
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()

        # # add noise to the selected action
        if add_noise == True:
            noise_sample = self.noise.sample() * self.epsilon
            action += noise_sample

        # update the scale factor for the noise
        self.epsilon = max(self.epsilon*EPSILON_DECAY, EPSILON_MIN)

        return np.clip(action, -1, 1)

    def soft_update(self, local_model, target_model, tau):
        """
        This method is for the soft update of the weights of the target neural networks.

        (input)
        - local_model (Actor/Critic module): local actor/critic model
        - target_model (Actor/Ciric module): target actor/critic model

        """

        for l_param, t_param in zip(local_model.parameters(), target_model.parameters()):
            t_param.data.copy_(tau * l_param.data + (1.0 - tau) * t_param.data)


class MADDPG_Agent():
    """
    This module is for defining the MADDPG Agent.

    """

    def __init__(self, state_size, action_size, num_agents, seed):
        """
        For initialization.

        (input)
        - state_size (int): size of a state for a DDPG agent (=24)
        - action_size (int): size of an action for a DDPG agent (=2)
        - num_agents (int): total number of DDPG agents to play with (=2)
        - seed (int): random seed

        """

        self.seed = torch.manual_seed(seed) # random seed
        self.state_size = state_size # state size (for each DDPG agent)
        self.action_size = action_size # action size (for eadch DDPG agent)
        self.num_agents = num_agents # number of DDPG agents

        # list of DDPG agents
        self.maddpg_agents = [DDPG_Agent(state_size, action_size, num_agents, seed) for _ in range(num_agents)]

        # replay buffer
        self.buffer = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, seed)

        # time step
        self.t_step = 0

    def reset(self):
        """
        This method is used for resetting noises (used at the beginning of each episode).

        """
        # reset time step
        self.t_step = 0

        # initialize the noise
        for i in range(self.num_agents):
            self.maddpg_agents[i].noise.reset()

    def act_all(self, full_state, add_noise = True):
        """
        This method is for getting actions for all the DDPG agents
        by using the policy determined by the actor local models.

        (input)
        - full_state (numpy array, float, dim = [batch_size, number of agents, state size]:
            list of states for all the DDPG agents
        (output)
        - a list storing the actions for all the DDPG agents (each action is a numpy array)
            (dim = [batch size, number of agents, action size])
        """

        actions= []

        # if avoiding slicing
        for state, ddpg_agent in zip(full_state, self.maddpg_agents):
            action = ddpg_agent.act(state, add_noise)
            actions.append(action)

        return actions

    def step(self, full_state, full_action, full_reward, full_next_state, full_done, is_learning = True):
        """
        This method is used for storing a new experience and run the learning process.

        (input)
        - full_states (tensor, float, dim = [number of agents, state size]):
            a tensor storing states of all the agents
        - full_action (tensor, float, dim = [number of agents, action size]):
            a tensor stroing actions of all the agents
        - full_reward (tensor, float, dim = [number of agents, 1]):
            a tensor soring rewards of all the agents
        - full_next_states (tensor, float, dim = [number of agents, state size]):
            a tensor storing next states of all the agents
        - full_done (tensor, int, dim = [number of agents, 1]):
            a tensor storing integers describing if the episode is done or note for all the agents
        """

        # add the experience tuple of each ddpg agent to the replay buffer
        self.buffer.add(full_state, full_action, full_reward, full_next_state, full_done)

        self.t_step  = (self.t_step + 1) % UPDATE_EVERY

        # learning process
        if (self.t_step == 0) and is_learning:
            if len(self.buffer) > BATCH_SIZE:
                for _ in range (NUM_LEARNING):
                    for agent_id in range(self.num_agents):
                        self.learn(agent_id, GAMMA)
                    # update the target networks
                    self.soft_update_all(TAU)

    def learn(self, agent_id, gamma):
        """
        This method is used for the learning process of the action/critic local models.
        (input)
        - agent_id (int): the id number of the agent
        - gamma (float): discount factor

        """

        # sample experiences from the replay buffer
        experiences = self.buffer.sample(agent_id)
        states, full_states, actions, full_actions, rewards, next_states, full_next_states, dones = experiences

        # send the tensors to the device
        states = states.to(device)
        full_states = full_states.to(device)
        actions = actions.to(device)
        full_actions = full_actions.to(device)
        rewards = rewards.to(device)
        next_states = next_states.to(device)
        full_next_states = full_next_states.to(device)
        dones = dones.to(device)

        ### compute loss for the critic local model and back propagate
        # choose the next actions based on the target networks
        full_next_actions = []
        for i in range(self.num_agents):
            next_state_i =  full_next_states[:,i,:]
            next_action_i = self.maddpg_agents[i].actor_target(next_state_i)
            full_next_actions.append(next_action_i)
        full_next_actions = torch.cat(full_next_actions, dim=1).to(device)
        Q_target_next = self.maddpg_agents[agent_id].critic_target(full_next_states, full_next_actions)
        Q_target = rewards + gamma * Q_target_next * (1 - dones)
        Q_expected = self.maddpg_agents[agent_id].critic_local(full_states, full_actions)
        critic_loss = F.mse_loss(Q_expected, Q_target)

        # back propagation for the critic model
        self.maddpg_agents[agent_id].critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.maddpg_agents[agent_id].critic_local.parameters(), 1.0) # gradient clipping
        self.maddpg_agents[agent_id].critic_optimizer.step()

        ### compute loss for the actor local model and back propagate
        # compute the loss for the actor local model
        actions_pred = self.maddpg_agents[agent_id].actor_local(states)
        full_actions_pred = [full_actions[:, i, :] if i != agent_id else actions_pred for i in range(self.num_agents)]
        full_actions_pred = torch.cat(full_actions_pred, dim = 1).to(device)
        actor_loss = - self.maddpg_agents[agent_id].critic_local(full_states, full_actions_pred).mean()

        # back propagation for the actor model
        self.maddpg_agents[agent_id].actor_optimizer.zero_grad()
        actor_loss.backward()
        self.maddpg_agents[agent_id].actor_optimizer.step()


    def soft_update_all(self, tau):
        """
        This method is used for soft updating all the target models.
        (input)
        - tau (float): a paramter for the soft update
        """

        for i in range(self.num_agents):
            self.maddpg_agents[i].soft_update(self.maddpg_agents[i].actor_local, self.maddpg_agents[i].actor_target, tau)
            self.maddpg_agents[i].soft_update(self.maddpg_agents[i].critic_local, self.maddpg_agents[i].critic_target, tau)
