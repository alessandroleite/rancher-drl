import random

import numpy as np

import torch
import torch.nn.functional as F
from torch.optim import Adam

from ddqn.model  import ActorNetwork, CriticNetwork
from ddqn.buffer import ReplayBuffer
from ddqn.noise  import Ornstein

device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")

class Agent:
    """DDPG agent that interacts and learns from the environment. """

    def __init__(self, state_size:int, action_size:int, n_agents:int, seed:int, lr_actor=1e-4, lr_critic=1e-3, batch_size=64, gamma=0.99, update_frequency=4):
      """
      Initialize a DDPG agent
      """

      self.state_size = state_size
      self.action_size = action_size
      self.seed = random.seed(seed)
      self.lr_actor = lr_actor
      self.lr_critic = lr_critic
      self.batch_size = batch_size
      self.gamma = gamma
      self.update_frequency = update_frequency

      # Local actor network
      self.local_actor = ActorNetwork(state_size, action_size, seed).to(device)
      self.target_actor = ActorNetwork(state_size,action_size, seed).to(device)
      # Actor optimizer
      self.actor_optimizer = Adam(self.local_actor.parameters(), lr=self.lr_actor)


      # Critic network
      self.local_critic = CriticNetwork(state_size, action_size, seed)
      self.target_critic = CriticNetwork(state_size, action_size, seed)
      self.critic_optimizer = Adam(self.local_critic.parameters(), lr=self.lr_critic, weight_decay=1e-2)

      # Noise process
      self.noise = Ornstein((n_agents, action_size), seed)

      # Replay memory
      self.memory = ReplayBuffer(1e6, seed)

      # Initialize the time step (for every update_frequency steps)
      self.t_step = 0

    def step(self, states, actions, rewards, next_states, dones):
      
      for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
        self.memory.add(state, action, reward, next_state, done)

      self.t_step = (self.t_step + 1) % self.update_frequency

      if len(self.memory) > self.batch_size:
        experiences = self.memory.sample(self.batch_size)
        self.learn(experiences, self.gamma)

    def learn(self, experiences, gamma, tau=1e-3):

      states, actions, rewards, next_states, dones = experiences

      # Get predicted next-state actions and Q values from the target model 
      actions_next = self.target_actor(next_states)
      # Get the predict Q-values from the target model
      Q_targets_next = self.target_critic(next_states, actions_next)
      # Computes the Q targets for current state 
      Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
      # Computes the critic loss: L = 1/n \sum{(y_i - Q(s_i, a_i | \theta^Q))^2}
      Q_expected = self.local_critic(next_states, actions_next)
      critic_loss = F.mse_loss(Q_expected, Q_targets)
      # Minimizes the loss
      self.critic_optimizer.zero_grad()
      critic_loss.backward()
      self.critic_optimizer.step()

      actions_hat = self.local_actor(states)
      actor_loss = -self.local_critic(states, actions_hat).mean()
      self.actor_optimizer.zero_grad()
      actor_loss.backward()
      self.actor_optimizer.step()


      self.soft_update(self.local_critic, self.target_critic, tau)
      self.soft_update(self.local_critic, self.target_actor, tau)

    def soft_update(self, local_model, target_model, tau):
      """
      Soft update model parameters.

      θ_target = τ*θ_local + (1-τ)*θ_target

      Args:
        local_model (PyTorch model): from where the weights will be copied from
        target_model (PyTorch model): to where the weights will be copied to
        tau (float): interpolation parameter
      """

      for local_params, target_params in zip(local_model.parameters(), target_model.parameters()):
        tensor = tau * local_params.data + (1-tau) * target_params.data
        target_params.data.copy_(tensor)

    def act(self, states, add_noise=True):
      
      states = torch.from_numpy(states).float().to(device)
      self.local_actor.eval()

      with torch.no_grad():
        actions = self.local_actor(states).cpu().data.numpy()
      self.local_actor.train()

      if add_noise:
        actions += self.noise.sample()

      return np.clip(actions, -1, 1)


      