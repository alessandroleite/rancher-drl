import random
from collections import deque, namedtuple

import numpy as np
import torch

class ReplayBuffer:

    def __init__(self, buffer_size:int, seed:int, device="cpu"):
        self.seed = random.seed(seed)
        self.max_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)
        self.length = 0
        self.device = device
    
    def add(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)

        if self.length > self.max_size:
            self.buffer.popleft()
        else:
            self.length = self.length + 1
        
        self.buffer.append(experience)

    def sample(self, batch_size):
        experiences = np.random.choice(self.buffer, k=self.length if self.length < batch_size else batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)

        return (states, actions, rewards, next_states, dones)
    
    def __len__(self):
        return self.length