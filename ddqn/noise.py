import random
import numpy as np
import copy

class Ornstein:

    def __init__(self, size, seed:int, mu=.0, theta=0.15, sigma=0.2):
        
        self.seed = random.seed(seed)
        self.size = size
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.state = copy.copy(self.mu)
    
    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x)
        dx += self.sigma * np.random.randn(*self.size)
        self.state = x + dx

        return self.state
    
    def reset_state(self):
        self.state = copy.copy(self.mu)

        