import torch
import numpy as np

class ReplayBuffer():
    def __init__(self, state_dim, action_dim, max_size=100000):
        self.s = np.zeros((max_size, *state_dim))
        self.a = np.zeros((max_size, *action_dim))
        self.r = np.zeros((max_size, 1))
        self.s_prime = np.zeros((max_size, *state_dim))
        self.terminated = np.zeros((max_size, 1))

        self.ptr = 0
        self.size = 0
        self.max_size = max_size

    def update(self, s, a ,r, s_prime, terminated):
        self.s[self.ptr] = s
        self.a[self.ptr] = a
        self.r[self.ptr] = r
        self.s_prime[self.ptr] = s_prime
        self.terminated[self.ptr] = terminated

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        idx = np.random.choice(self.size, batch_size, replace=False)
        return (
            torch.FloatTensor(self.s[idx]),
            torch.FloatTensor(self.a[idx]),
            torch.FloatTensor(self.r[idx]),
            torch.FloatTensor(self.s_prime[idx]),
            torch.FloatTensor(self.terminated[idx])
        )