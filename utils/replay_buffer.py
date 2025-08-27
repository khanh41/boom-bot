import random
from collections import deque
import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, capacity: int, obs_shape):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.obs_shape = obs_shape

    def push(self, s, a, r, s2, d, t):
        self.buffer.append((s, a, r, s2, d, t))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s2, d, t = zip(*batch)
        s = torch.as_tensor(s, dtype=torch.float32)
        a = torch.as_tensor(a, dtype=torch.int64).unsqueeze(-1)
        r = torch.as_tensor(r, dtype=torch.float32).unsqueeze(-1)
        s2 = torch.as_tensor(s2, dtype=torch.float32)
        d = torch.as_tensor(d, dtype=torch.float32).unsqueeze(-1)
        t = torch.as_tensor(t, dtype=torch.float32).unsqueeze(-1)
        return s, a, r, s2, d, t

    def __len__(self):
        return len(self.buffer)
