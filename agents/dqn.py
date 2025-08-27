import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple
from utils.nets import ConvNetSmall
from utils.replay_buffer import ReplayBuffer
from utils.common import device

class DQNAgent:
    def __init__(self, in_channels: int, n_actions: int, lr=1e-3, gamma=0.99, eps_start=1.0, eps_end=0.05, eps_decay=20000, buffer_size=200000, batch_size=64, target_sync=2000):
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_sync = target_sync
        self.step_count = 0
        self.eps_start, self.eps_end, self.eps_decay = eps_start, eps_end, eps_decay

        self.q = ConvNetSmall(in_channels, n_actions).to(device())
        self.q_target = ConvNetSmall(in_channels, n_actions).to(device())
        self.q_target.load_state_dict(self.q.state_dict())
        self.optim = optim.AdamW(self.q.parameters(), lr=lr)
        self.replay = ReplayBuffer(buffer_size, obs_shape=None)
        self.loss_fn = nn.SmoothL1Loss()

    def epsilon(self):
        # exponential decay
        return self.eps_end + (self.eps_start - self.eps_end) * np.exp(-1.0 * self.step_count / self.eps_decay)

    def act(self, obs: np.ndarray, greedy: bool=False) -> int:
        self.step_count += 1
        if (not greedy) and np.random.random() < self.epsilon():
            return np.random.randint(0, self.n_actions)
        with torch.no_grad():
            x = torch.as_tensor(obs, dtype=torch.float32, device=device()).unsqueeze(0)
            q = self.q(x)
            return int(torch.argmax(q, dim=1).item())

    def push(self, s,a,r,s2,done,trunc):
        self.replay.push(s,a,r,s2, float(done), float(trunc))

    def learn(self):
        if len(self.replay) < self.batch_size:
            return None
        s, a, r, s2, d, t = self.replay.sample(self.batch_size)
        s, s2 = s.to(device()), s2.to(device())
        a, r, d = a.to(device()), r.to(device()), d.to(device())

        with torch.no_grad():
            q_next = self.q_target(s2).max(dim=1, keepdim=True).values
            y = r + (1.0 - d) * 0.99 * q_next

        q = self.q(s).gather(1, a)
        loss = self.loss_fn(q, y)

        self.optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q.parameters(), 5.0)
        self.optim.step()

        if self.step_count % self.target_sync == 0:
            self.q_target.load_state_dict(self.q.state_dict())
        return float(loss.item())

    def save(self, path: str):
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.q.state_dict(), path)

    def load(self, path: str):
        state = torch.load(path, map_location=device())
        self.q.load_state_dict(state)
        self.q_target.load_state_dict(self.q.state_dict())
