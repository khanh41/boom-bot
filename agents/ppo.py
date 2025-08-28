import torch
import torch.nn as nn
import torch.optim as optim
from utils.nets import ConvNetSmall
from utils.common import device
import numpy as np

class Actor(nn.Module):
    def __init__(self, in_channels, n_actions, H=18, W=28):
        super().__init__()
        self.backbone = ConvNetSmall(in_channels, 256, H=H, W=W)
        # use a separate small head for policy
        self.policy_head = nn.Sequential(nn.ReLU(), nn.Linear(256, n_actions))

    def forward(self, x):
        z = self.backbone.features(x)
        # re-use head's first linear to compress to 256 if needed
        z = self.backbone.head[0](z)  # Linear(flat,256)
        z = torch.relu(z)
        logits = self.policy_head[1](z) if False else self.policy_head(z)
        return logits

class Critic(nn.Module):
    def __init__(self, in_channels, H=18, W=28):
        super().__init__()
        self.backbone = ConvNetSmall(in_channels, 256, H=H, W=W)
        self.v = nn.Sequential(nn.ReLU(), nn.Linear(256, 1))

    def forward(self, x):
        z = self.backbone.features(x)
        z = self.backbone.head[0](z)
        z = torch.relu(z)
        v = self.v[1](z) if False else self.v(z).squeeze(-1)
        return v

class PPOAgent:
    def __init__(self, in_channels, n_actions, H=18, W=28, gamma=0.99, lam=0.95, clip=0.2, vf_coef=0.5, ent_coef=0.01, lr=3e-4, batch_size=4096, update_epochs=4):
        self.actor = Actor(in_channels, n_actions, H=H, W=W).to(device())
        self.critic = Critic(in_channels, H=H, W=W).to(device())
        self.opt = optim.AdamW(list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr)
        self.gamma, self.lam = gamma, lam
        self.clip, self.vf_coef, self.ent_coef = clip, vf_coef, ent_coef
        self.batch_size = batch_size
        self.update_epochs = update_epochs

    def get_action(self, obs):
        with torch.no_grad():
            arr = np.array(obs, dtype=np.float32)
            x = torch.as_tensor(arr, dtype=torch.float32, device=device()).unsqueeze(0)
            logits = self.actor(x)
            dist = torch.distributions.Categorical(logits=logits)
            a = dist.sample()
            logp = dist.log_prob(a)
            v = self.critic(x)
        return int(a.item()), float(logp.item()), float(v.item())

    def compute_gae(self, rewards, values, dones, gamma, lam):
        adv = []
        gae = 0.0
        next_value = 0.0
        for r, v, d in zip(reversed(rewards), reversed(values), reversed(dones)):
            delta = r + gamma * (0 if d else next_value) - v
            gae = delta + gamma * lam * (0 if d else gae)
            adv.append(gae)
            next_value = v
        adv = list(reversed(adv))
        returns = [a + v for a, v in zip(adv, values)]
        return adv, returns

    def update(self, batch):
        obs = torch.as_tensor(batch['obs'], dtype=torch.float32, device=device())
        act = torch.as_tensor(batch['act'], dtype=torch.int64, device=device())
        old_logp = torch.as_tensor(batch['logp'], dtype=torch.float32, device=device())
        adv = torch.as_tensor(batch['adv'], dtype=torch.float32, device=device())
        ret = torch.as_tensor(batch['ret'], dtype=torch.float32, device=device())

        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        for _ in range(self.update_epochs):
            logits = self.actor(obs)
            dist = torch.distributions.Categorical(logits=logits)
            logp = dist.log_prob(act)
            ratio = torch.exp(logp - old_logp)
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1.0 - self.clip, 1.0 + self.clip) * adv
            policy_loss = -torch.min(surr1, surr2).mean()

            v = self.critic(obs).squeeze(-1)
            v_loss = 0.5 * (ret - v).pow(2).mean()

            ent = dist.entropy().mean()
            loss = policy_loss + self.vf_coef * v_loss - self.ent_coef * ent

            self.opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.critic.parameters()), 0.5)
            self.opt.step()

        return {
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(v_loss.item()),
            "entropy": float(ent.item())
        }

    def save(self, path_actor: str, path_critic: str):
        import os
        os.makedirs(os.path.dirname(path_actor), exist_ok=True)
        os.makedirs(os.path.dirname(path_critic), exist_ok=True)
        torch.save(self.actor.state_dict(), path_actor)
        torch.save(self.critic.state_dict(), path_critic)

    def load(self, path_actor: str, path_critic: str):
        self.actor.load_state_dict(torch.load(path_actor, map_location=device()))
        self.critic.load_state_dict(torch.load(path_critic, map_location=device()))
