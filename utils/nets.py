import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )
    def forward(self, x):
        return self.net(x)

class ConvNetSmall(nn.Module):
    def __init__(self, in_channels: int, num_actions: int, H: int = 18, W: int = 28):
        super().__init__()
        # small conv trunk
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Flatten()
        )
        # compute flattened size dynamically by passing a dummy tensor in constructor
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, H, W)
            flat = self.features(dummy).shape[1]
        self.head = nn.Sequential(
            nn.Linear(flat, 256), nn.ReLU(),
            nn.Linear(256, num_actions)
        )

    def forward(self, x):
        # x: [B, C, H, W]
        z = self.features(x)
        return self.head(z)
