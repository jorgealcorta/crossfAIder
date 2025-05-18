import torch.nn as nn
from torch.nn.utils import spectral_norm

class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.model = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels, 64, 4, stride=2, padding=1)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv2d(64, 128, 4, stride=2, padding=1)),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d((30, 30)),
            spectral_norm(nn.Conv2d(128, 1, 1)),
            # Removed Sigmoid for WGAN-GP
        )

    def forward(self, x):
        return self.model(x)