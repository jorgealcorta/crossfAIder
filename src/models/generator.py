import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            spectral_norm(nn.Conv2d(channels, channels, 3, padding=1)),
            nn.InstanceNorm2d(channels),
            nn.ReLU(),
            spectral_norm(nn.Conv2d(channels, channels, 3, padding=1)),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, x):
        return x + self.block(x)

class UNetGenerator(nn.Module):
    def __init__(self, in_channels=2, out_channels=1):
        super().__init__()
        
        # Encoder
        self.enc1 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels, 64, 4, stride=2, padding=1)),
            ResidualBlock(64),
            nn.LeakyReLU(0.2)
        )
        
        self.enc2 = nn.Sequential(
            spectral_norm(nn.Conv2d(64, 128, 4, stride=2, padding=1)),
            ResidualBlock(128),
            nn.LeakyReLU(0.2)
        )

        # Decoder
        self.dec1 = nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)),
            ResidualBlock(64),
            nn.ReLU()
        )
        
        self.dec2 = nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(128, out_channels, 4, stride=2, padding=1)),
            nn.Tanh()
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)  # 64x64
        e2 = self.enc2(e1)  # 32x32
        
        # Decoder with skip
        d1 = self.dec1(e2)
        d1 = torch.cat([d1, e1], dim=1)
        return self.dec2(d1)