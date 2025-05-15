import torch
import torch.nn as nn

class UNetGenerator(nn.Module):
    """
    U-Net style pix2pix generator
    """
    def __init__(self, in_channels=2, out_channels=1):  # in channels: track A + track B; out channels: transition
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, out_channels, 4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

class UNetGenerator(nn.Module):
    def __init__(self, in_channels=2, out_channels=1):
        super().__init__()
        
        # Encoder
        self.enc1 = nn.Sequential(nn.Conv2d(in_channels, 64, 4, stride=2, padding=1), nn.LeakyReLU(0.2))
        self.enc2 = nn.Sequential(nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2))
        
        # Decoder with skip connections
        self.dec1 = nn.Sequential(nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU())
        self.dec2 = nn.Sequential(nn.ConvTranspose2d(128, out_channels, 4, stride=2, padding=1), nn.Tanh())  # 64*2 from skip

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)  # 64x64
        e2 = self.enc2(e1)  # 32x32
        
        # Decoder with skips
        d1 = self.dec1(e2)  # 64x64
        d1 = torch.cat([d1, e1], dim=1)  # Skip connection
        return self.dec2(d1)  # 128x128