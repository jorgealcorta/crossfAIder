from models.generator import UNetGenerator
from models.discriminator import Discriminator
from dataset import MelTransitionDataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

# TODO: import dataset and train GAN
# df = pd.read_csv("transitions_dataset.csv")
dataset = MelTransitionDataset(df)
loader = DataLoader(dataset, batch_size=4, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

G = UNetGenerator().to(device)
D = Discriminator().to(device)

loss_fn = nn.BCELoss()
optimizer_G = optim.Adam(G.parameters(), lr=0.0002)
optimizer_D = optim.Adam(D.parameters(), lr=0.0002)

for epoch in range(50):
    for x, y in loader:
        x, y = x.to(device), y.to(device)

        # Train Discriminator
        D.zero_grad()
        real_input = torch.cat([x, y], dim=1)
        real_label = torch.ones(x.size(0), 1, 30, 30).to(device)
        fake_transition = G(x)
        fake_input = torch.cat([x, fake_transition.detach()], dim=1)
        fake_label = torch.zeros(x.size(0), 1, 30, 30).to(device)

        loss_real = loss_fn(D(real_input), real_label)
        loss_fake = loss_fn(D(fake_input), fake_label)
        loss_D = (loss_real + loss_fake) / 2
        loss_D.backward()
        optimizer_D.step()

        # Train Generator
        G.zero_grad()
        fake_input = torch.cat([x, fake_transition], dim=1)
        loss_G = loss_fn(D(fake_input), real_label)
        loss_G.backward()
        optimizer_G.step()

    print(f"Epoch {epoch}: Loss_D={loss_D.item():.4f}  Loss_G={loss_G.item():.4f}")
