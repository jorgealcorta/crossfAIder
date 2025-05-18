from models.generator import UNetGenerator
from models.discriminator import Discriminator
from models.dataset import MelTransitionDataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import os

# --- Configuration ---
ROOT_DIR = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
RES_DIR = os.path.join(ROOT_DIR, "res")
MODELS_DIR = os.path.join(ROOT_DIR, "outputs", "models")

EXPERIMENT_NAME = "improvedGAN_WGAN-GP"
N_EPOCH = 50
os.makedirs(os.path.join(MODELS_DIR, EXPERIMENT_NAME), exist_ok=True)

# --- WGAN-GP Parameters ---
LAMBDA_GP = 10
N_CRITIC = 5  # Train critic 5x more than generator

# --- Dataset ---
dataset = MelTransitionDataset(
    dataset_path=os.path.join(RES_DIR, "datasets", "transition_dataset_processed_III.csv"), 
    config_path=os.path.join(RES_DIR, "config", "config.json"),
    spectrogram_path=os.path.join(RES_DIR, "mel_specs")
)
loader = DataLoader(dataset, batch_size=4, shuffle=True, pin_memory=True)

# --- Models ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
G = UNetGenerator().to(device)
D = Discriminator().to(device)

# --- Optimizers (TTUR) ---
optimizer_G = optim.Adam(G.parameters(), lr=0.0001, betas=(0.5, 0.999))
optimizer_D = optim.Adam(D.parameters(), lr=0.0004, betas=(0.5, 0.999))

# --- Losses ---
loss_l1 = nn.L1Loss()

def gradient_penalty(D, real, fake, device):
    batch_size = real.shape[0]
    epsilon = torch.rand(batch_size, 1, 1, 1).to(device)
    
    # Create interpolated inputs WITH GRADIENTS
    interpolated = epsilon * real + (1 - epsilon) * fake
    interpolated.requires_grad_(True)  # Force gradients for this tensor
    
    # Calculate critic scores
    mixed_scores = D(interpolated)
    
    # Calculate gradients
    gradient = torch.autograd.grad(
        inputs=interpolated,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    
    gradient = gradient.view(gradient.shape[0], -1)
    return torch.mean((gradient.norm(2, dim=1) - 1) ** 2)

# --- Training Loop ---
for epoch in range(1, N_EPOCH + 1):
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        
        # ======== Train Critic ========
        for _ in range(N_CRITIC):
            D.zero_grad()
            
            # Real data (no detach)
            real_input = torch.cat([x, y], dim=1)
            real_score = D(real_input)
            
            # Fake data: generate WITHOUT no_grad()
            fake_transition = G(x)  # Keep in computation graph
            fake_input = torch.cat([x, fake_transition], dim=1)
            fake_score = D(fake_input.detach())  # Detach here to avoid generator update
            
            # Gradient penalty uses non-detached fake_input
            gp = gradient_penalty(D, real_input, fake_input, device)
            
            # Loss and backward
            loss_D = -torch.mean(real_score) + torch.mean(fake_score) + LAMBDA_GP * gp
            loss_D.backward()
            optimizer_D.step()
        
        # ======== Train Generator ========
        G.zero_grad()
        fake_transition = G(x)
        fake_input = torch.cat([x, fake_transition], dim=1)
        gen_score = D(fake_input)
        
        loss_G = -torch.mean(gen_score) + 100 * loss_l1(fake_transition, y)
        loss_G.backward()
        optimizer_G.step()
    
    # --- Progress & Saving ---
    print(f"Epoch {epoch} | D: {loss_D.item():.4f} | G: {loss_G.item():.4f}")
    if epoch % 10 == 0:
        torch.save(G.state_dict(), f"{MODELS_DIR}/{EXPERIMENT_NAME}/G_epoch_{epoch}.pth")
        torch.save(D.state_dict(), f"{MODELS_DIR}/{EXPERIMENT_NAME}/D_epoch_{epoch}.pth")