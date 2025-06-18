import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import v2
from torch.utils.data import TensorDataset, DataLoader
from torchvision.io import write_video
import matplotlib.pyplot as plt
import torch.distributions as dist
import torch.nn.functional as F


# Hyperparameters
BATCH_SIZE = 16
SEQ_LEN = 64  # Length of image sequences
LATENT_DIM = 32  # Latent dimension for VAE
HIDDEN_DIM = 128  # Hidden dimension for GRU
IMAGE_SIZE = 128  
NUM_EPOCHS = 1000
LEARNING_RATE = 1e-3

# Model components
class CNNEncoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),  # 64x64
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),  # 32x32
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),  # 16x16
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),  # 8x8
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, latent_dim * 2)  # Output mean and logvar
        )

    def forward(self, x):
        x = self.encoder(x)
        mu, logvar = torch.chunk(x, 2, dim=-1)
        return mu, logvar

class CNNDecoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256 * 8 * 8),
            nn.Unflatten(1, (256, 8, 8)),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, z):
        return self.decoder(z)

class RSSM(nn.Module):
    def __init__(self, latent_dim, hidden_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        
        # RNN
        self.rnn = nn.GRUCell(latent_dim, hidden_dim)
        
        # Prior network
        self.prior_net = nn.Linear(hidden_dim, latent_dim * 2)
        
        # Posterior network
        self.posterior_net = nn.Linear(hidden_dim + latent_dim, latent_dim * 2)
        
        # Initial states
        self.h0 = nn.Parameter(torch.zeros(1, hidden_dim))
        self.z0 = nn.Parameter(torch.zeros(1, latent_dim))

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, encoder):
        batch_size, seq_len, *_ = x.shape
        device = x.device
        
        # Initialize states
        h_t = self.h0.expand(batch_size, -1).to(device)
        z_t = self.z0.expand(batch_size, -1).to(device)
        
        # Lists to store outputs
        prior_mu_list, prior_logvar_list = [], []
        post_mu_list, post_logvar_list = [], []
        z_list, h_list = [], []
        
        for t in range(seq_len):
            # Update RNN
            h_t = self.rnn(z_t, h_t)
            h_list.append(h_t)
            
            # Prior
            prior_out = self.prior_net(h_t)
            prior_mu, prior_logvar = torch.chunk(prior_out, 2, dim=-1)
            prior_mu_list.append(prior_mu)
            prior_logvar_list.append(prior_logvar)
            
            # Encode observation
            x_t = x[:, t]
            enc_mu, enc_logvar = encoder(x_t)
            
            # Posterior
            post_in = torch.cat([h_t, enc_mu], dim=-1)
            post_out = self.posterior_net(post_in)
            post_mu, post_logvar = torch.chunk(post_out, 2, dim=-1)
            post_mu_list.append(post_mu)
            post_logvar_list.append(post_logvar)
            
            # Sample latent
            z_t = self.reparameterize(post_mu, post_logvar)
            z_list.append(z_t)
        
        # Stack all outputs
        prior_mu = torch.stack(prior_mu_list, dim=1)
        prior_logvar = torch.stack(prior_logvar_list, dim=1)
        post_mu = torch.stack(post_mu_list, dim=1)
        post_logvar = torch.stack(post_logvar_list, dim=1)
        z_seq = torch.stack(z_list, dim=1)
        h_seq = torch.stack(h_list, dim=1)
        
        return prior_mu, prior_logvar, post_mu, post_logvar, z_seq, h_seq

    def compute_kl_loss(self, prior_mu, prior_logvar, post_mu, post_logvar):
        """Compute KL divergence between prior and posterior"""
        kl_div = -0.5 * torch.sum(1 + post_logvar - prior_logvar - 
                                 (post_mu - prior_mu).pow(2) / prior_logvar.exp() - 
                                 post_logvar.exp() / prior_logvar.exp(), dim=-1)
        return kl_div.mean()

local_datapath = 'expert_demos/episode-0_128x128.pt'
obs_traj = torch.load(local_datapath)
obs_traj = obs_traj.permute(0, 3, 1, 2) # Change (N, H, W, C) to (N, C, H, W)
transforms = v2.ToDtype(torch.float32, scale=True) #  map the [0, 255] range into [0, 1]
obs_traj = transforms(obs_traj)

# Create sequences for training
def create_sequences(data, seq_length):
    """Create overlapping sequences from the trajectory"""
    sequences = []
    for i in range(len(data) - seq_length + 1):
        sequences.append(data[i:i+seq_length])
    return torch.stack(sequences)

# Create sequences
sequences = create_sequences(obs_traj, SEQ_LEN)

# Create dataset and dataloader
train_dataset = TensorDataset(sequences)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

print(f"Dataset size: {len(train_dataset)} sequences")
print(f"Sequence shape: {sequences[0].shape}")

# Initialize models
encoder = CNNEncoder(LATENT_DIM)
decoder = CNNDecoder(LATENT_DIM)
rssm = RSSM(LATENT_DIM, HIDDEN_DIM)

# Optimizer
optimizer = optim.Adam(list(encoder.parameters()) + 
                      list(decoder.parameters()) + 
                      list(rssm.parameters()), 
                      lr=LEARNING_RATE)

def save_reconstruction_video(original, reconstructed, epoch, save_dir='videos'):
    """Save original and reconstructed sequences as video files"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Convert to uint8 and scale to [0, 255]
    original = (original * 255).clamp(0, 255).to(torch.uint8)
    reconstructed = (reconstructed * 255).clamp(0, 255).to(torch.uint8)
    
    # Save videos
    original_path = os.path.join(save_dir, f'original_epoch_{epoch}.mp4')
    reconstructed_path = os.path.join(save_dir, f'reconstructed_epoch_{epoch}.mp4')
    
    # Write videos (30 fps)
    write_video(original_path, original.permute(0, 2, 3, 1), fps=30)
    write_video(reconstructed_path, reconstructed.permute(0, 2, 3, 1), fps=30)
    
    print(f"Saved videos to {save_dir}")

# Training loop
for epoch in range(NUM_EPOCHS):
    total_recon_loss = 0
    total_kl_loss = 0
    
    for batch_idx, (seq,) in enumerate(train_loader):
        # Forward pass
        prior_mu, prior_logvar, post_mu, post_logvar, z_seq, h_seq = rssm(seq, encoder)
        
        # Decode latent sequence
        recon_seq = decoder(z_seq.view(-1, LATENT_DIM))
        recon_seq = recon_seq.view(seq.shape)
        
        # Compute losses
        recon_loss = F.mse_loss(recon_seq, seq, reduction='sum')
        kl_loss = rssm.compute_kl_loss(prior_mu, prior_logvar, post_mu, post_logvar)
        
        # Total loss
        loss = recon_loss + kl_loss
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_recon_loss += recon_loss.item()
        total_kl_loss += kl_loss.item()
        
        if batch_idx % 10 == 0:
            print(f"Epoch: {epoch}, Batch: {batch_idx}, "
                  f"Recon Loss: {recon_loss.item():.3f}, "
                  f"KL Loss: {kl_loss.item():.3f}")
    
    # Save videos every 100 epochs
    if epoch % 100 == 0 or epoch == NUM_EPOCHS - 1:
        print("\nSaving reconstruction videos...")
        # Take first sequence from last batch
        save_reconstruction_video(seq[0], recon_seq[0], epoch)
    
    # Print epoch statistics
    avg_recon_loss = total_recon_loss / len(train_loader)
    avg_kl_loss = total_kl_loss / len(train_loader)
    print(f"\nEpoch {epoch} Summary:")
    print(f"Average Reconstruction Loss: {avg_recon_loss:.3f}")
    print(f"Average KL Loss: {avg_kl_loss:.3f}")
    print(f"Total Loss: {avg_recon_loss + avg_kl_loss:.3f}")
