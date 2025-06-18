import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import v2
from torchvision.utils import make_grid
from torch.utils.data import TensorDataset, DataLoader
from torchvision.io import write_video
import matplotlib.pyplot as plt
import torch.distributions as dist
import random  # Add this import at the top

# Hyperparameters
BATCH_SIZE = 30
SEQ_LEN = 236  # Length of image sequences
LATENT_DIM = 32  # Latent dimension for VAE
HIDDEN_DIM = 128  # Hidden dimension for GRU
IMAGE_SIZE = 128  
NUM_EPOCHS = 1000
LEARNING_RATE = 1e-3

local_datapath = 'expert_demos/episode-0_128x128.pt'
obs_traj = torch.load(local_datapath)
obs_traj = obs_traj.permute(0, 3, 1, 2) # Change (N, H, W, C) to (N, C, H, W)
transforms = v2.ToDtype(torch.float32, scale=True) #  map the [0, 255] range into [0, 1]
obs_traj = transforms(obs_traj)
train_set = TensorDataset(obs_traj)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=False)

import gymnasium as gym
import gymnasium_robotics # v1.3.0
from gymnasium.wrappers import AddRenderObservation
gym.register_envs(gymnasium_robotics)
env = gym.make('FrankaKitchen-v1', render_mode='rgb_array', width=128, height=128)
env = AddRenderObservation(env, render_only=True)


class Encoder(nn.Module):
    def __init__(self, hidden_dim, latent_dim, image_feature_dim=256):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),  # 128x128 -> 64x64
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=4, stride=2, padding=1),  # 64x64 -> 32x32
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 32x32 -> 16x16
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 256),
            nn.ReLU()
        )
        self.fc_h = nn.Linear(hidden_dim, image_feature_dim)
        self.fc_z = nn.Linear(image_feature_dim * 2, latent_dim * 2)

    def forward(self, xt, ht):
        phi_x = self.cnn(xt.unsqueeze(0))
        phi_h = self.fc_h(ht)
        phi = torch.cat([phi_x, phi_h], dim=-1)
        z_params = self.fc_z(phi)
        mu, logvar = z_params.chunk(2, dim=-1)
        return mu, logvar
    
class Decoder(nn.Module):
    def __init__(self, hidden_dim, latent_dim):
        super().__init__()
        self.decoder_fc = nn.Sequential(
            nn.Linear(hidden_dim + latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64 * 16 * 16),
            nn.ReLU()
        )
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 16x16 -> 32x32
            nn.ReLU(),
            nn.ConvTranspose2d(32, 64, kernel_size=4, stride=2, padding=1),  # 32x32 -> 64x64
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),  # 64x64 -> 128x128
            nn.Sigmoid()
        )

    def forward(self, zt, ht):
        decoder_input = torch.cat([zt, ht], dim=-1)
        h = self.decoder_fc(decoder_input)
        h = h.view(-1, 64, 16, 16)
        return self.decoder_conv(h)
    
class SequentialVAE(nn.Module):
    def __init__(self, latent_dim, hidden_dim):
        super().__init__()
        # self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        # RNN (GRU) for temporal dependencies
        self.rnn = nn.GRU(latent_dim, hidden_size=hidden_dim, batch_first=True)

        # Encoder q(z_t | x_t, h_t)
        self.encoder = Encoder(hidden_dim, latent_dim)  

        # Prior p(z_t | z_{t-1}, h_t)
        self.prior = nn.Linear(hidden_dim + latent_dim, 2 * latent_dim)

        # Decoder p(x_t | z_t, h_t)
        self.decoder = Decoder(hidden_dim, latent_dim)

    def reparameterize(self, mu, logvar):
        """Reparameterization Trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        batch_size, seq_len = x.shape[0], x.shape[0]
        h_t = torch.zeros(1, self.hidden_dim).to(x.device)
        z_t = torch.zeros(1, self.latent_dim).to(x.device)

        recon_loss = 0
        kl_loss = 0

        for t in range(seq_len):
            # x_t = x[:, t, :]
            x_t = x[t, :, :, :]

            # --- Prior Network p(z_t | z_{t-1}, h_t) ---
            prior_input = torch.cat([z_t, h_t], dim=-1)
            mu_p, logvar_p = self.prior(prior_input).chunk(2, dim=-1)
            prior_dist = dist.Normal(mu_p, torch.exp(0.5 * logvar_p))

            # --- Encoder q(z_t | x_t, h_t) ---
            mu_q, logvar_q = self.encoder(x_t, h_t)
            posterior_dist = dist.Normal(mu_q, torch.exp(0.5 * logvar_q))

            # --- サンプリング ---
            z_t = self.reparameterize(mu_q, logvar_q)

            # --- RNN 更新 ---
            # rnn_input = torch.cat([x_t, z_t], dim=-1).unsqueeze(1)
            rnn_input = z_t
            h_t, _ = self.rnn(rnn_input, h_t)

            # --- デコーダ p(x_t | z_t, h_t) ---
            x_hat = self.decoder(z_t, h_t)

            # --- 損失計算 ---
            recon_loss += nn.MSELoss()(x_hat, x_t)  # 再構成誤差
            kl_loss += torch.distributions.kl_divergence(posterior_dist, prior_dist).mean()  # KL ダイバージェンス

        return recon_loss + kl_loss, recon_loss, kl_loss
    
def train(model, data_loader, optimizer, device, epochs=50):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for x in data_loader:
            x = x[0].to(device)

            optimizer.zero_grad()
            loss, rec_loss, kl_loss = model(x)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

model = SequentialVAE(latent_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

train(model, train_loader, optimizer, device)

import matplotlib.pyplot as plt

def generate_samples(model, seq_len, device, latent_dim):
    """学習済みモデルを使って時系列データを生成"""
    model.eval()
    with torch.no_grad():
        batch_size = 1  # 1つの系列を生成
        h_t = torch.zeros(1, model.hidden_dim).to(device)
        z_t = torch.zeros(batch_size, latent_dim).to(device)

        generated_sequence = []

        for t in range(seq_len):
            prior_input = torch.cat([z_t, h_t], dim=-1)
            mu_p, logvar_p = model.prior(prior_input).chunk(2, dim=-1)
            prior_dist = torch.distributions.Normal(mu_p, torch.exp(0.5 * logvar_p))

            # 潜在変数をサンプリング
            z_t = prior_dist.rsample()

            # RNN 更新
            rnn_input = z_t
            h_t, _ = model.rnn(rnn_input, h_t)

            # Decoder に通してデータを生成
            x_hat = model.decoder(z_t, h_t)
            generated_sequence.append(x_hat.cpu())

    generated_sequence = torch.stack(generated_sequence, dim=1).squeeze(0).cpu()

    return generated_sequence


# 学習済みモデルを使ってデータを生成
generated_data = generate_samples(model, seq_len=50, device=device, latent_dim=model.latent_dim)

from torchvision.io import write_video
video_array = generated_data.permute(0, 2, 3, 1)
video_array = (video_array * 255).to(torch.uint8)
write_video(filename='seqVAE.mp4', video_array=video_array, fps=12)
