import numpy as np
import matplotlib.pyplot as plt
from torchvision.io import write_video
# requires pip install av
import os

class ReplayBuffer:
    def __init__(self, capacity, state_shape, action_dim):
        self.capacity = capacity
        self.states = np.zeros((capacity, *state_shape), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, *state_shape), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def add(self, state, action, reward, next_state, done):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, batch_len):
        """ select the starting point at random, then collect the sequence data from there"""
        starting_idxs = np.random.randint(0, (self.ptr % self.capacity) - batch_len, (batch_size,))
        idxs = np.stack([np.arange(start, start + batch_len) for start in starting_idxs])
        # idxs.shape: [batch_size, batch_len]
        return (
            self.states[idxs], 
            self.actions[idxs], 
            self.rewards[idxs], 
            self.next_states[idxs],
            self.dones[idxs]
        )

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist

class CNNEncoder(nn.Module):
    """ input:  (batch_size, C, H, W)
        output: (batch_size, feature_dim)"""
    
    def __init__(self, feature_dim=256):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),  # (32, 32, 32)
            nn.LayerNorm([32, 32, 32]),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), # (64, 16, 16)
            nn.LayerNorm([64, 16, 16]),
            nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),# (128, 8, 8)
            nn.LayerNorm([128, 8, 8]),
            nn.SiLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),# (256, 4, 4)
            nn.LayerNorm([256, 4, 4]),
            nn.SiLU()
        )
        self.fc = nn.Linear(256 * 4 * 4, feature_dim)  # 256*4*4 → latent_dim

    def forward(self, x):
        x = self.conv_layers(x)  # CNN適用
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)  # 全結合層で次元圧縮
        return x
    
class CNNDecoder(nn.Module):
    """ input:  (batch_size, batch_len, hidden_dim + num_categorical * class_size)
        output: (batch_size, batch_len, C, H, W)"""
    
    def __init__(self, hidden_dim=128, num_categorical=32, class_size=32, batch_len=64, output_channels=3):
        super().__init__()
        self.batch_len = batch_len
        categorical_dim = num_categorical * class_size
        
        # Modified input dimension to match categorical latents
        self.fc = nn.Linear(hidden_dim + categorical_dim, 256 * 4 * 4)
        
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 8x8
            nn.LayerNorm([128, 8, 8]),
            nn.SiLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 16x16
            nn.LayerNorm([64, 16, 16]),
            nn.SiLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 32x32
            nn.LayerNorm([32, 32, 32]),
            nn.SiLU(),
            nn.ConvTranspose2d(32, output_channels, kernel_size=4, stride=2, padding=1),  # 64x64
            nn.Sigmoid()  # output in range [0,1]
        )

    def forward(self, x):
        # x shape: (batch_size, batch_len, hidden_dim + categorical_dim)
        x = self.fc(x)  # → (batch_size, batch_len, 256*4*4)
        x = x.view(-1, 256, 4, 4)  # -> (batch_size * batch_len, 256, 4, 4)
        x = self.deconv_layers(x)  # (batch_size * batch_len, 3, 64, 64)
        x = x.view(-1, self.batch_len, 3, 64, 64)
        return x


class RSSM(nn.Module):
    def __init__(self, action_dim, num_categorical=32, class_size=32, hidden_dim=128, feature_dim=256):
        super().__init__()
        self.num_categorical = num_categorical  # number of categorical variables
        self.class_size = class_size          # number of classes per categorical
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # RNN (GRU)
        categorical_dim = num_categorical * class_size
        self.rnn = nn.GRUCell(categorical_dim + action_dim, hidden_dim)

        # Prior network (outputs logits for each categorical)
        self.prior_net = nn.Linear(hidden_dim, num_categorical * class_size)

        self.encoder = CNNEncoder()

        # Posterior network
        self.posterior_net = nn.Linear(hidden_dim + feature_dim, num_categorical * class_size)

    def forward(self, states, actions):
        batch_size, seq_len, *_ = states.shape
        device = states.device

        # Initialize hidden state and categorical samples
        h_t = torch.zeros(batch_size, self.hidden_dim).to(device)
        z_t = torch.zeros(batch_size, self.num_categorical * self.class_size).to(device)

        prior_logits_list, post_logits_list = [], []
        z_list, h_list = [], []

        for t in range(seq_len):
            a_t = actions[:, t]
            x_t = states[:, t]

            # 1. Update RNN
            h_t = self.rnn(torch.cat([z_t, a_t], dim=-1), h_t)
            h_list.append(h_t)

            # 2. Prior (Transition): p(z_t | h_t)
            prior_logits = self.prior_net(h_t)
            prior_logits = prior_logits.view(batch_size, self.num_categorical, self.class_size)
            prior_logits_list.append(prior_logits)

            # 3. Encode observations
            xt_feature = self.encoder(x_t)# -> [batch_size, feature_dim]

            # 4. Posterior: q(z_t | x_t, h_t)
            posterior_logits = self.posterior_net(torch.cat([h_t, xt_feature], dim=-1))
            posterior_logits = posterior_logits.view(batch_size, self.num_categorical, self.class_size)
            post_logits_list.append(posterior_logits)

            # 5. Sample categorical variables using straight-through estimator
            if self.training:
                # Gumbel-Softmax with straight-through estimator
                z_t = F.gumbel_softmax(posterior_logits, tau=1.0, hard=True)
            else:
                # During evaluation, use argmax
                z_t = F.one_hot(posterior_logits.argmax(dim=-1), num_classes=self.class_size)
            
            # Flatten categorical samples
            z_t = z_t.view(batch_size, -1)
            z_list.append(z_t)

        # Stack tensors
        prior_logits = torch.stack(prior_logits_list, dim=1)
        post_logits = torch.stack(post_logits_list, dim=1)
        z_seq = torch.stack(z_list, dim=1)
        h_seq = torch.stack(h_list, dim=1)

        return prior_logits, post_logits, h_seq, z_seq

    def compute_KLloss(self, prior_logits, post_logits, dyn_scale=0.5, rep_scale=0.1):
        """Compute KL divergence between categorical distributions"""
        prior_probs = F.softmax(prior_logits, dim=-1)
        post_probs = F.softmax(post_logits, dim=-1)
        prior_probs_detached = F.softmax(prior_logits.detach(), dim=-1)
        post_probs_detached = F.softmax(post_logits.detach(), dim=-1)
        
        # KL divergence for categorical distributions
        dyn_kl_loss = (post_probs_detached * (torch.log(post_probs_detached + 1e-8) - 
                                torch.log(prior_probs + 1e-8))).sum(dim=-1)
        rep_kl_loss = (post_probs * (torch.log(post_probs + 1e-8) - 
                                torch.log(prior_probs_detached + 1e-8))).sum(dim=-1)
        
        # Kl balancing
        kl_loss = dyn_scale * max(1, dyn_kl_loss.sum()) + \
                  rep_scale * max(1, rep_kl_loss.sum())

        return kl_loss

class RewardPredictor(nn.Module):
    """ 
    input:  (batch_size, batch_len, hidden_dim + latent_dim)
    output: (batch_size, batch_len, 1)
    """
    def __init__(self, latent_dim=32, hidden_dim=128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim + hidden_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, 1)  # scalar reward
        )

    def forward(self, x):
        rewards = self.fc(x)  # (batch_size, batch_len, 1)
        return rewards
    
class ContinuePredictor(nn.Module):
    """ 
    input:  (batch_size, batch_len, hidden_dim + latent_dim)
    output: (batch_size, batch_len, 1) - continuation probability
    1 means episode continues, 0 means episode terminates
    """
    def __init__(self, latent_dim=32, hidden_dim=128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim + hidden_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, 1),  # binary prediction
            nn.Sigmoid()  # output in range [0,1]
        )

    def forward(self, x):
        # x shape: (batch_size, batch_len, hidden_dim + latent_dim)
        continue_pred = self.fc(x)  # (batch_size, batch_len, 1)
        return continue_pred

class WorldModel(nn.Module):
    def __init__(self, rssm, decoder, reward_predictor, continue_predictor):
        super().__init__()
        self.rssm = rssm
        self.decoder = decoder
        self.reward_predictor = reward_predictor
        self.continue_predictor = continue_predictor


import gymnasium as gym
from gymnasium.wrappers import AddRenderObservation

# MuJoCo 環境のセットアップ
env = gym.make("HalfCheetah-v5", render_mode="rgb_array", height=64, width=64)
env = AddRenderObservation(env, render_only=True)
state_shape = (3, 64, 64)  # 画像 (RGB, 64x64)
action_dim = env.action_space.shape[0]

# Replay Buffer の作成
buffer = ReplayBuffer(capacity=50000, state_shape=state_shape, action_dim=action_dim)
# state_seq, action_seq, reward_seq, next_state_seq, done_seq = [], [], [], [], []

# 環境からデータ収集
obs, _ = env.reset()
done = False
ENV_STEPS = 10000
for t in range(ENV_STEPS):
    action = env.action_space.sample()  # ランダム行動
    next_obs, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    if done:
        obs, _ = env.reset()

    # 画像をリサイズ (64x64) してバッファに保存
    resized_obs = F.interpolate(
        torch.tensor(obs.copy(), dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0,
        size=(64, 64),
        mode="bilinear",
        align_corners=False
    ).squeeze(0).numpy()

    resized_next_obs = F.interpolate(
        torch.tensor(next_obs.copy(), dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0,
        size=(64, 64),
        mode="bilinear",
        align_corners=False
    ).squeeze(0).numpy()

    
    buffer.add(resized_obs, action, reward, resized_next_obs, done)
    obs = next_obs


rssm = RSSM(action_dim=action_dim, num_categorical=32, class_size=32)
decoder = CNNDecoder(num_categorical=32, class_size=32)  # match RSSM parameters
reward_predictor = RewardPredictor(latent_dim=32*32)  # adjust for categorical dim
continue_predictor = ContinuePredictor(latent_dim=32*32)  # adjust for categorical dim
world_model = WorldModel(rssm, decoder, reward_predictor, continue_predictor)
world_optimizer = torch.optim.Adam(world_model.parameters(), lr=3e-4)
torch.nn.utils.clip_grad_norm_(world_model.parameters(), max_norm=1000.0)

def save_reconstruction_video(original_states, reconstructed_states, epoch, save_dir='cheetah_videos'):
    """
    Save original and reconstructed sequences as video files
    Args:
        original_states: tensor of shape (batch_size, batch_len, C, H, W)
        reconstructed_states: tensor of shape (batch_size, batch_len, C, H, W)
        epoch: current epoch number
        save_dir: directory to save videos
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Take first sequence from batch
    original = original_states[0]  # [batch_len, C, H, W]
    reconstructed = reconstructed_states[0]  # [batch_len, C, H, W]
    
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

# Initialize lists to track training losses
recon_losses = []
kl_losses = []
reward_losses = []
continue_losses = []
total_losses = []

N_EPOCHS = 2000
BATCH_SIZE = 16
BATCH_LEN = 64
for epoch in range(1, N_EPOCHS+1):  # Start training
    states, actions, rewards, next_states, dones = buffer.sample(batch_size=BATCH_SIZE, batch_len=BATCH_LEN)
    
    states = torch.tensor(states, dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.float32)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    next_states = torch.tensor(next_states, dtype=torch.float32)
    dones = torch.tensor(dones, dtype=torch.float32)
    
    # RSSM forward pass
    prior_logits, post_logits, h_seq, z_seq = world_model.rssm(states, actions)

    # Compute KL loss for categorical variables
    kl_loss = world_model.rssm.compute_KLloss(prior_logits, post_logits) / (BATCH_SIZE)

    # Rest of the losses remain the same
    decoder_input = torch.cat([z_seq, h_seq], dim=-1)
    states_pred = world_model.decoder(decoder_input)
    recon_loss = F.mse_loss(states, states_pred, reduction='sum') / (BATCH_SIZE)

    # Save videos every 100 epochs
    if epoch % 100 == 0:
        print("\nSaving reconstruction videos...")
        save_reconstruction_video(states, states_pred, epoch)

    # compute rewards log loss
    reward_input = torch.cat([z_seq, h_seq], dim=-1)
    rewards_pred = world_model.reward_predictor(reward_input)
    reward_loss = F.mse_loss(rewards, rewards_pred, reduction='sum') / BATCH_SIZE

    # compute continue prediction loss using binary cross entropy
    continue_input = torch.cat([z_seq, h_seq], dim=-1)
    continue_pred = world_model.continue_predictor(continue_input)
    continue_target = 1 - dones
    continue_loss = F.binary_cross_entropy(continue_pred, continue_target, reduction='sum') / BATCH_SIZE

    # add to total loss
    loss = recon_loss + kl_loss + reward_loss + continue_loss

    # Track losses
    recon_losses.append(recon_loss.item())
    kl_losses.append(kl_loss.item())
    reward_losses.append(reward_loss.item())
    continue_losses.append(continue_loss.item())
    total_losses.append(loss.item())

    # Update world model parameters
    world_optimizer.zero_grad()
    loss.backward()
    world_optimizer.step()

    print(f"\n[Epoch {epoch}] "
          f"Recon Loss: {recon_loss.item():.3f} | "
          f"KL Loss: {kl_loss.item():.3f} | "
          f"Reward Loss: {reward_loss.item():.3f} | "
          f"Continue Loss: {continue_loss.item():.3f} |\n"
          f"Total Loss: {loss.item():.3f}")

    # Plot and save training curves every 100 epochs
    if epoch % 10 == 0:
        print("\nSaving training curves...")
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot reconstruction loss
        ax1.plot(range(1, epoch+1), recon_losses, 'b-', label='Reconstruction Loss')
        ax1.set_title('Reconstruction Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot KL loss
        ax2.plot(range(1, epoch+1), kl_losses, 'r-', label='KL Loss')
        ax2.set_title('KL Divergence Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        # Plot reward loss
        ax3.plot(range(1, epoch+1), reward_losses, 'g-', label='Reward Loss')
        ax3.set_title('Reward Prediction Loss')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Loss')
        ax3.legend()
        ax3.grid(True)
        
        # Plot total loss
        ax4.plot(range(1, epoch+1), total_losses, 'k-', label='Total Loss')
        ax4.set_title('Total Loss')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Loss')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        
        # Create directory if it doesn't exist
        os.makedirs('training_curves', exist_ok=True)
        
        # Save the plot
        plt.savefig(f'training_curves/training_curves_epoch_{epoch}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Also save a combined plot with all losses
        plt.figure(figsize=(12, 8))
        plt.plot(range(1, epoch+1), recon_losses, 'b-', label='Reconstruction Loss', alpha=0.7)
        plt.plot(range(1, epoch+1), kl_losses, 'r-', label='KL Loss', alpha=0.7)
        plt.plot(range(1, epoch+1), reward_losses, 'g-', label='Reward Loss', alpha=0.7)
        plt.plot(range(1, epoch+1), continue_losses, 'm-', label='Continue Loss', alpha=0.7)
        plt.plot(range(1, epoch+1), total_losses, 'k-', label='Total Loss', linewidth=2)
        plt.title('Training Losses Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')  # Use log scale for better visualization
        plt.tight_layout()
        plt.savefig(f'training_curves/combined_losses_epoch_{epoch}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Training curves saved to training_curves/")

