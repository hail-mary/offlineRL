import numpy as np

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

    def sample(self, batch_size, sequence_length):
        """ select the starting point at random, then collect the sequence data from there"""
        starting_idxs = np.random.randint(0, (self.ptr % self.capacity) - sequence_length, (batch_size,))
        idxs = np.stack([np.arange(start, start + sequence_length) for start in starting_idxs])
        # idxs.shape: [batch_size, seq_length]
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
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), # (64, 16, 16)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),# (128, 8, 8)
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),# (256, 4, 4)
            nn.ReLU()
        )
        self.fc = nn.Linear(256 * 4 * 4, feature_dim)  # 256*4*4 → latent_dim

    def forward(self, x):
        x = self.conv_layers(x)  # CNN適用
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)  # 全結合層で次元圧縮
        return x
    
class CNNDecoder(nn.Module):
    """ input:  (batch_size, seq_length, hidden_dim + latent_dim)
        output: (batch_size, seq_length, C, H, W)"""
    
    def __init__(self, hidden_dim=128, latent_dim=32, seq_length=50, output_channels=3):
        super().__init__()
        self.seq_length = seq_length
        self.fc = nn.Linear(hidden_dim + latent_dim, 256 * 4 * 4)  # まずは4×4×256の特徴マップを作る
        
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 8x8
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 32x32
            nn.ReLU(),
            nn.ConvTranspose2d(32, output_channels, kernel_size=4, stride=2, padding=1),  # 64x64
            nn.Sigmoid()  # 出力を [0,1] にする
        )

    def forward(self, x):
        x = self.fc(x)  # → (batch_size, seq_length, 256*4*4)
        x = x.view(-1, 256, 4, 4)  # -> (batch_size * seq_length, 256, 4, 4)
        x = self.deconv_layers(x)  # (batch_size * seq_length, 3, 64, 64)
        x = x.view(-1, self.seq_length, 3, 64, 64)
        return x


class RSSM(nn.Module):
    def __init__(self, action_dim, latent_dim=32, hidden_dim=128, feature_dim=256):
        super().__init__()
        self.latent_dim = latent_dim  # 潜在変数 z の次元
        self.action_dim = action_dim  # 行動 a の次元
        self.hidden_dim = hidden_dim  # 隠れ状態 h の次元

        # RNN (GRU)
        self.rnn = nn.GRUCell(latent_dim + action_dim, hidden_dim)

        # 事前分布 (Prior) のネットワーク
        self.prior_net = nn.Linear(hidden_dim, 2 * latent_dim)  # (μ_prior, logvar_prior)

        self.encoder = CNNEncoder()

        # 事後分布 (Posterior) のネットワーク
        self.posterior_net = nn.Linear(hidden_dim + feature_dim, 2 * latent_dim)  # (μ_post, logvar_post)

    def forward(self, states, actions):
        """
        states: (batch_size, sequence_length, latent_dim)  # 観測データの潜在表現
        actions: (batch_size, sequence_length, action_dim) # 行動
        """
        batch_size, seq_len, *_ = states.shape

        # 初期隠れ状態 h_0 をゼロで初期化
        h_t = torch.zeros(batch_size, self.hidden_dim).to(states.device)

        # 事前分布と事後分布を保存するリスト
        prior_mu_list, prior_logvar_list = [], []
        post_mu_list, post_logvar_list = [], []
        z_list, h_list = [], []

        # 潜在状態をランダム初期化
        z_t = torch.zeros(batch_size, self.latent_dim).to(states.device)

        for t in range(seq_len):
            a_t = actions[:, t]  # actions at timestep t
            x_t = states[:, t]   # image observations at timestep t

            # 1. Update RNN (Sequence): h_t = f(h_{t-1}, z_{t-1}, a{t-1})
            h_t = self.rnn(torch.cat([z_t, a_t], dim=-1), h_t)
            h_list.append(h_t)

            # 2. Prior (Transition): p(z_t | h_t) 
            prior_out = self.prior_net(h_t)
            mu_prior, logvar_prior = torch.chunk(prior_out, 2, dim=-1)  # divide into two
            prior_mu_list.append(mu_prior)
            prior_logvar_list.append(logvar_prior)

            # 3. Encode (extract features from images)
            xt_feature = self.encoder(x_t) # -> [batch_size, feature_dim]

            # 4. Posterior (Representation): q(z_t | x_t, h_t) 
            posterior_out = self.posterior_net(torch.cat([h_t, xt_feature], dim=-1))
            mu_post, logvar_post = torch.chunk(posterior_out, 2, dim=-1)
            post_mu_list.append(mu_post)
            post_logvar_list.append(logvar_post)

            # 5. sample the latent variable z_t
            std_post = torch.exp(0.5 * logvar_post)  
            eps = torch.randn_like(std_post)  
            z_t = mu_post + std_post * eps  # reparametrization trick
            z_list.append(z_t)

        # convert lists to tensors
        prior_mu = torch.stack(prior_mu_list, dim=1)
        prior_logvar = torch.stack(prior_logvar_list, dim=1)
        post_mu = torch.stack(post_mu_list, dim=1)
        post_logvar = torch.stack(post_logvar_list, dim=1)
        z_seq = torch.stack(z_list, dim=1)
        h_seq = torch.stack(h_list, dim=1)

        return prior_mu, prior_logvar, post_mu, post_logvar, h_seq, z_seq
    
class RewardPredictor(nn.Module):
    """ 
    input:  (batch_size, seq_length, hidden_dim + latent_dim)
    output: (batch_size, seq_length, 1)
    """
    def __init__(self, latent_dim=32, hidden_dim=128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim + hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # scalar reward
        )

    def forward(self, x):
        rewards = self.fc(x)  # (batch_size, seq_length, 1)
        return rewards
    
class ContinuePredictor(nn.Module):
    """ 
    input:  (batch_size, seq_length, hidden_dim + latent_dim)
    output: (batch_size, seq_length, 1) - continuation probability
    1 means episode continues, 0 means episode terminates
    """
    def __init__(self, latent_dim=32, hidden_dim=128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim + hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),  # binary prediction
            nn.Sigmoid()  # output in range [0,1]
        )

    def forward(self, x):
        # x shape: (batch_size, seq_length, hidden_dim + latent_dim)
        continue_pred = self.fc(x)  # (batch_size, seq_length, 1)
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
seq_length = 50

# Replay Buffer の作成
buffer = ReplayBuffer(capacity=50000, state_shape=state_shape, action_dim=action_dim)
# state_seq, action_seq, reward_seq, next_state_seq, done_seq = [], [], [], [], []

# 環境からデータ収集
obs, _ = env.reset()
done = False
for t in range(10000):
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


rssm = RSSM(action_dim=action_dim)
decoder = CNNDecoder()
reward_predictor = RewardPredictor()
continue_predictor = ContinuePredictor()
world_model = WorldModel(rssm, decoder, reward_predictor, continue_predictor)
world_optimizer = torch.optim.Adam(world_model.parameters(), lr=3e-4)

for epoch in range(1000):  # Start training
    states, actions, rewards, next_states, dones = buffer.sample(batch_size=32, sequence_length=seq_length)
    
    states = torch.tensor(states, dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.float32)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    next_states = torch.tensor(next_states, dtype=torch.float32)
    dones = torch.tensor(dones, dtype=torch.float32)
    
    # RSSM forward includes RNN, prior, encoder, and posterior predictions 
    prior_mu, prior_logvar, post_mu, post_logvar, h_seq, z_seq = world_model.rssm(states, actions)

    # compute reconstruction loss
    decoder_input = torch.cat([z_seq, h_seq], dim=-1)
    states_pred = world_model.decoder(decoder_input)
    recon_loss = F.mse_loss(states, states_pred, reduction='sum')

    # compute kl-divergence loss between prior and posterior
    post_dist = dist.Normal(post_mu, torch.exp(0.5 * post_logvar))
    post_dist_detached = dist.Normal(post_mu.detach(), torch.exp(0.5 * post_logvar.detach()))
    prior_dist = dist.Normal(prior_mu, torch.exp(0.5 * prior_logvar))
    prior_dist_detached = dist.Normal(prior_mu.detach(), torch.exp(0.5 * prior_logvar.detach()))
    alpha = 0.8
    kl_loss = alpha * dist.kl.kl_divergence(
        post_dist_detached,
        prior_dist
    ).sum() + \
        (1 - alpha) * dist.kl.kl_divergence(
        post_dist,
        prior_dist_detached
    ).sum()

    # compute rewards log loss
    reward_input = torch.cat([z_seq, h_seq], dim=-1)
    rewards_pred = world_model.reward_predictor(reward_input)
    reward_loss = F.mse_loss(rewards, rewards_pred, reduction='sum')

    # compute continue prediction loss using binary cross entropy
    continue_input = torch.cat([z_seq, h_seq], dim=-1)
    continue_pred = world_model.continue_predictor(continue_input)
    # 1 - dones because dones=1 means termination, but we want to predict continuation
    continue_target = 1 - dones  
    continue_loss = F.binary_cross_entropy(continue_pred, continue_target, reduction='sum')

    # add to total loss
    loss = recon_loss + kl_loss + reward_loss + continue_loss

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
    
