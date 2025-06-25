import gymnasium as gym
from gymnasium.wrappers import AddRenderObservation, RecordVideo
import numpy as np
import cv2
import os
import multiprocessing
from collections import deque
import random
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

class PreprocessObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, width=64, height=64, grayscale=False):
        super().__init__(env)
        self.width = width
        self.height = height
        self.grayscale = grayscale
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0,
            shape=(height, width, 1 if grayscale else 3),
            dtype=np.float32
        )

    def observation(self, obs):
        # Assume obs is an image (H, W, C) in uint8 [0,255]
        img = cv2.resize(obs, (self.width, self.height))
        if self.grayscale:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = np.expand_dims(img, axis=-1)
        img = torch.from_numpy(img).permute(2, 0, 1) # (H, W, C) -> (C, H, W)
        img = img.float() / 255.0  # Scale to [0, 1]
        return img

class CNNEncoder(nn.Module):
    def __init__(self, in_channels=3, width=64, height=64, features_dim=512):
        super().__init__()
        self.features_dim = features_dim
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )
        self.fc = nn.Linear(256 * 4 * 4, features_dim)

    def forward(self, x):
        x = self.conv(x)
        x = x.reshape(x.size(0), -1) # (batch_size, 256 * 4 * 4)
        x = self.fc(x)
        return x
        
class Actor(nn.Module):
    def __init__(self, feature_dim, action_dim, log_std_min=-20, log_std_max=2):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(256, action_dim)
        self.log_std_head = nn.Linear(256, action_dim)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
    def forward(self, x):
        x = self.fc(x)
        mu = self.mu_head(x)
        log_std = self.log_std_head(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        return mu, std
    
    def sample_action(self, features: torch.Tensor, deterministic: bool = False):
        mu, std = self.forward(features)
        dist = torch.distributions.Normal(mu, std)
        if deterministic:
            action = mu
        else:
            action = dist.rsample() # (batch_size, action_dim)
        action = torch.tanh(action)
        return action
    
class Critic(nn.Module):
    def __init__(self, feature_dim, action_dim):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(feature_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(feature_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
        
    def forward(self, features, action):
        x = torch.cat([features, action], dim=1)
        q1 = self.q1(x)
        q2 = self.q2(x)
        return q1, q2

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.stack(states), # (batch_size, C, H, W)
            np.stack(actions), # (batch_size, action_dim)
            np.stack(rewards), # (batch_size, 1)
            np.stack(next_states), # (batch_size, C, H, W)
            np.stack(dones), # (batch_size, 1)
        )
    
    def __len__(self):
        return len(self.buffer)

class SAC:
    def __init__(self, env, feature_dim=512, action_dim=2, hidden_dim=256, log_std_min=-20, log_std_max=2, gamma=0.99, tau=0.005, alpha=0.2, warmup_steps=1000, device="cpu"):
        self.env = env
        self.feature_dim = feature_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.device = device
        self.warmup_steps = warmup_steps
        self.encoder = CNNEncoder(in_channels=3, width=64, height=64, features_dim=feature_dim).to(device)
        self.actor = Actor(feature_dim, action_dim, log_std_min, log_std_max).to(device)
        self.critic = Critic(feature_dim, action_dim).to(device)
        self.critic_target = Critic(feature_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
        self.replay_buffer = ReplayBuffer(capacity=1_000_000)
        self.total_rewards = []

    def update_parameters(self, batch_size=256):
        # sample from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device).unsqueeze(1)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device).unsqueeze(1)

        z = self.encoder(states)
        z_next = self.encoder(next_states).detach()

        # target value
        with torch.no_grad():
            next_actions = self.actor.sample_action(z_next, deterministic=True)
            next_q1, next_q2 = self.critic_target(z_next, next_actions)
            next_q = torch.min(next_q1, next_q2)
            target_q = rewards + (1 - dones) * self.gamma * next_q

        # update critic
        q1, q2 = self.critic(z, actions.reshape(batch_size, -1))
        q1_loss = F.mse_loss(q1, target_q)
        q2_loss = F.mse_loss(q2, target_q)
        critic_loss = q1_loss + q2_loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        mu, std = self.actor(z.detach())
        action_dist = torch.distributions.Normal(mu, std)
        log_prob = action_dist.log_prob(actions).sum(dim=-1, keepdim=True)
        q1_pi, q2_pi = self.critic(z.detach(), action_dist.rsample())
        q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = -(self.alpha * log_prob + q_pi).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # update target networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        return critic_loss.item(), actor_loss.item()
        
    def learn(self, total_timesteps=100_000, eval_interval=1000):
        self.total_steps = np.arange(0, total_timesteps + 1, eval_interval)
        step = 0
        state, info = self.env.reset()
        while step < total_timesteps:
            if step < self.warmup_steps:
                action = self.env.action_space.sample()
            else:
                with torch.no_grad():
                    z = self.encoder(state.unsqueeze(0))
                    action = self.actor.sample_action(z)
                    action = action.squeeze(0).cpu().numpy()

            next_state, reward, done, truncated, info = self.env.step(action)
            self.replay_buffer.add(state, action, reward, next_state, done)
            state = next_state
            if done or truncated:
                state, info = self.env.reset()

            if len(self.replay_buffer) > self.warmup_steps:
                critic_loss, actor_loss = self.update_parameters(batch_size=256)
                if step % 100 == 0:
                    print(f"Step {step}: Critic Loss: {critic_loss:.4f}, Actor Loss: {actor_loss:.4f}")
            if step % eval_interval == 0:
                total_reward = self.eval_policy(record=False)
                print(f"Step {step}: Env: {self.env.unwrapped.spec.id}, Total Reward: {total_reward:.4f}")
            step += 1
                    
    def eval_policy(self, record=False):
        if record:
            self.env = RecordVideo(self.env,
                                   video_folder=f"videos/{self.env.unwrapped.spec.id}",
                                   episode_trigger=lambda x: True,
                                   video_length=100,
                                   name_prefix=f"eval_{self.env.unwrapped.spec.id}")
        state, info = self.env.reset()
        total_reward = 0
        done = False
        truncated = False
        while not done and not truncated:
            with torch.no_grad():
                z = self.encoder(state.unsqueeze(0))
                action = self.actor.sample_action(z, deterministic=True)
                next_state, reward, done, truncated, info = self.env.step(action.squeeze(0).cpu().numpy())
                total_reward += reward
                state = next_state
                if done or truncated:
                    break
        self.env.close()
        self.total_rewards.append(total_reward)
        return total_reward
    
    def save(self, path):
        torch.save({
            'encoder': self.encoder.state_dict(),
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'target_critic': self.critic_target.state_dict(),
        }, path)
    
    def load(self, path):
        checkpoint = torch.load(path)
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_target.load_state_dict(checkpoint['target_critic'])

    def plot_learning_curve(self, path):
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        plt.plot(self.total_steps, self.total_rewards)
        plt.xlabel("Frames")
        plt.ylabel("Episode Return")
        plt.savefig(path)
        plt.close()


# 学習関数（1つの環境）
def train_env(env_id, logdir="models", steps=2000, eval_interval=1000):
    print(f"Training on {env_id}")
    env = gym.make(env_id, render_mode='rgb_array', height=64, width=64)
    env = AddRenderObservation(env, render_only=True)
    env = PreprocessObservationWrapper(env)
    action_dim = env.action_space.shape[0]

    model = SAC(env, feature_dim=512, action_dim=action_dim, hidden_dim=256, log_std_min=-20, log_std_max=2, gamma=0.99, tau=0.005, alpha=0.2, device="cpu")
    model.learn(total_timesteps=steps, eval_interval=eval_interval)
    env.close()
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    model.save(f"{logdir}/{env_id}_sac_model_final.pth")
    model.load(f"{logdir}/{env_id}_sac_model_final.pth")
    final_reward = model.eval_policy(record=False)
    print(f"Env: {env_id}, Final Reward: {final_reward:.4f}")
    model.plot_learning_curve(f"learning_curves/{env_id}.pdf")

def eval_env(env_id, logdir="models", record=False):
    env = gym.make(env_id, render_mode='rgb_array', height=64, width=64)
    env = AddRenderObservation(env, render_only=True)
    env = PreprocessObservationWrapper(env)
    action_dim = env.action_space.shape[0]

    model = SAC(env, feature_dim=512, action_dim=action_dim, hidden_dim=256, log_std_min=-20, log_std_max=2, gamma=0.99, tau=0.005, alpha=0.2, device="cpu")
    model.load(f"{logdir}/{env_id}_sac_model_final.pth")
    final_reward = model.eval_policy(record=record)
    print(f"Env: {env_id}, Final Reward: {final_reward:.4f}")
    env.close()

if __name__ == '__main__':
    env_ids = ["Ant-v5", "Walker2d-v5"]
    processes = []
    for env_id in env_ids:
        p = multiprocessing.Process(target=train_env, args=(env_id, "models", 1_000_000, 10000))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    for env_id in env_ids:
        p = multiprocessing.Process(target=eval_env, args=(env_id, "models", True))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
