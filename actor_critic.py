import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from rssm import RSSM, CNNDecoder, RewardPredictor, ContinuePredictor, WorldModel

class Actor(nn.Module):
    def __init__(self, action_dim, hidden_dim=128, num_categorical=32, class_size=32):
        super().__init__()
        self.action_dim = action_dim
        categorical_dim = num_categorical * class_size
        
        # Actor network that takes hidden state and categorical state as input
        self.net = nn.Sequential(
            nn.Linear(hidden_dim + categorical_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, action_dim * 2)  # Output mean and log_std for each action dimension
        )
        
    def forward(self, h, z):
        x = torch.cat([h, z], dim=-1)
        x = self.net(x)
        mean, log_std = torch.chunk(x, 2, dim=-1)
        log_std = torch.clamp(log_std, -20, 2)  # Prevent too small or large std
        return mean, log_std
    
    def get_action(self, h, z, deterministic=False):
        mean, log_std = self.forward(h, z)
        if deterministic:
            return mean
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        action = normal.rsample()  # Reparameterization trick
        return action

class Critic(nn.Module):
    def __init__(self, hidden_dim=128, num_categorical=32, class_size=32):
        super().__init__()
        categorical_dim = num_categorical * class_size
        
        # Critic network that takes hidden state, categorical state, and action as input
        self.net = nn.Sequential(
            nn.Linear(hidden_dim + categorical_dim + action_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, 1)  # Output Q-value
        )
    
    def forward(self, h, z, action):
        x = torch.cat([h, z, action], dim=-1)
        return self.net(x)

class ActorCriticTrainer:
    def __init__(self, world_model, actor, critic, action_dim, 
                 num_categorical=32, class_size=32, hidden_dim=128,
                 imagination_horizon=15, batch_size=50,
                 actor_lr=3e-4, critic_lr=3e-4, gamma=0.99, lambda_=0.95):
        self.world_model = world_model
        self.actor = actor
        self.critic = critic
        self.action_dim = action_dim
        self.num_categorical = num_categorical
        self.class_size = class_size
        self.hidden_dim = hidden_dim
        self.imagination_horizon = imagination_horizon
        self.batch_size = batch_size
        self.gamma = gamma
        self.lambda_ = lambda_  # GAE parameter
        
        # Optimizers
        self.actor_optimizer = torch.optim.Adam(actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(critic.parameters(), lr=critic_lr)
        
    def imagine_trajectory(self, h, z):
        """Generate imagined trajectory using the world model"""
        batch_size = h.shape[0]
        device = h.device
        
        # Initialize trajectory tensors
        imagined_h = [h]
        imagined_z = [z]
        imagined_actions = []
        imagined_rewards = []
        imagined_values = []
        imagined_log_probs = []
        
        # Generate imagined trajectory
        for t in range(self.imagination_horizon):
            # Get action from actor
            action = self.actor.get_action(h, z)
            
            # Get value estimate from critic
            value = self.critic(h, z, action)
            
            # Get action log probability
            mean, log_std = self.actor(h, z)
            std = log_std.exp()
            normal = torch.distributions.Normal(mean, std)
            log_prob = normal.log_prob(action).sum(dim=-1, keepdim=True)
            
            # Step world model
            h = self.world_model.rssm.rnn(torch.cat([z, action], dim=-1), h)
            prior_logits = self.world_model.rssm.prior_net(h)
            prior_logits = prior_logits.view(batch_size, self.num_categorical, self.class_size)
            
            # Sample next latent state
            z = F.gumbel_softmax(prior_logits, tau=1.0, hard=True)
            z = z.view(batch_size, -1)
            
            # Predict reward
            reward = self.world_model.reward_predictor(torch.cat([h, z], dim=-1))
            
            # Store trajectory
            imagined_h.append(h)
            imagined_z.append(z)
            imagined_actions.append(action)
            imagined_rewards.append(reward)
            imagined_values.append(value)
            imagined_log_probs.append(log_prob)
        
        # Stack tensors
        imagined_h = torch.stack(imagined_h[:-1], dim=1)  # [batch, horizon, hidden_dim]
        imagined_z = torch.stack(imagined_z[:-1], dim=1)  # [batch, horizon, categorical_dim]
        imagined_actions = torch.stack(imagined_actions, dim=1)  # [batch, horizon, action_dim]
        imagined_rewards = torch.stack(imagined_rewards, dim=1)  # [batch, horizon, 1]
        imagined_values = torch.stack(imagined_values, dim=1)  # [batch, horizon, 1]
        imagined_log_probs = torch.stack(imagined_log_probs, dim=1)  # [batch, horizon, 1]
        
        return imagined_h, imagined_z, imagined_actions, imagined_rewards, imagined_values, imagined_log_probs
    
    def compute_gae(self, rewards, values, dones):
        """Compute Generalized Advantage Estimation"""
        batch_size, horizon = rewards.shape[:2]
        device = rewards.device
        
        # Initialize advantage and returns
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        
        # Compute advantages using GAE
        gae = 0
        for t in reversed(range(horizon)):
            if t == horizon - 1:
                next_value = values[:, t]
            else:
                next_value = values[:, t + 1]
            
            delta = rewards[:, t] + self.gamma * next_value * (1 - dones[:, t]) - values[:, t]
            gae = delta + self.gamma * self.lambda_ * (1 - dones[:, t]) * gae
            advantages[:, t] = gae
        
        # Compute returns
        returns = advantages + values
        
        return advantages, returns
    
    def update(self, states, actions, rewards, next_states, dones):
        """Update actor and critic using imagined trajectories"""
        # Get initial hidden and latent states from world model
        with torch.no_grad():
            h = self.world_model.rssm.h0.expand(self.batch_size, -1)
            z = self.world_model.rssm.z0.expand(self.batch_size, -1)
        
        # Generate imagined trajectory
        imagined_h, imagined_z, imagined_actions, imagined_rewards, imagined_values, imagined_log_probs = \
            self.imagine_trajectory(h, z)
        
        # Compute advantages and returns
        advantages, returns = self.compute_gae(
            imagined_rewards, imagined_values, 
            torch.zeros_like(imagined_rewards)  # No dones in imagined trajectory
        )
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Update critic
        critic_loss = F.mse_loss(imagined_values, returns.detach())
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update actor
        actor_loss = -(imagined_log_probs * advantages.detach()).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'mean_reward': imagined_rewards.mean().item(),
            'mean_value': imagined_values.mean().item()
        }

# Example usage:
if __name__ == "__main__":
    # Initialize models
    action_dim = 8  # Example action dimension
    world_model = WorldModel(
        rssm=RSSM(action_dim=action_dim),
        decoder=CNNDecoder(),
        reward_predictor=RewardPredictor(),
        continue_predictor=ContinuePredictor()
    )
    
    actor = Actor(action_dim=action_dim)
    critic = Critic(action_dim=action_dim)
    
    # Initialize trainer
    trainer = ActorCriticTrainer(
        world_model=world_model,
        actor=actor,
        critic=critic,
        action_dim=action_dim
    )
    
    # Training loop
    for epoch in range(1000):
        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = buffer.sample(
            batch_len=50, sequence_length=1
        )
        
        # Convert to tensors
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)
        
        # Update actor and critic
        metrics = trainer.update(states, actions, rewards, next_states, dones)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}:")
            print(f"  Critic Loss: {metrics['critic_loss']:.3f}")
            print(f"  Actor Loss: {metrics['actor_loss']:.3f}")
            print(f"  Mean Reward: {metrics['mean_reward']:.3f}")
            print(f"  Mean Value: {metrics['mean_value']:.3f}") 