import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.transforms import v2
from torchvision.utils import make_grid
from torch.utils.data import TensorDataset, DataLoader
from torchvision.io import write_video
import matplotlib.pyplot as plt

# Hyperparameters
BATCH_SIZE = 1
SEQ_LEN = 16  # Length of image sequences
LATENT_DIM = 32  # Latent dimension for VAE
HIDDEN_DIM = 128  # Hidden dimension for GRU
IMAGE_SIZE = 128  
NUM_EPOCHS = 1000
LEARNING_RATE = 1e-3

def create_dataset(obs_traj, capacity=10, sequence_length=SEQ_LEN):
    """ select the starting point at random, then collect the sequence data from there"""
    max_steps, obs_shape = obs_traj.shape[0], obs_traj.shape[1:]
    starting_idxs = torch.tensor([idx * (sequence_length - 1) for idx in range(max_steps // SEQ_LEN)] )
    # starting_idxs = torch.randint(0, max_steps - sequence_length, (capacity,))
    idxs = torch.stack([torch.arange(start, start + sequence_length) for start in starting_idxs])
    return TensorDataset(obs_traj[idxs])

local_datapath = 'expert_demos/episode-0_128x128.pt'
obs_traj = torch.load(local_datapath)
obs_traj = obs_traj.permute(0, 3, 1, 2) # Change (N, H, W, C) to (N, C, H, W)
transforms = v2.ToDtype(torch.float32, scale=True) #  map the [0, 255] range into [0, 1]
obs_traj = transforms(obs_traj)
train_set = create_dataset(obs_traj)
# train_set = TensorDataset(obs_traj)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=False)

import gymnasium as gym
import gymnasium_robotics # v1.3.0
from gymnasium.wrappers import AddRenderObservation
gym.register_envs(gymnasium_robotics)
env = gym.make('FrankaKitchen-v1', render_mode='rgb_array', width=128, height=128)
env = AddRenderObservation(env, render_only=True)


class Encoder(nn.Module):
    def __init__(self, latent_dim, feature_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),  # 128x128 -> 64x64
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=4, stride=2, padding=1),  # 64x64 -> 32x32
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 32x32 -> 16x16
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, feature_dim),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(feature_dim, latent_dim)
        self.fc_logvar = nn.Linear(feature_dim, latent_dim)

    def forward(self, x):
        feature = self.encoder(x)
        mu = self.fc_mu(feature)
        logvar = self.fc_logvar(feature)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, 256),
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

    def forward(self, z):
        h = self.decoder_fc(z)
        h = h.view(-1, 64, 16, 16)
        return self.decoder_conv(h)

class SequencePredictor(nn.Module):
    def __init__(self, latent_dim, hidden_dim):
        super().__init__()
        self.rnn = nn.GRU(latent_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, latent_dim)

    def forward(self, z, h):
        y, h = self.rnn(z, h)
        pred_z = self.fc(y)
        return pred_z, h

class SequentialVAE(nn.Module):
    def __init__(self, latent_dim, hidden_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
        self.sequence = SequencePredictor(latent_dim, hidden_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, h_last):
        batch_size, seq_length, *_ = x.shape
        device = x.device

        mu_list, logvar_list = [], []
        zt_list, zt_pred_list = [], []
        recon_list = []
        ht = h_last
        for t in range(seq_length):
            xt = x[:, t] # shape:[batch_size, C, H, W]

            # Encode images into latent representations
            mu, logvar = self.encoder(xt)
            mu_list.append(mu)
            logvar_list.append(logvar)
            zt = self.reparameterize(mu, logvar) # z: (batch_size. latent_dim)
            zt_list.append(zt)

            # Predict the next latent variable
            zt_pred, ht = self.sequence(zt, ht)
            zt_pred_list.append(zt_pred)
            # Decode latent representations back into images
            recon_x = self.decoder(zt)  # recon_x: (batch_size, C, H, W)
            recon_list.append(recon_x)
        
        recon_seq = torch.stack(recon_list, dim=1)
        mu_seq = torch.stack(mu_list, dim=1)
        logvar_seq = torch.stack(logvar_list, dim=1)
        z_seq = torch.stack(zt_list[1:], dim=1)
        z_seq_pred = torch.stack(zt_pred_list[:-1], dim=1)
        h_last = ht.detach()
        return recon_seq, mu_seq, logvar_seq, z_seq, z_seq_pred, h_last

    # Loss Functions
    def compute_loss(self, recon_seq, input_seq, mu_seq, logvar_seq, z_seq, z_seq_pred):
        _, seq_len, *_ = recon_seq.shape
        recon_loss, kl_loss, rnn_loss = 0, 0, 0
        for t in range(seq_len):
            recon_loss += F.mse_loss(recon_seq[:, t], input_seq[:, t], reduction='sum')
            kl_loss += -0.5 * torch.sum(1 + logvar_seq[:, t] - mu_seq[:, t].pow(2) - logvar_seq[:, t].exp())
            if t != (seq_len - 1):
                rnn_loss += F.mse_loss(z_seq[:, t], z_seq_pred[:, t], reduction='sum')

        # recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
        # kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss, kl_loss, rnn_loss

    # def rnn_loss(self, z, rnn_batch=5):
    #     input_seq = z[0:-rnn_batch]
    #     target_seq = z[rnn_batch:]
    #     train_set = TensorDataset(input_seq, target_seq)
    #     train_loader = DataLoader(train_set, batch_size=rnn_batch, shuffle=False)
    #     rnn_loss = 0
    #     h = None
    #     for batch_id, (z, next_z) in enumerate(train_loader):
    #         pred_z, h = self.sequence(z, h)
    #         rnn_loss += nn.functional.mse_loss(pred_z, next_z, reduction='sum')
    #     return rnn_loss

# Dataset and DataLoader
# transform = transforms.Compose([transforms.ToTensor()])
# dataset = MNIST(root="./data", train=True, transform=transform, download=True)
# dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


# Initialize model, optimizer, and scheduler
model = SequentialVAE(LATENT_DIM, HIDDEN_DIM)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Generate New Data
def generate_sequence(model, initial_image, seq_len=SEQ_LEN):
    model.eval()
    with torch.no_grad():
        # Encode initial image
        mu, logvar = model.encoder(initial_image)
        z = model.reparameterize(mu, logvar)
        pred_z = z[0].unsqueeze(0)  # Initialize the predicted sequence with the first latent vector
        ht = None
        for t in range(seq_len - 1):
            y, ht = model.sequence(pred_z[t].unsqueeze(0), ht)
            pred_z = torch.cat([pred_z, y], dim=0)
            
        # Decode sequence
        generated_images = model.decoder(pred_z.squeeze(1))
        return generated_images

# Training Loop
losses = []
loss_history = []
for epoch in range(1, NUM_EPOCHS+1):
    pred_seqz = []
    loss_sum = 0
    L1_sum, L2_sum = 0, 0
    batch_cnt = 0
    initial_imgs = []
    h_last = None
    for x in train_loader:
        batch_cnt += 1
        input_seq = x[0] # shape [batch_size, seq_length, C, H, W]
        initial_imgs.append(input_seq)
        # Forward pass
        # recon_x, mu, logvar, z = model(x)
        recon_seq, mu_seq, logvar_seq, z_seq, z_seq_pred, h_last = model(input_seq, h_last)
        # recon_x: (batch_dim, 1, 28, 28), mu: (batch_dim, latent_dim), logvar: (batch_dim, latent_dim), z: (batch_dim, latent_dim)

        # Compute loss
        recon_loss, kl_loss, rnn_loss = model.compute_loss(recon_seq, input_seq, mu_seq, logvar_seq, z_seq, z_seq_pred)
        L1 = recon_loss + kl_loss
        L2 = rnn_loss
        loss = L1 + L2

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()
        L1_sum += L1.item()
        L2_sum += L2.item()
        
        # if batch_cnt % 100 == 0:  
        #     print(f"Epoch [{epoch}/{NUM_EPOCHS}], Batch [{batch_cnt}/{len(train_loader)}], Loss: {loss.item():.2f}, L1: {L1.item():.2f}, L2: {L2.item():.2f}")

    loss_avg = loss_sum / batch_cnt+1
    l1_avg = L1_sum / batch_cnt+1
    l2_avg = L2_sum / batch_cnt+1
    loss_history.append(loss_avg)
    print(f"Epoch [{epoch}/{NUM_EPOCHS}], L1 loss: {l1_avg:.2f}, L2 loss {l2_avg:.2f} Average Loss: {loss_avg:.2f}\n")

    # Generate a sequence of images starting from the real env's first observation
    if epoch % 10 == 0:
        initial_image = transforms(torch.as_tensor(env.reset()[0].copy())).permute(2, 0, 1).unsqueeze(0)
        random_idx = torch.randint(0, len(initial_imgs)-1, (1,))
        initial_image = initial_imgs[random_idx].squeeze(0)
        generated_images = generate_sequence(model, initial_image, seq_len=236)

        video_array = generated_images.permute(0, 2, 3, 1)
        # Convert from float32 [0,1] to uint8 [0,255]
        video_array = (video_array * 255).to(torch.uint8)
        write_video(filename=f'epoch-{epoch}.mp4', video_array=video_array, fps=12)
        # grid_img = make_grid(
        # generated_images,
        # nrow=8,
        # padding=2,
        # normalize=True
        # )

        # plt.imshow(grid_img.permute(1, 2, 0))
        # plt.axis("off")
        # if not os.path.exists('figures'):
        #     os.mkdir('figures')
        # plt.savefig(f'figures/Epoch{epoch}_{IMAGE_SIZE}x{IMAGE_SIZE}.png')

# loss_history = [3855.8399574825326, 3124.046598680755, 3007.084531802378, 2952.9197947238526, 2916.269733180613, 2893.5113998299094, 2874.27133234079, 2860.4884756244996, 2847.418282458978, 2838.09587066227]
plt.figure(figsize=(10, 5))
plt.plot(range(1, NUM_EPOCHS+1), loss_history)
plt.xlabel("Epoch", fontsize=15)
plt.ylabel("Total Loss", fontsize=15)
plt.tight_layout()
plt.grid(True)
plt.savefig('figures/Learning_curve')
print("Training completed. Saved generated images and learning curve to ./figures")