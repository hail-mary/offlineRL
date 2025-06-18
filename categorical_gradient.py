import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.one_hot_categorical import OneHotCategorical

sample = OneHotCategorical(torch.tensor([0.1, 0.3, 0.6]))
print(sample.sample((4,)))  # Sample a category

# class CategoricalStraightThrough(nn.Module):
#     def __init__(self, num_categories, latent_dim):
#         super().__init__()
#         self.num_categories = num_categories  # Number of categories
#         self.latent_dim = latent_dim  # Dimension of the latent space

#         # Neural network to output logits for each category and latent dimension
#         self.fc = nn.Linear(latent_dim, num_categories * latent_dim)

#     def forward(self, x):
#         # Compute logits
#         logits = self.fc(x)  # Shape: (batch_size, num_categories * latent_dim)
#         logits = logits.view(-1, self.num_categories, self.latent_dim)  # Reshape to (batch_size, num_categories, latent_dim)

#         # Gumbel-Softmax sampling (differentiable)
#         probs = F.softmax(logits, dim=1)  # Softmax along the num_categories dimension
#         gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits)))  # Gumbel noise
#         gumbel_logits = (logits + gumbel_noise) / 1.0  # Temperature = 1.0
#         samples = F.softmax(gumbel_logits, dim=1)

#         # Straight-through gradients
#         # During forward pass: use the sampled discrete value
#         # During backward pass: use the continuous approximation (softmax)
#         discrete_samples = torch.argmax(samples, dim=1)  # Shape: (batch_size, latent_dim)
#         one_hot_samples = F.one_hot(discrete_samples, num_classes=self.num_categories).float()  # Shape: (batch_size, latent_dim, num_categories)
#         one_hot_samples = one_hot_samples.permute(0, 2, 1)  # Shape: (batch_size, num_categories, latent_dim)
#         straight_through_samples = one_hot_samples - samples.detach() + samples

#         return straight_through_samples, probs

# # Example Usage
# if __name__ == "__main__":
#     # Define the model
#     num_categories = 3  # Number of categories
#     latent_dim = 4  # Dimension of the latent space
#     model = CategoricalStraightThrough(num_categories, latent_dim)

#     # Generate random input
#     batch_size = 1
#     x = torch.randn(batch_size, latent_dim)

#     # Forward pass
#     samples, probs = model(x)
#     print("Samples shape:", samples.shape)  # Should be (batch_size, num_categories, latent_dim)
#     print("Probs shape:", probs.shape)  # Should be (batch_size, num_categories, latent_dim)
#     print(samples)
#     print(probs)

#     # Backward pass (example loss)
#     target = torch.randint(0, num_categories, (batch_size, latent_dim))  # Random target
#     target_one_hot = F.one_hot(target, num_classes=num_categories).float()  # Shape: (batch_size, latent_dim, num_categories)
#     target_one_hot = target_one_hot.permute(0, 2, 1)  # Shape: (batch_size, num_categories, latent_dim)
#     loss = F.cross_entropy(probs, target_one_hot)
#     loss.backward()

#     print("Gradients computed successfully!")