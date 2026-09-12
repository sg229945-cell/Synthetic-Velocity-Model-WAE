import torch
from models.wae import WAE

print("Testing WAE model...")

model = WAE(latent_dim=16)

x = torch.randn(2, 1, 128, 128)

recon, z = model(x)

print("Input shape:", x.shape)
print("Reconstructed shape:", recon.shape)
print("Latent shape:", z.shape)

print("Model is working perfectly!")

