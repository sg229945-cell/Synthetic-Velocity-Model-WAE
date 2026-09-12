import torch
import numpy as np
import matplotlib.pyplot as plt
from models.wae import WAE
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# -----------------------------
# Load model
# -----------------------------
model = WAE(latent_dim=16)

# Initialize encoder.fc
dummy_input = torch.randn(1, 1, 128, 128)
model(dummy_input)

# Load trained weights
model.load_state_dict(
    torch.load("best_wae_model.pth", map_location="cpu")
)

model.eval()

# -----------------------------
# Load sample
# -----------------------------
x_original = np.load("dataset/train/model_1853.npy").astype(np.float32)

# -----------------------------
# SAME NORMALIZATION AS TRAINING
# -----------------------------
x_normalized = (x_original - x_original.min()) / (
    x_original.max() - x_original.min() + 1e-8
)

# Convert to tensor
x = torch.tensor(x_normalized).unsqueeze(0).unsqueeze(0)

# -----------------------------
# Reconstruction
# -----------------------------
with torch.no_grad():
    recon, z = model(x)

# Convert reconstruction to numpy
recon = recon.squeeze().numpy()

# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.title("Original (Normalized)")
plt.imshow(x_normalized, cmap="viridis")
plt.colorbar()

plt.subplot(1, 2, 2)
plt.title("Reconstructed")
plt.imshow(recon, cmap="viridis")
plt.colorbar()

plt.tight_layout()
plt.show()