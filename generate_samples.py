import torch
import numpy as np
import matplotlib.pyplot as plt
from models.wae import WAE
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# load model
model = WAE(latent_dim=16)
model.load_state_dict(torch.load("wae_model.pth", map_location="cpu"))
model.eval()

# generate random latent vectors
num_samples = 4
z = torch.randn(num_samples, 16)

# decode only
with torch.no_grad():
    generated = model.decoder(z)

generated = generated.squeeze().numpy()

# plot
plt.figure(figsize=(8,8))

for i in range(num_samples):
    plt.subplot(2,2,i+1)
    plt.title(f"Generated {i+1}")
    plt.imshow(generated[i], cmap="viridis")

plt.show()