import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from models.wae import WAE
from dataset_loader import VelocityDataset

# -----------------------------
# IoU Function
# -----------------------------
def iou(pred, target):
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return intersection / (union + 1e-8)

# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Training on:", device)

# -----------------------------
# Dataset
# -----------------------------
train_dataset = VelocityDataset("dataset/train")
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

val_dataset = VelocityDataset("dataset/val")
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# -----------------------------
# Model
# -----------------------------
model = WAE(latent_dim=16).to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3)
mse_loss = nn.MSELoss()

# -----------------------------
# MMD Loss
# -----------------------------
def mmd_loss(z, prior_z):
    x_kernel = torch.mm(z, z.t())
    y_kernel = torch.mm(prior_z, prior_z.t())
    xy_kernel = torch.mm(z, prior_z.t())
    return x_kernel.mean() + y_kernel.mean() - 2 * xy_kernel.mean()

# -----------------------------
# Tracking
# -----------------------------
epochs = 10
loss_list = []
iou_list = []

best_loss = float("inf")

# -----------------------------
# Training Loop
# -----------------------------
for epoch in range(epochs):
    model.train()

    total_loss = 0
    total_iou = 0

    for x in train_loader:
        x = x.to(device)

        # forward
        recon, z = model(x)

        # losses
        recon_loss = mse_loss(recon, x)
        prior_z = torch.randn_like(z)
        mmd = mmd_loss(z, prior_z)

        loss = recon_loss + 0.1 * mmd

        # IoU
        iou_score = iou(recon, x)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_iou += iou_score.item()

        loss_list.append(loss.item())
        iou_list.append(iou_score.item())

    avg_loss = total_loss / len(train_loader)
    avg_iou = total_iou / len(train_loader)

    # -----------------------------
    # Validation
    # -----------------------------
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for x in val_loader:
            x = x.to(device)
            recon, z = model(x)
            loss_val = mse_loss(recon, x)
            val_loss += loss_val.item()

    val_loss = val_loss / len(val_loader)

    print(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f} IoU: {avg_iou:.4f} ValLoss: {val_loss:.4f}")

    # -----------------------------
    # Save Best Model
    # -----------------------------
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), "best_wae_model.pth")

# -----------------------------
# Save final model
# -----------------------------
torch.save(model.state_dict(), "wae_model.pth")
print("Training complete. Model saved.")

# -----------------------------
# Plot Loss
# -----------------------------
plt.plot(loss_list)
plt.title("Training Loss Curve")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.show()

# -----------------------------
# Plot IoU
# -----------------------------
plt.plot(iou_list)
plt.title("IoU Curve")
plt.xlabel("Iterations")
plt.ylabel("IoU")
plt.show()