import torch
import torch.nn as nn

# =========================
# Encoder
# =========================
class Encoder(nn.Module):
    def __init__(self, latent_dim=16):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),   # 64 → 32
            nn.ReLU(),

            nn.Conv2d(32, 64, 4, 2, 1),  # 32 → 16
            nn.ReLU(),

            nn.Conv2d(64, 128, 4, 2, 1), # 16 → 8
            nn.ReLU()
        )

        # FIX: no manual flatten assumption
        self.flatten = nn.Flatten()
        self.fc = nn.LazyLinear(latent_dim)

    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        z = self.fc(x)
        return z

#    Decoder 

class Decoder(nn.Module):
    def __init__(self, latent_dim=16):
        super().__init__()

        self.fc = nn.Linear(latent_dim, 128 * 8 * 8)

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 4, 2, 1),  # 8 → 16
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, 4, 2, 1),   # 16 → 32
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, 4, 2, 1),    # 32 → 64
            nn.ReLU(),

            nn.ConvTranspose2d(32, 1, 4, 2, 1),     # 64 → 128
            nn.Sigmoid()
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 128, 8, 8)
        x = self.deconv(x)
        return x
# =========================
# WAE Model
# =========================
class WAE(nn.Module):
    def __init__(self, latent_dim=16):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z