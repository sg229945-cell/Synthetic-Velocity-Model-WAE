import torch
from torch.utils.data import Dataset
import numpy as np
import os
import torch
print("Torch version:", torch.__version__)
print(" dataset_loader started")

class VelocityDataset(Dataset):
    def __init__(self, folder):
        self.files = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.endswith(".npy")
        ]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        x = np.load(self.files[idx]).astype(np.float32)

        #  normalization (VERY IMPORTANT for WAE)
        x = (x - x.min()) / (x.max() - x.min() + 1e-8)

        x = torch.tensor(x)

        # add channel dimension → (1, H, W)
        x = x.unsqueeze(0)

        return x