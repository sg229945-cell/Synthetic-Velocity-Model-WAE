import os
import numpy as np
import matplotlib.pyplot as plt

files = [f for f in os.listdir("dataset") if f.endswith(".npy")]
print("Total files:", len(files))

for idx in [25, 100, 250, 500, 1000]:

    model = np.load(f"dataset/model_{idx}.npy")

    plt.figure(figsize=(5,5))
    plt.imshow(model, cmap="jet")
    plt.colorbar()
    plt.title(f"Model {idx}")
    plt.show()

print("Shape:", model.shape)
print("Min:", model.min())
print("Max:", model.max())
print("Unique values:", np.unique(model))

plt.imshow(model, cmap="jet")
plt.colorbar()
plt.title("Model 0")
plt.show()