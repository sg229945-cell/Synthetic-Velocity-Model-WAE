import numpy as np

x = np.load("dataset/train/model_1853.npy")

print("Shape:", x.shape)
print("Min:", x.min())
print("Max:", x.max())