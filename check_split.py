import os

train_files = [f for f in os.listdir("dataset/train") if f.endswith(".npy")]
val_files = [f for f in os.listdir("dataset/val") if f.endswith(".npy")]
test_files = [f for f in os.listdir("dataset/test") if f.endswith(".npy")]

print("Train:", len(train_files))
print("Validation:", len(val_files))
print("Test:", len(test_files))