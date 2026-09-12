import os
import random
import shutil

dataset_dir = "dataset"

train_dir = os.path.join(dataset_dir, "train")
val_dir = os.path.join(dataset_dir, "val")
test_dir = os.path.join(dataset_dir, "test")

os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

files = [f for f in os.listdir(dataset_dir) if f.endswith(".npy")]

random.shuffle(files)

n = len(files)

train_files = files[:int(0.8*n)]
val_files = files[int(0.8*n):int(0.9*n)]
test_files = files[int(0.9*n):]

for f in train_files:
    shutil.copy(
        os.path.join(dataset_dir, f),
        os.path.join(train_dir, f)
    )

for f in val_files:
    shutil.copy(
        os.path.join(dataset_dir, f),
        os.path.join(val_dir, f)
    )

for f in test_files:
    shutil.copy(
        os.path.join(dataset_dir, f),
        os.path.join(test_dir, f)
    )

print("Train:", len(train_files))
print("Validation:", len(val_files))
print("Test:", len(test_files))
print("Dataset split completed!")