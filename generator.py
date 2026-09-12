import numpy as np

def create_background(size=128):
    model = np.ones((size, size))
    return model

print(create_background())
background = create_background()

print(background.shape)
def normalize_01(data):

    return (data - data.min()) / (data.max() - data.min())
sample = np.array([0.7, 0.8, 1.0])
print(normalize_01(sample))
def normalize_minus1_1(data):

    return 2 * data - 1
sample = np.array([0.7, 0.8, 1.0])

print(normalize_minus1_1(sample))
