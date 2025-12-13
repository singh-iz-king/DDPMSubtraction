import pandas as pd
import numpy as np

# Take in Mnist Data, randomly choose two digits, and their difference, record indices

mnist = pd.read_csv("/Users/keshavsingh/Downloads/MNIST_CSV/mnist_train.csv")

nrows, ncols = mnist.shape

indices = np.arange(nrows)

labels = mnist.iloc[:, 0].to_numpy()

labels_to_indices = {lab: np.where(labels == lab)[0] for lab in range(10)}

n = int(100000)

x1_idx = np.random.choice(indices, size=n, replace=True)
x2_idx = np.random.choice(indices, size=n, replace=True)
r_idx = np.zeros(n, dtype=np.int32)

for i in range(n):
    x1_lab = labels[x1_idx[i]]
    x2_lab = labels[x2_idx[i]]
    r_lab = abs(x1_lab - x2_lab)
    r_idx[i] = np.random.choice(labels_to_indices[r_lab], size=1)[0]

subtraction_mnist = pd.DataFrame({"x1": x1_idx, "x2": x2_idx, "r": r_idx})

subtraction_mnist.to_csv("subtraction_mnist.csv", index=False)
