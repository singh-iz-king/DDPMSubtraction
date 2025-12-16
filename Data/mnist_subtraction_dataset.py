import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np


class MnistSubtractionDataset(Dataset):

    def __init__(self, mnist_file_path, subtraction_file_path, transform=None):

        super().__init__()
        self.mnist = pd.read_csv(mnist_file_path)
        self.images = self.mnist.iloc[:, 1:].to_numpy(dtype=np.float32)
        self.subtraction = pd.read_csv(subtraction_file_path)
        self.transform = transform

    def __len__(self):
        return len(self.subtraction)

    def __getitem__(self, index):

        im1_idx, im2_idx, res_idx = map(int, self.subtraction.iloc[index].to_list())

        im1 = self.images[im1_idx]
        im2 = self.images[im2_idx]
        res = self.images[res_idx]

        im1, im2, res = im1.reshape(28, 28), im2.reshape(28, 28), res.reshape(28, 28)

        if self.transform:
            im1, im2, res = (
                self.transform(im1),
                self.transform(im2),
                self.transform(res),
            )

        return im1, im2, res
