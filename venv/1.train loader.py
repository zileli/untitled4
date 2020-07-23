from __future__ import print_function, division


import os
import torch
import pandas
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from normalization import normalization

class ourDataset(Dataset):

    data=normalization.normal_data(file_name='validate.csv', train_rows_num=10)



    def __len__(self):
        return len(self.data)



A=ourDataset()
print(ourDataset.__len__(A))
print(A.data)