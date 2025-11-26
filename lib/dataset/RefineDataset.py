import os
import cv2
import random
import numpy as np
import pandas as pd
from PIL import Image
from glob import glob

import torch
from torch.utils.data import Dataset
from torchvision import transforms


## noiser 训练数据
def to_rgb(image):
    rgb_image = Image.new("RGB", image.size)
    rgb_image.paste(image)
    return rgb_image


class RefineDataset(Dataset):
    def __init__(self, data_dirA, data_dirB, transforms_=None, unaligned=False):
        self.transform = transforms_
        self.unaligned = unaligned

        self.files_A = sorted(glob(os.path.join(data_dirA, "*.*")))
        self.files_B = sorted(glob(os.path.join(data_dirB, "*.*")))

    def __getitem__(self, index):
        A_file = self.files_A[index % len(self.files_A)]
        A_filename = os.path.split(A_file)[-1]
        image_A = Image.open(A_file)

        if self.unaligned:
            B_file = self.files_B[random.randint(0, len(self.files_B) - 1)]
        else:
            B_file = self.files_B[index % len(self.files_B)]
        B_filename = os.path.split(B_file)[-1]
        image_B = Image.open(B_file)

        # Convert grayscale images to rgb
        if image_A.mode != "RGB":
            image_A = to_rgb(image_A)
        if image_B.mode != "RGB":
            image_B = to_rgb(image_B)

        item_A = self.transform(image_A)[:1, ...]
        item_B = self.transform(image_B)[:1, ...]
        return {"A": item_A, "B": item_B, "A_filename": A_filename, "B_filename": B_filename}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))