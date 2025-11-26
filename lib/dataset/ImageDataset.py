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


# 简单图片数据集
class ImageDataset(Dataset):
    def __init__(self, data_dir, transforms_=transforms.ToTensor()) -> None:
        super(ImageDataset, self).__init__()
        self.data_dir = data_dir
        self.imgfiles = glob(os.path.join(self.data_dir, "*.png")) + glob(os.path.join(self.data_dir, "*.jpg"))
        self.transforms_ = transforms_

    def __len__(self):
        return len(self.imgfiles)

    def resize(self, img, hw=[512, 256]):
        h, w, d = img.shape

        if w < h / 2:
            margin = (h / 2 - w) // 2
            img = cv2.copyMakeBorder(img, 0, 0, int(margin), int(margin), cv2.BORDER_CONSTANT)
        else:
            margin = (w * 2 - h) // 2
            img = cv2.copyMakeBorder(img, int(margin), int(margin), 0, 0, cv2.BORDER_CONSTANT)

        img = cv2.resize(img, [hw[1], hw[0]])
        return img

    def __getitem__(self, index):
        imgfile = self.imgfiles[index]
        _, filename = os.path.split(imgfile)
        img = cv2.imread(imgfile)
        # img=self.resize(img)

        return self.transforms_(img), filename