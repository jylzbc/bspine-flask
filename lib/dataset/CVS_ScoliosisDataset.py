import os
import cv2
import json
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms
from scipy.interpolate import interp1d
from libs.models.bspline.bspline import BSpline

class CJUHJLU_Dataset(Dataset):

    def __init__(self, args, split=None, transforms_=transforms.ToTensor()):
        super(CJUHJLU_Dataset, self).__init__()
        self.transforms_ = transforms_
        # self.split = 'test' if args.mode == 'test' else ('val' if split == 'val' else 'training')
        if args.mode == 'training':
            self.split = 'training'
        elif args.mode == 'val':
            self.split = 'val'
        elif args.mode == 'test':
            self.split = 'test'
        else:
            raise ValueError("Invalid mode. Must be 'training', 'val', or 'test'.")
        self.data_dir = os.path.join(args.data_dir, self.split)
        self.n = args.general.n
        self.k = args.general.k       
        self.json_list = sorted(glob(os.path.join(self.data_dir, "label", "*.json")))
        self.load_data(self.json_list)

    def get_images(self):
        return self.images.permute(0, 3, 1, 2), self.filenames, self.vertebra_masks.permute(0,3,1,2), self.angles

    def interploate(self, centerline):
        """ interploate 34 points from 17 points 

        Args:
            centerline (_type_): _description_

        Returns:
            _type_: _description_
        """
        centerline = np.array(centerline) 
        from scipy.interpolate import interp1d
        f = interp1d(centerline[:, 1], centerline[:, 0], kind="cubic") 
        x = np.linspace(centerline[:, 1][0], centerline[:, 1][-1], 34) 
        y = f(x)
        centerline = np.stack([y, x], axis=-1) 
        return centerline

    def gene_heatmap(self, centerline):
        """Generate a heatmap from the given centerline.

        Args:
            centerline (np.ndarray): Normalized centerline points.

        Returns:
            np.ndarray: Generated heatmap.
        """
        heatmap = np.zeros([512, 256, 1])
        centerline[:, 0], centerline[:, 1] = centerline[:, 0] * 255, centerline[:, 1] * 511 
        for x, y in centerline:
            x, y = min(x, 256 - 2), min(y, 512 - 2)  
            heatmap[int(y), int(x)] = 255 
            heatmap[int(y + 1), int(x)] = 255
            heatmap[int(y), int(x + 1)] = 255
            heatmap[int(y + 1), int(x + 1)] = 255
        heatmap = heatmap.astype(np.uint8)
        return heatmap

    def load_data(self, json_list):
        """load data, including images, masks, heatmaps, bsling paras and angles

        Args:
            json_list (list): store the json files
        """
        images, spine_masks, vertebra_masks, heatmaps, centerlines, bsplines, angles, filenames = [], [], [], [], [], [], [], []
        
        for jsonfile in json_list:
            prefix, filename = os.path.split(jsonfile)

            with open(jsonfile, "r") as f:
                label = json.loads(f.read())
            image_path = label.get("imagePath", filename[:-5] + ".jpg")
            
            # image
            image_file = os.path.join(
                self.data_dir, "data", filename[:-5] + ".jpg" if os.path.exists(os.path.join(self.data_dir, "data", filename[:-5] + ".jpg")) else filename[:-5] + ".png")
            image = cv2.imread(image_file)/255

            # spine mask
            spine_mask = cv2.imread(
                os.path.join(
                    self.data_dir, "spine_mask", filename[:-5] + ".jpg" if os.path.exists(os.path.join(self.data_dir, "spine_mask", filename[:-5] + ".jpg")) else filename[:-5] + ".png"))[..., :1]/255

            # vertebra mask
            vertebra_mask = cv2.imread(
                os.path.join(
                    self.data_dir, "vertebra_mask", filename[:-5] + ".jpg" if os.path.exists(os.path.join(self.data_dir, "vertebra_mask", filename[:-5] + ".jpg")) else filename[:-5] + ".png"))[..., :1]/255
            
            # centerline
            centerline = label['shapes'][-1]['points'] 
            centerline = self.interploate(centerline)
            centerline[:, 0], centerline[:, 1] = centerline[:, 0] / 256, centerline[:, 1] / 512 

            # heatmap
            heatmap = self.gene_heatmap(centerline)

            # cp & knots
            cp, knots, angle = np.array(label['cp']), np.array(label['knots']), label['angles'] 
            cp_ = cp.copy()
            cp_[:, 0], cp_[:, 1] = cp[:, 1] / 256, cp[:, 0] / 512 
            cp = cp_.copy()
            bspline = np.concatenate([cp.reshape([-1,]), knots[self.k + 1:-(self.k + 1)]], axis=0)

            # angles
            angle = np.array([angle['pt'], angle['mt'], angle['tl']])  

            bsplines.append(bspline) 
            angles.append(angle)
            images.append(image)
            spine_masks.append(spine_mask)
            vertebra_masks.append(vertebra_mask)
            heatmaps.append(heatmap)
            centerlines.append(centerline)
            filenames.append([image_path])

        self.images = torch.from_numpy(np.stack(images)).float()
        self.spine_masks = torch.from_numpy(np.stack(spine_masks)).float()
        self.vertebra_masks = torch.from_numpy(np.stack(vertebra_masks)).float()
        self.heatmaps = torch.from_numpy(np.stack(heatmaps)).float()
        self.centerlines = torch.from_numpy(np.stack(centerlines)).float()
        self.angles = torch.from_numpy(np.stack(angles)).float()/180.*torch.pi
        self.filenames = np.array(filenames)
        self.H, self.W = self.images.shape[1:3]

    def fit_bspline(self, heatmap):

        heatmap = heatmap.reshape([512, 256])
        index = torch.where(heatmap > 0.5)
        centerline = torch.stack([index[1], index[0]], dim=-1).float()
        centerline[:, 0], centerline[:, 1] = centerline[:, 0] / 256, centerline[:, 1] / 512
        idx = torch.linspace(0, len(centerline) - 1, 34, dtype=torch.long)
        centerline = centerline[idx]

        bs = BSpline(self.n, self.k, D=centerline)
        cp, knots = bs.P, bs.U
        bspline = torch.cat([cp.reshape([-1,]), knots[self.k + 1: -(self.k + 1)]], dim=0)
        uq = torch.linspace(0, 1, 34).to(heatmap)
        centerline = bs.forward(uq)

        return centerline, bspline
    def __len__(self):
        return len(self.json_list)

    def add_noise(self, mask):
        import random
        noise_size_x = random.randint(20, 50)
        noise_size_y = random.randint(20, 50)
        x1 = random.randint(0, self.W - noise_size_x)
        y1 = random.randint(0, self.H - noise_size_y)
        x2 = x1 + noise_size_x
        y2 = y1 + noise_size_y
        mask[:, y1:y2, x1:x2] = 1

        return mask

    def gen_random_data(self, index, stage):
        angles = self.angles[index]
        filenames = [self.filenames[i] for i in index]

        image_batch = []
        spine_mask_batch = []
        vertebra_mask_batch = []
        bspline_batch =[]
        centerline_batch = []
        for idx in index:
            data = self.transforms_(torch.cat([self.images[idx], self.spine_masks[idx], self.heatmaps[idx], self.vertebra_masks[idx]], dim=-1).numpy())
            image, spine_mask, heatmap, vertebra_mask = torch.split(data, [3, 1, 1, 1], dim=0)
            if stage == '3':
                spine_mask = self.add_noise(spine_mask)
                vertebra_mask = self.add_noise(vertebra_mask)
                centerline, bspline = self.fit_bspline(heatmap)
                bspline_batch.append(bspline)
                centerline_batch.append(centerline)
            image_batch.append(image)
            spine_mask_batch.append(spine_mask)
            vertebra_mask_batch.append(vertebra_mask)
        images = torch.stack(image_batch, dim=0)
        spine_masks = torch.stack(spine_mask_batch, dim=0)
        vertebra_masks = torch.stack(vertebra_mask_batch, dim=0)

        if stage == '3':
            bsplines = torch.stack(bspline_batch, dim=0)
            centerlines = torch.stack(centerline_batch, dim=0)
        else:
            bsplines = None
            centerlines = None
        return images, spine_masks, vertebra_masks, bsplines, centerlines, angles, filenames


class AASCE2019_Dataset(Dataset):

    def __init__(self, args, split = None, transforms_=transforms.ToTensor()) -> None:
        super(AASCE2019_Dataset, self).__init__()
        self.data_dir = args.data_dir
        self.split = 'test' if args.mode == 'test' or split == 'val' else 'training'
        self.transforms_ = transforms_
        self.n = args.general.n
        self.k = args.general.k

        self.filenames = pd.read_csv(os.path.join(self.data_dir, "labels", self.split, "filenames.csv"), header=None).values
        self.landmarks = pd.read_csv(os.path.join(self.data_dir, "labels", self.split, "landmarks.csv"), header=None).values
        self.angles = torch.from_numpy(pd.read_csv(os.path.join(self.data_dir, "labels", self.split, "angles.csv"), header=None).values).float()/180*torch.pi
        self.loads_images(self.filenames)
        self.loads_centerlines(self.landmarks)

    def loads_images(self, filenames):
        """load images and masks

        Args:
            filenames (_type_): _description_
        """
        print("Loading images.")
        images, masks, vertebra_masks, edge_images = [], [], [], []
        for [filename] in tqdm(filenames):
            img = cv2.imread(os.path.join(self.data_dir, "data", self.split, filename))
            mask = cv2.imread(os.path.join(self.data_dir, "mask" , self.split, "spine_mask", filename))[..., :1]
            vertebra_mask = cv2.imread(os.path.join(self.data_dir, "mask" , self.split, "vertebra_mask", filename))[..., :1]
            images.append(img)
            masks.append(mask)
            vertebra_masks.append(vertebra_mask)
        self.images = torch.from_numpy(np.stack(images, axis=0)).float()/255
        self.masks = torch.from_numpy(np.stack(masks, axis=0)).float()/255
        self.vertebra_masks = torch.from_numpy(np.stack(vertebra_masks, axis=0)).float()/255
        self.H, self.W = self.images.shape[1:3]

    def cal_heatmap(self, centerline):
        """calculate heatmap of centerline for augmentation

        Args:
            centerline (_type_): _description_

        Returns:
            _type_: _description_
        """
        H, W = self.H, self.W
        heatmap = np.zeros([H, W, 1])
        centerline[:, 0], centerline[:, 1] = centerline[:, 0] * (W - 1), centerline[:, 1] * (H - 1)
        for x, y in centerline:
            x, y = min(x, W - 2), min(y, H - 2)
            heatmap[int(y), int(x)] = 255
            heatmap[int(y + 1), int(x)] = 255
            heatmap[int(y), int(x + 1)] = 255
            heatmap[int(y + 1), int(x + 1)] = 255
        heatmap = heatmap.astype(np.uint8)

        return heatmap

    def cal_centerline(self, landmark):
        """claculate the centerline in discrete points from landmarks which represented the keypoints of vertebreas

        Args:
            landmark (_type_): _description_

        Returns:
            _type_: _description_
        """
        x, y = landmark[:68], landmark[68:] 
        x_left, x_right = x[0::2], x[1::2] 
        y_left, y_right = y[0::2], y[1::2]

        left_points = np.stack([x_left, y_left], axis=-1) 
        left_points = left_points[left_points[:, 1].argsort()]
        right_points = np.stack([x_right, y_right], axis=-1)
        right_points = right_points[right_points[:, 1].argsort()] 
        centerline = np.clip((left_points + right_points) / 2, 0, 1) 
        return centerline

    def loads_centerlines(self, landmarks):
        """load centerlines, actually we calculate the centerline in discrete points. In the meanwhile we fit a bspline 
        curve with the centerlines and get params (control points, knots vector). Additionally, we get a heatmap of celterlines 
        which will be used to augment our data. 

        Args:
            landmarks (_type_): _description_
        """
        print("Loading centerlines.")
        centerlines, heatmaps = [], []
        for i, landmark in enumerate(tqdm(landmarks)):
            centerline = self.cal_centerline(landmark)
            heatmap = self.cal_heatmap(centerline)

            centerlines.append(centerline)
            heatmaps.append(heatmap)

        self.centerlines = torch.from_numpy(np.stack(centerlines, axis=0)).float()
        self.heatmaps = torch.from_numpy(np.stack(heatmaps, axis=0)).float()

    def fit_bspline(self, heatmap):
        """ fit a bspline curve of the discrete points in heatmap 

        Args:
            heatmap (_type_): _description_

        Returns:
            _type_: _description_
        """
        H, W = self.H, self.W

        # extract the centerline from heatmap
        heatmap = heatmap.reshape([H, W])
        index = torch.where(heatmap > 0.5)
        centerline = torch.stack([index[1], index[0]], dim=-1).float()
        centerline[:, 0], centerline[:, 1] = centerline[:, 0] / W, centerline[:, 1] / H
        idx = torch.linspace(0, len(centerline) - 1, 34, dtype=torch.long)
        centerline = centerline[idx]

        # get bspline parameters
        bs = BSpline(self.n, self.k, D=centerline)
        cp, knots = bs.P, bs.U
        bspline = torch.cat([cp.reshape([-1,]), knots[self.k + 1:-(self.k + 1)]], axis=0)
        uq = torch.linspace(0, 1, 34).to(heatmap)
        centerline = bs.forward(uq)

        return centerline, bspline

    def __len__(self):
        return len(self.filenames)

    def get_images(self):
        return self.images.permute(0,3,1,2), self.filenames, self.vertebra_masks.permute(0,3,1,2), self.angles #9.19
    
    def add_noise(self, mask):
        import random
        noise_size_x = random.randint(20, 50)
        noise_size_y = random.randint(20, 50)
        x1 = random.randint(0, self.W - noise_size_x)
        y1 = random.randint(0, self.H - noise_size_y)
        x2 = x1 + noise_size_x
        y2 = y1 + noise_size_y
        mask[:, y1:y2, x1:x2] = 1
        
        return mask
    
    def compare_masks_and_heatmaps(self, mask, heatmap_vertebra):
        return torch.equal(mask, heatmap_vertebra)
    
    def gen_random_data(self, index, stage='1'):
        """ generate random data for training, key 'stage' denote the training stage,
        1 for segmentation model, 3 for regression model.

        Args:
            index (tensor): a batch of index
            stage (str, optional): _description_. Defaults to '1'.

        Returns:
            _type_: _description_
        """
        filename = self.filenames[index][:,0]
        angle = self.angles[index]

        image_batch = []
        mask_batch = []
        bspline_batch = []
        centerline_batch = []
        landmarks = self.landmarks[index]
        heatmaps = []
        for idx in index:
            landmark = self.landmarks[idx]
            
            data = self.transforms_(torch.cat([self.images[idx], self.masks[idx], self.heatmaps[idx], self.vertebra_masks[idx]], dim=-1).numpy()) 
            image, mask, heatmap_centerline, heatmap_vertebra = torch.split(data, [3, 1, 1, 1], dim=0) 

            image_batch.append(image)
            mask_batch.append(mask)
            heatmaps.append(heatmap_vertebra)
            
            if stage =='3':
                # heatmap_centerline = self.add_noise(heatmap_centerline)
                # heatmap_vertebra = self.add_noise(heatmap_vertebra)                
                centerline, bspline = self.fit_bspline(heatmap_centerline) 
                bspline_batch.append(bspline)
                centerline_batch.append(centerline)

        image = torch.stack(image_batch, dim=0)
        mask = torch.stack(mask_batch, dim=0)
        heatmap_vertebra = torch.stack(heatmaps, dim=0)
        
        if stage =='3':
            bspline = torch.stack(bspline_batch, dim=0)
            centerline = torch.stack(centerline_batch, dim=0)
        else:
            bspline = None
            centerline = None
        
        return image, mask, heatmap_vertebra, bspline, centerline, angle, filename
