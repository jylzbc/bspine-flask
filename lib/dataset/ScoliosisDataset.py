import os
import cv2
import json
import numpy as np
import pandas as pd
from glob import glob

import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms
from scipy.interpolate import interp1d
from utils.bspline import BSpline_np,BSpline


class ScoliosisDataset_Sanyuan(Dataset):
    def __init__(self,data_dir,split="train",transforms_=transforms.ToTensor(), n=9, k=3):
        super(ScoliosisDataset_Sanyuan,self).__init__()
        self.split=split
        self.n=n
        self.k=k
        self.data_dir=os.path.join(data_dir,split)
        self.json_list=sorted(glob(os.path.join(self.data_dir,"label","*.json")))
        self.transforms=transforms_
        self.load_data(self.json_list)
    

    # 从17个centerline中插值出34个
    def interploate(self,centerline):
        centerline=np.array(centerline)
        from scipy.interpolate import interp1d
        f=interp1d(centerline[:,1],centerline[:,0],kind="cubic")
        x=np.linspace(centerline[:,1][0],centerline[:,1][-1],34)
        y=f(x)
        centerline=np.stack([y,x],axis=-1)
        return centerline


    # 生成heatmap，以数据增强
    def gene_heatmap(self,centerline):
        heatmap=np.zeros([512,256,1])
        centerline[:,0],centerline[:,1]=centerline[:,0]*255,centerline[:,1]*511
        for x,y in centerline:
            x,y=min(x,256-2), min(y,512-2)
            heatmap[int(y),int(x)]=255
            heatmap[int(y+1),int(x)]=255
            heatmap[int(y),int(x+1)]=255
            heatmap[int(y+1),int(x+1)]=255
        heatmap=heatmap.astype(np.uint8)
        return heatmap


    # 加载数据，images,masks,heatmaps,cp&bspline,angles
    def load_data(self,json_list):
        images,masks,heatmaps,centerlines,bsplines,angles,filenames=[],[],[],[],[],[],[]
        for jsonfile in json_list:
            prefix,filename=os.path.split(jsonfile)
            # image
            image_file=os.path.join(self.data_dir,"data",filename[:-5] + ".jpg" if os.path.exists(os.path.join(self.data_dir,"data",filename[:-5]+".jpg")) else filename[:-5] + ".png")
            image=cv2.imread(image_file)
            # mask
            mask=cv2.imread(os.path.join(self.data_dir,"mask",filename[:-5] + ".jpg" if os.path.exists(os.path.join(self.data_dir,"mask",filename[:-5]+".jpg")) else filename[:-5] + ".png"))[...,:1]
            # centerline
            with open(jsonfile,"r") as f:
                label=json.loads(f.read())
            centerline=label['shapes'][-1]['points']
            centerline=self.interploate(centerline)
            centerline[:,0],centerline[:,1]=centerline[:,0]/256,centerline[:,1]/512
            # heatmap
            heatmap=self.gene_heatmap(centerline)
            # cp & knots
            cp,knots,angle=np.array(label['cp']),np.array(label['knots']),label['angles']
            cp_=cp.copy() 
            cp_[:,0],cp_[:,1]=cp[:,1]/256,cp[:,0]/512
            cp=cp_.copy()
            bspline=np.concatenate([cp.reshape([-1,]),knots[self.k+1:-(self.k+1)]],axis=0)
            # angles
            angle=np.array([angle['pt'],angle['mt'],angle['tl']]) # 注意三者顺序

            bsplines.append(bspline)
            angles.append(angle)
            images.append(image)
            masks.append(mask)
            heatmaps.append(heatmap)
            centerlines.append(centerline)
            filenames.append(filename)

        self.images=np.stack(images)
        self.masks=np.stack(masks)
        self.heatmaps=np.stack(heatmaps)
        self.centerlines=np.stack(centerlines)
        self.angles=np.stack(angles)
        self.bsplines=np.stack(bsplines)
        self.filenames=filenames


    # 用于数据增强
    def update_curve(self,heatmap):
        # 从heatmap中提取中心线
        heatmap=heatmap.reshape([512,256])
        index=torch.where(heatmap>0.5)
        centerline=torch.stack([index[1],index[0]],dim=-1).float()
        centerline[:,0],centerline[:,1]=centerline[:,0]/256,centerline[:,1]/512
        idx=torch.linspace(0,len(centerline)-1,34,dtype=torch.long)
        centerline=centerline[idx]
        
        # mask=np.zeros([512,256],dtype=np.uint8)
        # for x,y in centerline:
        #     mask[int(y*512),int(x*256)]=255
        # bspline parameters
        bs = BSpline_np(self.n, self.k, D=centerline.numpy())
        cp, knots=bs.P, bs.U
        bspline=np.concatenate([cp.reshape([-1,]),knots[self.k+1:-(self.k+1)]],axis=0).astype(np.float32)    
        uq=np.linspace(0,1,34)
        ret=bs.run(uq)
        centerline=torch.from_numpy(ret).float()

        return centerline, bspline


    def __len__(self):
        return len(self.json_list)
    

    def __getitem__(self, index):
        image=self.images[index].astype(np.uint8)
        mask=self.masks[index].astype(np.uint8)
        heatmap=self.heatmaps[index].astype(np.uint8)
        bspline=self.bsplines[index].astype(np.float32)
        centerline=self.centerlines[index].astype(np.float32)
        angle=self.angles[index].astype(np.float32)
        filename=self.filenames[index]

        data=self.transforms(np.concatenate([image,mask,heatmap],axis=-1))
        image,mask,heatmap=torch.split(data,[3,1,1],dim=0)
        centerline,bspline=self.update_curve(heatmap)

        return image,mask,bspline,centerline,angle,filename


class ScoliosisDataset_Public(Dataset):
    def __init__(self, data_dir, transforms_=transforms.ToTensor(), split="training", n=9, k=3) -> None:
        super(ScoliosisDataset_Public, self).__init__()
        self.data_dir = data_dir
        self.split = split
        self.transforms_=transforms_
        self.n = n
        self.k = k
        
        self.filenames = pd.read_csv(os.path.join(data_dir, "labels", self.split, "filenames.csv"), header=None).values
        self.landmarks = pd.read_csv(os.path.join(data_dir, "labels", self.split, "landmarks.csv"), header=None).values
        self.angles = pd.read_csv(os.path.join(data_dir, "labels", self.split, "angles.csv"), header=None).values
        self.loads_images(self.filenames)
        self.loads_centerlines(self.landmarks)


    def loads_images(self, filenames):
        images,masks=[],[]
        for [filename] in filenames:
            img = cv2.imread(os.path.join(self.data_dir, "data", self.split, filename))
            mask = cv2.imread(os.path.join(self.data_dir, "mask", self.split, filename))[..., :1]
            images.append(img)
            masks.append(mask)
        self.images=np.stack(images,axis=0)
        self.masks=np.stack(masks,axis=0)


    def gene_heatmap(self,centerline):
        heatmap=np.zeros([512,256,1])
        centerline[:,0],centerline[:,1]=centerline[:,0]*255,centerline[:,1]*511

        ## bspline train
        for x,y in centerline:
            x,y=min(x,256-2), min(y,512-2)
            heatmap[int(y),int(x)]=255
            heatmap[int(y+1),int(x)]=255
            heatmap[int(y),int(x+1)]=255
            heatmap[int(y+1),int(x+1)]=255
        heatmap=heatmap.astype(np.uint8)
        return heatmap


    def cal_centerline(self, landmark):
        x, y = landmark[:68], landmark[68:]
        x_left, x_right = x[0::2], x[1::2]
        y_left, y_right = y[0::2], y[1::2]

        left_points=np.stack([x_left,y_left],axis=-1)
        left_points=left_points[left_points[:,1].argsort()]
        right_points=np.stack([x_right,y_right],axis=-1)
        right_points=right_points[right_points[:,1].argsort()]
        centerline=np.clip((left_points+right_points)/2,0,1)
        return centerline


    def fit_BSpline(self, centerline):
        centerline=torch.from_numpy(centerline).cuda()
        bs = BSpline(self.n, self.k, D=centerline)
        cp, knots=bs.get_P_U()
        heatmap=self.gene_heatmap(centerline.clone())
        return cp, knots, heatmap


    def loads_centerlines(self,landmarks):
        centerlines,bsplines,heatmaps=[],[],[]
        for i,landmark in enumerate(landmarks):
            centerline = self.cal_centerline(landmark)
            cp, knots, heatmap= self.fit_BSpline(centerline)
            bspline=torch.cat([cp.reshape([-1,]),knots[self.k+1:-(self.k+1)]],dim=0).cpu().numpy()
            centerlines.append(centerline)
            bsplines.append(bspline)
            heatmaps.append(heatmap)

        self.centerlines=np.stack(centerlines,axis=0)
        self.bsplines=np.stack(bsplines,axis=0)
        self.heatmaps=np.stack(heatmaps,axis=0)
        

    def update_curve(self,heatmap):
        # 从heatmap中提取中心线
        heatmap=heatmap.reshape([512,256])
        index=torch.where(heatmap>0.5)
        centerline=torch.stack([index[1],index[0]],dim=-1).float()
        centerline[:,0],centerline[:,1]=centerline[:,0]/256,centerline[:,1]/512
        idx=torch.linspace(0,len(centerline)-1,34,dtype=torch.long)
        centerline=centerline[idx]
        
        # mask=np.zeros([512,256],dtype=np.uint8)
        # for x,y in centerline:
        #     mask[int(y*512),int(x*256)]=255
        # bspline parameters
        bs = BSpline_np(self.n, self.k, D=centerline.numpy())
        cp, knots=bs.P, bs.U
        bspline=np.concatenate([cp.reshape([-1,]),knots[self.k+1:-(self.k+1)]],axis=0).astype(np.float32)    
        uq=np.linspace(0,1,34)
        ret=bs.run(uq)
        centerline=torch.from_numpy(ret).float()

        return centerline, bspline


    def __len__(self):
        return len(self.filenames)


    def __getitem__(self, index):
        filename = self.filenames[index][0]
        angle = self.angles[index].astype(np.float32)
        bspline = self.bsplines[index].astype(np.float32)
        centerline = self.centerlines[index].astype(np.float32)

        data=self.transforms_(np.concatenate([self.images[index],self.masks[index],self.heatmaps[index]],axis=-1))
        image,mask,heatmap=torch.split(data,[3,1,1],dim=0)
        centerline,bspline=self.update_curve(heatmap)

        return image,mask,bspline,centerline,angle,filename