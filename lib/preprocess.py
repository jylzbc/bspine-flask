import os
import numpy as np
import pandas as pd
import cv2
from scipy.interpolate import interp1d
from tqdm import tqdm

def resize(img,landmark,hw=[512,256]):
    h,w,d=img.shape
    x,y=landmark[:68],landmark[68:]
    
    if w < h/2:
        margin=(h/2-w)//2
        img=cv2.copyMakeBorder(img,0,0,int(margin),int(margin),cv2.BORDER_CONSTANT)
        hp,wp,d=img.shape
        xp=(x*w+margin)/wp
        yp=y
    else:
        margin=(w*2-h)//2
        img=cv2.copyMakeBorder(img,int(margin),int(margin),0,0,cv2.BORDER_CONSTANT)
        hp,wp,d=img.shape
        yp=(y*h+margin)/hp
        xp=x

    img=cv2.resize(img,[hw[1],hw[0]])
    landmark=np.concatenate([xp,yp])
    return img,landmark

def interplote(x,y,N):
    f=interp1d(y,x,kind="quadratic")
    ynew=np.linspace(np.min(y),np.max(y),N)
    xnew=f(ynew)

    return xnew,ynew

def smooth(x,y,M=5):
    length=len(x)
    for i in range(length):
        x[i]=np.mean(x[max(i-M,0):min(i+M,length-1)])
        y[i]=np.mean(y[max(i-M,0):min(i+M,length-1)])
    return x,y

def gene_mask(landmark,hw):
    x_np,y_np=(landmark[:68]*hw[1]).astype(np.int64),(landmark[68:]*hw[0]).astype(np.int64)
    x_np_left,x_np_right=x_np[0::2],x_np[1::2]
    y_np_left,y_np_right=y_np[0::2],y_np[1::2]

    points_left=np.stack([x_np_left,y_np_left],axis=-1)
    points_left = points_left[points_left[:,1].argsort()]
    points_right=np.stack([x_np_right,y_np_right],axis=-1)
    points_right = points_right[points_right[:,1].argsort()][::-1]
    polygon=np.concatenate([points_left,points_right],axis=0).astype(np.int32).reshape([-1,1,2])

    mask=np.zeros([hw[0],hw[1]],dtype=np.uint8)
    mask=cv2.polylines(mask,[polygon],1,255)
    mask=cv2.fillPoly(mask,[polygon],255)
    
    return mask

def resize_img_gene_mask(hw,img_dir,label_dir,resize_img_dir,resize_label_dir,mask_dir):
    os.makedirs(resize_img_dir,exist_ok=True)
    os.makedirs(resize_label_dir,exist_ok=True)
    os.makedirs(mask_dir,exist_ok=True)
    
    filenames=pd.read_csv(os.path.join(label_dir,"filenames.csv"),header=None).values
    landmarks=pd.read_csv(os.path.join(label_dir,"landmarks.csv"),header=None).values
    angles=pd.read_csv(os.path.join(label_dir,"angles.csv"),header=None).values
    
    pbar=tqdm(total=len(filenames))
    for i,filename in enumerate(filenames):
        img=cv2.imread(os.path.join(img_dir,filename[0]))
        landmark=landmarks[i]
        angle=angles[i]

        resized_img,resized_landmark=resize(img,landmark,hw)
        mask=gene_mask(resized_landmark,hw)
        cv2.imwrite(os.path.join(resize_img_dir,filename[0]),resized_img)
        landmarks[i]=resized_landmark
        angles[i]=angle
        cv2.imwrite(os.path.join(mask_dir,filename[0]),mask[...,None])
        pbar.update(1)
    
    pd.DataFrame(landmarks).to_csv(os.path.join(resize_label_dir,"landmarks.csv"),header=False,index=False)
    pd.DataFrame(angles).to_csv(os.path.join(resize_label_dir,"angles.csv"),header=False,index=False)
    pd.DataFrame(filenames).to_csv(os.path.join(resize_label_dir,"filenames.csv"),header=False,index=False)

if __name__=="__main__":
    ### 处理AASCE公开数据集
    data_root="data/boostnet_labeldata"
    train_img_dir=os.path.join(data_root,"data","training")
    test_img_dir=os.path.join(data_root,"data","test")
    # val_img_dir=os.path.join(data_root,"data","val")
    train_label_dir=os.path.join(data_root,"labels","training")
    test_label_dir=os.path.join(data_root,"labels","test")
    # val_label_dir=os.path.join(data_root,"labels","val")

    resize_root="data/public_data/"
    resize_train_img_dir=os.path.join(resize_root,"training","data")
    resize_test_img_dir=os.path.join(resize_root,"test","data")
    # resize_val_img_dir=os.path.join(resize_root,"val","data")
    
    resize_train_label_dir=os.path.join(resize_root,"training","label")
    resize_test_label_dir=os.path.join(resize_root,"test","label")
    # resize_val_label_dir=os.path.join(resize_root,"val","label")

    mask_train_dir=os.path.join(resize_root,"training","mask")
    mask_test_dir=os.path.join(resize_root,"test","mask")
    # mask_val_dir=os.path.join(resize_root,"val","mask")
    
    resize_img_gene_mask([512,256],train_img_dir,train_label_dir,resize_train_img_dir,resize_train_label_dir,mask_train_dir)
    resize_img_gene_mask([512,256],test_img_dir,test_label_dir,resize_test_img_dir,resize_test_label_dir,mask_test_dir)
    # resize_img_gene_mask([512,256],val_img_dir,val_label_dir,resize_val_img_dir,resize_val_label_dir,mask_val_dir)


    ### 处理AASCE公开数据集
