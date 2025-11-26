#from builtins import zip
import os,cv2
from glob import glob
from torchvision import transforms
from torch.nn import functional as F
import torch
import numpy as np
from lib.model.ssformer.build_model import build
from lib.model.ssformer.cvs_build_model import build_cvs
from lib.model.cyclegan import Generator
from lib.utils import get_resnet2,getCobb_bspline
from lib.utils.bspline import BSpline
from lib.utils.draw import plot
from config import *
from config import mode

def postprocess(images,cp_all, knots_all, points_all, angles_all):
    images_drawed=[]
    cobb=[]
    for image,cp,knots,points,angles in zip(images, cp_all, knots_all, points_all, angles_all):
        # 计算Cobb角
        bs = BSpline(9, 3, P=cp.reshape(-1, 2), U=torch.cat([torch.zeros(3 + 1).to(device), knots, torch.ones(3 + 1).to(device)]))
        mt_cobb, pt_cobb, tl_cobb, end_points, ret, grad= getCobb_bspline(9, 3, cp, knots, bs)
        
        ## hybird
        if dataset=="sanyuan":
            alpha, beta, gamma = 0.5, 0.4, 0.7 # sanyuan
        elif dataset=="public":
            alpha, beta, gamma = 0.4, 0.5, 0.5 # public
        mt_cobb2, pt_cobb2, tl_cobb2 = angles * 90
        mt_cobb = mt_cobb * alpha + mt_cobb2 * (1 - alpha)
        pt_cobb = pt_cobb * beta + pt_cobb2 * (1 - beta)
        tl_cobb = tl_cobb * gamma + tl_cobb2 * (1 - gamma)
        cobb.append([torch.float(mt_cobb),torch.float(pt_cobb),torch.float(tl_cobb)])

        # 绘制控制点
        cp = cp.reshape([-1, 2])
        cp[:, 0], cp[:, 1] = cp[:, 0] * 256, cp[:, 1] * 512

        # 绘制中心线
        mask_raw=np.transpose(np.uint8(image.cpu().numpy()*255), [1, 2, 0]).copy()
        # mask = np.transpose(np.uint8(image.cpu().numpy()*255), [1, 2, 0]).copy()
        mask = np.ones([512,256,4],dtype=np.uint8)*255
        mask[...,3]=0
        for j in torch.range(len(ret) - 1):
            pt1 = (torch.int(ret[j][0]), torch.int(ret[j][1]))
            pt2 = (int(ret[j + 1][0]), torch.int(ret[j + 1][1]))
            cv2.line(mask, pt1, pt2, (0, 0, 255, 255), 4)
        
        mask_centerline=mask


        # 控制点折虚线绘制
        mask = np.ones([512,256,4],dtype=np.uint8)*255
        mask[...,3]=0
        # mask=plot(mask,cp.cpu().numpy(),[240,176,0],2,"--",gap=6, alpha=1)
        for x, y in cp:
            cv2.circle(mask, [int(x), int(y)], 3, (240,176,0,255), 5)
        mask_cp=mask

        # 绘制端椎
        mask = np.ones([512,256,4],dtype=np.uint8)*255
        mask[...,3]=0
        ## 定义绘制终板的方法
        def draw_endplates(image, xs, ys, k, l=50):
            dx=torch.sqrt(l*l/(4*k*k+4))
            x0=xs-dx
            x1=xs+dx
            y0=k*(x0-xs)+ys
            y1=k*(x1-xs)+ys

            image=cv2.circle(image, (int(ys), int(xs)), 2, (0, 255, 0, 255), 4)
            image=cv2.line(image, (int(y0),int(x0)), (int(y1),int(x1)), (0, 255, 0, 255), 2)

            if x0<x1:
                line=[[int(y0),int(x0)],[int(y1),int(x1)]]
            else:
                line=[[int(y1),int(x1)],[int(y0),int(x0)]]

            return image, line

        ## 绘制端椎和终板
        lines=[]
        for i in end_points:
            x,y=ret[i, 0],ret[i,1]
            mask,line=draw_endplates(mask,y,x,-1/grad[i],l=70)
            lines.append(line)
        mask_endplate=mask


        # 绘制Cobb角度在图像上
        mask = np.ones([512,256,4],dtype=np.uint8)*255
        mask[...,3]=0
        def midPoint(p0,p1):
            return [int((p0[0]+p1[0])/2)-10,int((p0[1]+p1[1])/2)]
        
        cv2.putText(mask, f"{float(pt_cobb):.2f}", midPoint(lines[0][1],lines[1][0]), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 250, 0, 255), 1)
        cv2.putText(mask, f"{float(mt_cobb):.2f}", midPoint(lines[1][1],lines[2][0]), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 250, 0, 255), 1)
        cv2.putText(mask, f"{float(tl_cobb):.2f}", midPoint(lines[2][1],lines[3][0]), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 250, 0, 255), 1)
        mask_cobb=mask
        
        # aaa=mask_centerline | mask_cobb | mask_cp | mask_endplate | mask_raw
        images_drawed.append([mask_raw,mask_centerline,mask_cp,mask_endplate,mask_cobb])
    
    return images_drawed,cobb

def inference(images,model_seg,model_refine,model_reg):
    ret_dict={}
    with torch.no_grad():
        mask1=model_seg(images.to(device))
        aa=mask1[1]
        if mode == "bspine":
            mask2=model_refine(mask1)
        elif mode == "cvspine":
            mask1 = mask1[1]
            mask2=model_refine(mask1)
        aaa=mask2[1]
        points_all, bspline_all, angles_all=model_reg(mask2)
        
        cp_all, knots_all = torch.split(bspline_all, [2 * (9 + 1), 9 - 3], dim=-1)
        images_drawed,cobb=postprocess(images,cp_all, knots_all, points_all, angles_all)

        ret_dict['cp']=cp_all
        ret_dict['knots']=knots_all
        ret_dict['points']=points_all
        ret_dict['cobb']=cobb
        ret_dict['images_drawed']=images_drawed
    return ret_dict

def main(infer_dir,ckpt_seg,ckpt_refine,ckpt_reg):
    # inference images
    images=[]
    totensor=transforms.ToTensor()
    for filename in glob(os.path.join(infer_dir,"*")):
        image=totensor(cv2.imread(filename))
        images.append(image)
    images=torch.stack(images,dim=0)

    # load models
    if mode == "bspine":
        # Load the BSPine model checkpoints
        model_seg = build("mit_PLD_b2", class_num=1).to(device)
        model_seg.load_state_dict(torch.load(ckpt_seg, map_location=device))
        
        model_refine = Generator(1, 1).to(device)
        model_refine.load_state_dict(torch.load(ckpt_refine, map_location=device))
        
        model_reg = get_resnet2(1, 2 * (9 + 1) + 9 - 3).to(device)
        model_reg.load_state_dict(torch.load(ckpt_reg, map_location=device))
        
        # The rest of your code for BSPine
        print("Using B-SPine model...")

    elif mode == "cvspine":
        # Load the CVSpline model checkpoints
        model_seg = build_cvs("mit_PLD_b2_cvs", class_num=5).to(device)
        model_seg.load_state_dict(torch.load(cvs_ckpt_seg, map_location=device))
        
        model_refine = Generator(1, 1).to(device)
        model_refine.load_state_dict(torch.load(cvs_ckpt_refine, map_location=device))
        
        model_reg = get_resnet2(1, 2 * (9 + 1) + 9 - 3).to(device)
        model_reg.load_state_dict(torch.load(cvs_ckpt_reg, map_location=device))
        
        # The rest of your code for CVSpine
        print("Using CVS-Spine model...")
    else:
        raise ValueError("Invalid mode selected. Choose either 'bspine' or 'cvspine'.")

    # get return
    ret=inference(images,model_seg,model_refine,model_reg)
    print(ret['cobb'])

if __name__=="__main__":
    infer_dir="data/infer_data"
    if mode == "bspine":
        main(infer_dir,ckpt_seg,ckpt_refine,ckpt_reg)
    elif mode == "cvspine":
        main(infer_dir,cvs_ckpt_seg,cvs_ckpt_refine,cvs_ckpt_reg)