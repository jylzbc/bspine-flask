import os
import json
from glob import glob
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from lib.model.ssformer.build_model import build
from lib.model.ssformer.cvs_build_model import build_cvs
from lib.model.cyclegan import Generator
from lib.utils import get_resnet2, getCobb_bspline
from lib.utils.bspline import BSpline
from lib.utils.draw import plot
from config import *
from config import mode  

def preprocess(data):
    data = np.array(data)
    h, w, c = data.shape
    if c == 4:
        data = data[..., :3]
        c = 3
    ratio = w / h
    if ratio > 0.5:
        new_w = w
        new_h = 2 * w
        image = np.zeros((new_h, new_w, c), dtype=np.uint8)
        image[(new_h - h) // 2:(new_h - h) // 2 + h, ...] = data
    elif ratio < 0.5:
        new_h = h
        new_w = int(h * 0.5)
        image = np.zeros((new_h, new_w, c), dtype=np.uint8)
        image[:, (new_w - w) // 2:(new_w - w) // 2 + w, ...] = data
    else:
        image = data

    transforms_ = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize([512, 256])
    ])
    image = transforms_(image)
    return image.unsqueeze(0)  #  [1, C, H, W]

def postprocess(images, cp_all, knots_all, points_all, angles_all):
    images_drawed = []
    cobb = []
    all_end_points = []
    all_grad = []
    for image, cp, knots, points, angles in zip(images, cp_all, knots_all, points_all, angles_all):

        bs = BSpline(9, 3, P=cp.reshape(-1, 2),
                     U=torch.cat([torch.zeros(3 + 1).to(device), knots, torch.ones(3 + 1).to(device)]))
        mt_cobb, pt_cobb, tl_cobb, end_points, ret, grad = getCobb_bspline(9, 3, cp, knots, bs)
        
        if dataset == "sanyuan":
            # alpha, beta, gamma = 0.5, 0.4, 0.7  # sanyuan-bspine
            alpha, beta, gamma = 0.6, 0.7, 0.5 # sanyuan-cvspine
        elif dataset == "public":
            # alpha, beta, gamma = 0.4, 0.5, 0.5  # public-bspine
            alpha, beta, gamma = 0.6, 0.5, 0.6 # public-cvspine
        mt_cobb2, pt_cobb2, tl_cobb2 = angles * 90
        mt_cobb = mt_cobb * alpha + mt_cobb2 * (1 - alpha)
        pt_cobb = pt_cobb * beta + pt_cobb2 * (1 - beta)
        tl_cobb = tl_cobb * gamma + tl_cobb2 * (1 - gamma)
        cobb.append([float(mt_cobb), float(pt_cobb), float(tl_cobb)])
        
        cp = cp.reshape([-1, 2])
        cp[:, 0], cp[:, 1] = cp[:, 0] * 256, cp[:, 1] * 512
        
        mask_raw = np.transpose(np.uint8(image.cpu().numpy() * 255), [1, 2, 0]).copy()
        mask = np.ones([512, 256, 4], dtype=np.uint8) * 255
        mask[..., 3] = 0
        for j in range(len(ret) - 1):
            pt1 = (int(ret[j][0]), int(ret[j][1]))
            pt2 = (int(ret[j + 1][0]), int(ret[j + 1][1]))
            cv2.line(mask, pt1, pt2, (0, 0, 255, 255), 4)
        mask_centerline = mask
        
        mask = np.ones([512, 256, 4], dtype=np.uint8) * 255
        mask[..., 3] = 0
        for x, y in cp:
            cv2.circle(mask, (int(x), int(y)), 3, (240, 176, 0, 255), 5)
        mask_cp = mask
        
        mask = np.ones([512, 256, 4], dtype=np.uint8) * 255
        mask[..., 3] = 0
        def draw_endplates(image, xs, ys, k, l=50):
            dx = torch.sqrt(l * l / (4 * k * k + 4))
            x0 = xs - dx
            x1 = xs + dx
            y0 = k * (x0 - xs) + ys
            y1 = k * (x1 - xs) + ys
            image = cv2.circle(image, (int(ys), int(xs)), 2, (0, 255, 0, 255), 4)
            image = cv2.line(image, (int(y0), int(x0)), (int(y1), int(x1)), (0, 255, 0, 255), 2)
            if x0 < x1:
                line = [[int(y0), int(x0)], [int(y1), int(x1)]]
            else:
                line = [[int(y1), int(x1)], [int(y0), int(x0)]]
            return image, line

        lines = []
        for i in end_points:
            x, y = ret[i, 0], ret[i, 1]
            mask, line = draw_endplates(mask, y, x, -1 / grad[i], l=70)
            lines.append(line)
        mask_endplate = mask
        
        mask = np.ones([512, 256, 4], dtype=np.uint8) * 255
        mask[..., 3] = 0
        def midPoint(p0, p1):
            return [int((p0[0] + p1[0]) / 2) - 10, int((p0[1] + p1[1]) / 2)]
        cv2.putText(mask, f"{float(pt_cobb):.2f}", midPoint(lines[0][1], lines[1][0]),
                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 250, 0, 255), 1)
        cv2.putText(mask, f"{float(mt_cobb):.2f}", midPoint(lines[1][1], lines[2][0]),
                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 250, 0, 255), 1)
        cv2.putText(mask, f"{float(tl_cobb):.2f}", midPoint(lines[2][1], lines[3][0]),
                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 250, 0, 255), 1)
        mask_cobb = mask
        
        images_drawed.append([mask_raw, mask_centerline, mask_cp, mask_endplate, mask_cobb])
        all_end_points.append([item.cpu().tolist() if torch.is_tensor(item) else item for item in end_points])
        all_grad.append(grad.cpu().numpy().tolist())
    return images_drawed, cobb, all_end_points, all_grad

def inference(images, model_seg, model_refine, model_reg):
    ret_dict = {}
    with torch.no_grad():
        mask1 = model_seg(images.to(device))
        if mode == "bspine":
            mask2 = model_refine(mask1)
        elif mode == "cvspine":
            mask1 = mask1[1]
            mask2 = model_refine(mask1)
        points_all, bspline_all, angles_all = model_reg(mask2)
        cp_all, knots_all = torch.split(bspline_all, [2 * (9 + 1), 9 - 3], dim=-1)
        
        _, cobb, end_points, grad = postprocess(images, cp_all, knots_all, points_all, angles_all)
        ret_dict['cp'] = cp_all.cpu().numpy().tolist()
        ret_dict['knots'] = knots_all.cpu().numpy().tolist()
        ret_dict['points'] = points_all.cpu().numpy().tolist()
        ret_dict['end_points'] = end_points
        ret_dict['derivative'] = grad
        ret_dict['cobb'] = cobb
    return ret_dict

def main(infer_dir, ckpt_seg, ckpt_refine, ckpt_reg):
    image_files = glob(os.path.join(infer_dir, "*"))
    image_files = sorted(image_files, key=lambda x: os.path.basename(x).lower())
    if mode == "bspine":
        model_seg = build("mit_PLD_b2", class_num=1).to(device)
        model_seg.load_state_dict(torch.load(ckpt_seg, map_location=device))
        
        model_refine = Generator(1, 1).to(device)
        model_refine.load_state_dict(torch.load(ckpt_refine, map_location=device))
        
        model_reg = get_resnet2(1, 2 * (9 + 1) + 9 - 3).to(device)
        model_reg.load_state_dict(torch.load(ckpt_reg, map_location=device))
        
        subfolder = "bspine_re"
    elif mode == "cvspine":
        model_seg = build_cvs("mit_PLD_b2_cvs", class_num=5).to(device)
        model_seg.load_state_dict(torch.load(cvs_ckpt_seg, map_location=device))
        
        model_refine = Generator(1, 1).to(device)
        model_refine.load_state_dict(torch.load(cvs_ckpt_refine, map_location=device))
        
        model_reg = get_resnet2(1, 2 * (9 + 1) + 9 - 3).to(device)
        model_reg.load_state_dict(torch.load(cvs_ckpt_reg, map_location=device))
        
        subfolder = "cvspine_re"
        print("Using CVS-Spine model...")
    else:
        raise ValueError("Invalid mode selected. Choose either 'bspine' or 'cvspine'.")

    save_path_base = "./B_CVS_Spine_result"
    json_dir = os.path.join(save_path_base, subfolder)
    if not os.path.exists(json_dir):
        os.makedirs(json_dir)

    count = 0
    total = len(image_files)

    for filepath in image_files:
        base_filename = os.path.splitext(os.path.basename(filepath))[0]
        try:
            img = Image.open(filepath).convert("RGB")
        except Exception as e:
            print(f"Unable to open image {filepath}: {e}")
            continue

        img_tensor = preprocess(img)
        ret = inference(img_tensor, model_seg, model_refine, model_reg)
        json_path = os.path.join(json_dir, f"{base_filename}.json")
        try:
            with open(json_path, 'w') as f:
                json.dump(ret, f, indent=4, ensure_ascii=False)
            count += 1
            print(f"The NO.{count} JSON file save succeed: {json_path}  (Total: {total})")
        except Exception as e:
            print(f"Failed to save JSON file for {base_filename}: {e}")

if __name__ == "__main__":
    infer_dir = "data/images" 
    if mode == "bspine":
        main(infer_dir, ckpt_seg, ckpt_refine, ckpt_reg)
    elif mode == "cvspine":
        main(infer_dir, cvs_ckpt_seg, cvs_ckpt_refine, cvs_ckpt_reg)
