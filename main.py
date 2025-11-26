import torch
import base64
import cv2
import os
import uuid
import shutil
import logging
import json
from torchvision import transforms
import numpy as np
from PIL import Image
from gevent import pywsgi
from flask import Flask, request, render_template, send_file, Response, jsonify, make_response
from werkzeug.utils import secure_filename  
from lib.model.ssformer.build_model import build
from lib.model.ssformer.cvs_build_model import build_cvs
from lib.model.cyclegan import Generator
from lib.utils import get_resnet2, getCobb_bspline
from lib.utils.bspline import BSpline
from lib.utils.draw import plot
from config import *
from config import mode as default_mode

app = Flask(__name__)

if mode == "bspine":
    # Load the BSPine model checkpoints
    model_seg = build("mit_PLD_b2", class_num=1).to(device)
    model_seg.load_state_dict(torch.load(ckpt_seg, map_location=device))
    
    model_refine = Generator(1, 1).to(device)
    model_refine.load_state_dict(torch.load(ckpt_refine, map_location=device))
    
    model_reg = get_resnet2(1, 2 * (9 + 1) + 9 - 3).to(device)
    model_reg.load_state_dict(torch.load(ckpt_reg, map_location=device))

elif mode == "cvspine":
    # Load the CVSpline model checkpoints
    cvs_model_seg = build_cvs("mit_PLD_b2_cvs", class_num=5).to(device)
    cvs_model_seg.load_state_dict(torch.load(cvs_ckpt_seg, map_location=device))
    
    cvs_model_refine = Generator(1, 1).to(device)
    cvs_model_refine.load_state_dict(torch.load(cvs_ckpt_refine, map_location=device))
    
    cvs_model_reg = get_resnet2(1, 2 * (9 + 1) + 9 - 3).to(device)
    cvs_model_reg.load_state_dict(torch.load(cvs_ckpt_reg, map_location=device))

else:
    raise ValueError("Invalid mode selected. Choose either 'bspine' or 'cvspine'.")

# data preprocess
def preprocess(data):
    print("preprocess")
    data = np.array(data)
    
    h, w, c = data.shape
    if c == 4:
        data = data[..., :3]
        c = 3
    
    ratio = w/h
    if ratio > 0.5:
        new_w = w
        new_h = 2*w
        image = np.zeros((new_h, new_w, c), dtype=np.uint8)
        image[(new_h-h)//2:(new_h-h)//2+h, ...] = data
    elif ratio < 0.5:
        new_h = h
        new_w = int(h*0.5)
        image = np.zeros((new_h, new_w, c), dtype=np.uint8)
        image[:, (new_w-w)//2:(new_w-w)//2+w, ...] = data
    else:
        image = data

    transforms_ = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize([512, 256])
    ])
    image = transforms_(image)
    return image.unsqueeze(0)

def postprocess(images, cp_all, knots_all, points_all, angles_all):
    """postprocess for return data package with drawed images including drawing control points,
    centerline points, endplates, and cobb angles.
    Args:
        images (_type_): _description_
        cp_all (_type_): _description_
        knots_all (_type_): _description_
        points_all (_type_): _description_
        angles_all (_type_): _description_
    Returns:
        _type_: _description_
    """
    print("postprocess")
    images_drawed = []
    cobb = []
    for image, cp, knots, points, angles in zip(images, cp_all, knots_all, points_all, angles_all):
        
        bs = BSpline(9, 3, P=cp.reshape(-1, 2), U=torch.cat(
            [torch.zeros(3 + 1).to(device), knots, torch.ones(3 + 1).to(device)]))
        mt_cobb, pt_cobb, tl_cobb, end_points, ret, grad = getCobb_bspline(
            9, 3, cp, knots, bs)
        
        if dataset == "sanyuan":
            alpha, beta, gamma = 0.5, 0.4, 0.7  
        elif dataset == "public":
            alpha, beta, gamma = 0.4, 0.5, 0.5  
        mt_cobb2, pt_cobb2, tl_cobb2 = angles * 90
        mt_cobb = mt_cobb * alpha + mt_cobb2 * (1 - alpha)
        pt_cobb = pt_cobb * beta + pt_cobb2 * (1 - beta)
        tl_cobb = tl_cobb * gamma + tl_cobb2 * (1 - gamma)
        cobb.append([float(mt_cobb), float(pt_cobb), float(tl_cobb)])
        
        cp = cp.reshape([-1, 2])
        cp[:, 0], cp[:, 1] = cp[:, 0] * 256, cp[:, 1] * 512
        
        mask_raw = np.transpose(
            np.uint8(image.cpu().numpy()*255), [1, 2, 0]).copy()
        mask = np.ones([512, 256, 4], dtype=np.uint8)*255
        mask[..., 3] = 0
        for j in range(len(ret) - 1):
            pt1 = (int(ret[j][0]), int(ret[j][1]))
            pt2 = (int(ret[j + 1][0]), int(ret[j + 1][1]))
            cv2.line(mask, pt1, pt2, (0, 0, 255, 255), 4)
        mask_centerline = mask
        
        mask = np.ones([512, 256, 4], dtype=np.uint8)*255
        mask[..., 3] = 0
        for x, y in cp:
            cv2.circle(mask, [int(x), int(y)], 3, (240, 176, 0, 255), 5)
        mask_cp = mask
        
        mask = np.ones([512, 256, 4], dtype=np.uint8)*255
        mask[..., 3] = 0

        def draw_endplates(image, xs, ys, k, l=50):
            dx = torch.sqrt(l*l/(4*k*k+4))
            x0 = xs-dx
            x1 = xs+dx
            y0 = k*(x0-xs)+ys
            y1 = k*(x1-xs)+ys

            image = cv2.circle(image, (int(ys), int(xs)),
                               2, (0, 255, 0, 255), 4)
            image = cv2.line(image, (int(y0), int(x0)),
                             (int(y1), int(x1)), (0, 255, 0, 255), 2)

            if x0 < x1:
                line = [[int(y0), int(x0)], [int(y1), int(x1)]]
            else:
                line = [[int(y1), int(x1)], [int(y0), int(x0)]]

            return image, line

        lines = []
        for i in end_points:
            x, y = ret[i, 0], ret[i, 1]
            mask, line = draw_endplates(mask, y, x, -1/grad[i], l=70)
            lines.append(line)
        mask_endplate = mask
        
        mask = np.ones([512, 256, 4], dtype=np.uint8)*255
        mask[..., 3] = 0
        def midPoint(p0, p1):
            return [int((p0[0]+p1[0])/2)-10, int((p0[1]+p1[1])/2)]
        cv2.putText(mask, f"{float(pt_cobb):.2f}", midPoint(
            lines[0][1], lines[1][0]), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 250, 0, 255), 1)
        cv2.putText(mask, f"{float(mt_cobb):.2f}", midPoint(
            lines[1][1], lines[2][0]), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 250, 0, 255), 1)
        cv2.putText(mask, f"{float(tl_cobb):.2f}", midPoint(
            lines[2][1], lines[3][0]), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 250, 0, 255), 1)
        mask_cobb = mask
        
        images_drawed.append(
            [mask_raw, mask_centerline, mask_cp, mask_endplate, mask_cobb])
        
        end_points = [item.cpu().tolist() for item in end_points]
    return images_drawed, cobb, end_points, grad

def postprocess_no_images(cp_all, knots_all, points_all, angles_all):
    """postprocess for return data package without drawed images
    Args:
        cp_all (_type_): _description_
        knots_all (_type_): _description_
        points_all (_type_): _description_
        angles_all (_type_): _description_
    Returns:
        _type_: _description_
    """
    print("postprocess_no_images")
    cobb = []
    for cp, knots, points, angles in zip(cp_all, knots_all, points_all, angles_all):
        
        bs = BSpline(9, 3, P=cp.reshape(-1, 2), U=torch.cat(
            [torch.zeros(3 + 1).to(device), knots, torch.ones(3 + 1).to(device)]))
        mt_cobb, pt_cobb, tl_cobb, end_points, ret, grad = getCobb_bspline(
            9, 3, cp, knots, bs)
        
        if dataset == "sanyuan":
            alpha, beta, gamma = 0.5, 0.4, 0.7  # sanyuan
        elif dataset == "public":
            alpha, beta, gamma = 0.4, 0.5, 0.5  # public
        mt_cobb2, pt_cobb2, tl_cobb2 = angles * 90
        mt_cobb = mt_cobb * alpha + mt_cobb2 * (1 - alpha)
        pt_cobb = pt_cobb * beta + pt_cobb2 * (1 - beta)
        tl_cobb = tl_cobb * gamma + tl_cobb2 * (1 - gamma)
        cobb.append([float(mt_cobb), float(pt_cobb), float(tl_cobb)])
        
        cp = cp.reshape([-1, 2])
        cp[:, 0], cp[:, 1] = cp[:, 0] * 256, cp[:, 1] * 512
        
        end_points = [item.cpu().tolist() for item in end_points]
    return cobb, end_points, grad

def inference(images, model_seg, model_refine, model_reg, mode, save_path_base):
    print("inference")
    ret_dict = {}
    with torch.no_grad():
        if mode == "bspine":
            mask1 = model_seg(images.to(device))
            mask2 = model_refine(mask1)
            points_all, bspline_all, angles_all = model_reg(mask2)
        elif mode == "cvspine":
            mask1 = cvs_model_seg(images.to(device))
            mask1 = mask1[1]
            mask2 = cvs_model_refine(mask1)
            points_all, bspline_all, angles_all = cvs_model_reg(mask2)
        
        cp_all, knots_all = torch.split(
            bspline_all, [2 * (9 + 1), 9 - 3], dim=-1)
        
        images_all = []
        if ret_images:
            images_drawed, cobb, end_points, grad = postprocess(
                images, cp_all, knots_all, points_all, angles_all)
            for masks in images_drawed:
                mask_all = []
                for mask in masks:
                    _, buffer = cv2.imencode('.png', mask)
                    img_byte = base64.b64encode(buffer)
                    img_b64 = img_byte.decode('utf-8')
                    mask_all.append(img_b64)
                images_all.append(mask_all)
        else:
            cobb, end_points, grad = postprocess_no_images(
                cp_all, knots_all, points_all, angles_all)
            end_points = [i*2 for i in end_points]
        
        ret_dict['cp'] = cp_all.cpu().numpy().tolist()
        ret_dict['knots'] = knots_all.cpu().numpy().tolist()
        ret_dict['points'] = points_all.cpu().numpy().tolist()
        ret_dict['end_points'] = end_points
        ret_dict['derivative'] = grad.cpu().numpy().tolist()
        ret_dict['cobb'] = cobb

    return ret_dict, None

# route page home for browser visit
@app.route('/')
def home():
    print("home")
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    print("predict")
    file = request.files['image']
    mode = request.args.get('mode', default=default_mode, type=str)
    
    base_path = os.path.dirname(__file__)
    save_path_base = "./B_CVS_Spine_result"  

    uploaded_filename = secure_filename(file.filename)
    if not os.path.exists(os.path.join(base_path, "temp")):
        os.makedirs(os.path.join(base_path, "temp"))
    upload_path = os.path.join(base_path, "temp", uploaded_filename)
    file.save(upload_path)
    
    base_filename, _ = os.path.splitext(uploaded_filename)
    
    img = Image.open(upload_path).convert("RGB")
    img_tensor = preprocess(img)
    
    if mode == "bspine":
        ret, _ = inference(img_tensor, model_seg, model_refine, model_reg, mode, save_path_base)
        json_path = os.path.join(save_path_base, "bspine_re", f"{base_filename}.json")
    elif mode == "cvspine":
        ret, _ = inference(img_tensor, cvs_model_seg, cvs_model_refine, cvs_model_reg, mode, save_path_base)
        json_path = os.path.join(save_path_base, "cvspine_re", f"{base_filename}.json")
    else:
        return jsonify({"error": "Invalid mode. Please choose 'bspine' or 'cvspine'."}), 400

    if not os.path.exists(os.path.dirname(json_path)):
        os.makedirs(os.path.dirname(json_path))
    try:
        with open(json_path, 'w') as json_file:
            json.dump(ret, json_file, indent=4)
        print(f"Saved JSON file at: {json_path}")
    except Exception as e:
        print(f"Failed to save JSON file: {e}")
        return jsonify({"error": f"Failed to save JSON file: {e}"}), 500

    data_to_return = {
        "ret": ret,
        "json_path": json_path
    }

    response_data = json.dumps(data_to_return, indent=4, ensure_ascii=False)
    response = make_response(response_data)
    response.headers["Content-Type"] = "application/json; charset=utf-8"
    response.headers["Content-Disposition"] = f'inline; filename="{base_filename}.json"'

    return response

if __name__ == '__main__':
    print("main")
    server = pywsgi.WSGIServer(('127.0.0.1', 5000), app)
    server.serve_forever()
