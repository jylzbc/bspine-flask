import torch

# ckpt path
dataset = "sanyuan"
ckpt_seg = "pretrained/seg_490.pth"
ckpt_refine = "pretrained/netG_A2B_145.pth"
ckpt_reg = "pretrained/bspline_24000.pth"

cvs_ckpt_seg = "pretrained/seg_9999.pth"
cvs_ckpt_refine = "pretrained/netG_A2B_199.pth"
cvs_ckpt_reg = "pretrained/reg_9999.pth"


device = torch.device("cpu")
ret_images = True
mode = "bspine"