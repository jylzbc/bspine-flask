import cv2
import random
import numpy as np
from scipy.interpolate import CubicSpline, CubicHermiteSpline
import math
import torch
from torch import nn
from torch.autograd import Variable

from lib.model.myvit import DeepViT
from lib.model.resnet import resnet101
from lib.model.densenet import densenet169
from lib.model.myeffnet import _efficientnet
from lib.model.myresnet import resnet101, resnet152
from lib.utils.bspline import BSpline, BSpline_np


def get_deepvit(in_channels, num_classes):
    return DeepViT(image_size=512,
                   channels=in_channels,
                   patch_size=32,
                   num_classes=num_classes,
                   dim=1024,
                   depth=6,
                   heads=16,
                   mlp_dim=2048,
                   dropout=0.1,
                   emb_dropout=0.1)


def get_resnet(in_channels, num_classes):
    model = resnet101(pretrained=False, num_classes=num_classes)

    # 修改第一层的卷积输入通道为1
    o_ch, ks, s, p = model.conv1.out_channels, model.conv1.kernel_size, model.conv1.stride, model.conv1.padding
    model.conv1 = nn.Conv2d(in_channels, o_ch, ks, s, p)
    # model.apply(weights_init_normal)
    return model


def get_resnet2(in_channels, num_classes):
    model = resnet101(pretrained=False, num_classes=num_classes)
    # model=resnet152(pretrained=False, num_classes=num_classes)

    # 修改第一层的卷积输入通道为1
    o_ch, ks, s, p = model.conv1.out_channels, model.conv1.kernel_size, model.conv1.stride, model.conv1.padding
    model.conv1 = nn.Conv2d(in_channels, o_ch, ks, s, p)

    return model


def get_effnet(in_channels, num_classes):
    model = _efficientnet('efficientnet_b5', in_channels=in_channels, num_classes=num_classes)
    return model


def get_densenet(in_channels, num_classes):
    model = densenet169(num_classes=num_classes)
    return model


def sample_bspline(args, cp_tensor, knots_tensor, centerline):
    points = []
    uq = torch.linspace(0, 1, centerline.shape[1])

    for cp, knots in zip(cp_tensor, knots_tensor):
        # knots
        knots_ = torch.cat([torch.zeros(args.k + 1).cuda(), knots, torch.ones(args.k + 1).cuda()])
        bs_ = BSpline(args.n, args.k, cp.reshape([-1, 2]), knots_)
        ret = bs_.run(uq)
        points.append(ret)
    points = torch.stack(points, dim=0)
    return points


def getCobb_derivative(derivative):
    if len(derivative) <= 1:
        cobb = torch.tensor(0, device="cpu")
        lidx = 0
        ridx = 0
    else:
        max_idx = derivative.argmax()
        min_idx = derivative.argmin()
        k1 = derivative[max_idx]
        k2 = derivative[min_idx]
        cobb = torch.arctan((k1 - k2) / (1 + k1 * k2)) * 180 / math.pi
        lidx = min(max_idx, min_idx)
        ridx = max(max_idx, min_idx)
    return abs(cobb), lidx, ridx


def getCobb_cspline(data):
    data = data.cpu().numpy()
    x, y = data[:, 1], data[:, 0]
    for i in range(len(x) - 1):
        if x[i + 1] - x[i] < 0:
            x[i + 1] = x[i] + 0.05
    # cs=CubicSpline(x,y)
    # cs = CubicHermiteSpline(x, y, [0] * len(x))
    # cs=KroghInterpolator(x,y)
    # cs=InterpolatedUnivariateSpline(x,y)
    cs=np.poly1d(np.polyfit(x,y,deg=6))

    EPISILON = 5e-2
    # uq=torch.linspace(0,1-EPISILON,17)
    uq = np.linspace(x[0], x[-1] - EPISILON, 17)
    uq_delta = uq + EPISILON
    y0 = cs(uq)
    xy0 = np.stack([uq, y0], axis=-1)
    xy0[:, 0], xy0[:, 1] = xy0[:, 0] * 512, xy0[:, 1] * 256
    y1 = cs(uq_delta)
    xy1 = np.stack([uq_delta, y1], axis=-1)
    xy1[:, 0], xy1[:, 1] = xy1[:, 0] * 512, xy1[:, 1] * 256
    mask = np.zeros([512, 256, 3])
    for y, x in xy0:
        cv2.circle(mask, (int(x), int(y)), 2, (0, 255, 0), 4)
    derivative = torch.from_numpy((xy1[:, 1] - xy0[:, 1]) / (xy1[:, 0] - xy0[:, 0])).float().cuda()

    # 计算Cobb
    mt_cobb, ml, mr = getCobb_derivative(derivative[:])
    pt_cobb, pl, pr = getCobb_derivative(derivative[:ml + 1])
    tl_cobb, tl, tr = getCobb_derivative(derivative[mr:])
    end_points = [pl, ml, mr, mr + tr]
    return abs(mt_cobb), abs(pt_cobb), abs(tl_cobb), end_points, xy0


def getCobb_bspline(n, k, cp, knots, bs=None, centerline=None):
    # EPISILON=2.23e-3
    EPISILON = 5e-2

    # 计算得斜率
    if bs is None:
        bs = BSpline(n, k, P=cp, U=knots)
    uq=torch.linspace(0.05,0.95-EPISILON,17)

    uq_delta = uq + EPISILON
    xy0 = bs.run(uq)
    xy0[:, 0], xy0[:, 1] = xy0[:, 0] * 256, xy0[:, 1] * 512
    xy1 = bs.run(uq_delta)
    xy1[:, 0], xy1[:, 1] = xy1[:, 0] * 256, xy1[:, 1] * 512
    # mask=np.zeros([512,256,3])
    # for x,y in xy0:
    #     cv2.circle(mask, (int(x), int(y)), 2, (0, 255, 0), 4)
    # derivative=(xy1[:,0]-xy0[:,0])/(xy1[:,1]-xy0[:,1])
    derivative = (xy0[:, 0] - xy1[:, 0]) / (xy0[:, 1] - xy1[:, 1])
    derivative = derivative.to("cpu")

    # 计算Cobb
    mt_cobb, ml, mr = getCobb_derivative(derivative[:])
    pt_cobb, pl, pr = getCobb_derivative(derivative[:ml + 1])
    tl_cobb, tl, tr = getCobb_derivative(derivative[mr:])
    end_points = [pl, ml, mr, mr + tr]
    
    return abs(mt_cobb), abs(pt_cobb), abs(tl_cobb), end_points, xy1, derivative


class ReplayBuffer:
    def __init__(self, max_size=50):
        assert max_size > 0, "Empty buffer or trying to create a black hole. Be careful."
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))


class LambdaLR:
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert (n_epochs - decay_start_epoch) > 0, "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)
