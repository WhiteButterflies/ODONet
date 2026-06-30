import torch
from torch import nn as nn
from torch.nn import functional as F
import os
import sys
env_path = os.path.join(os.path.dirname(__file__), '..', '..', '..')
if env_path not in sys.path:
    sys.path.append(env_path)
from ltr.models.head.utils import conv, relu, interpolate, adaptive_cat
try:
    from torch2trt import torch2trt
    from torch2trt import TRTModule
except:
    print('no tensorrt installed')
import os

class TSE(nn.Module):

    def __init__(self, fc, ic, oc):
        super().__init__()

        nc = ic + oc
        self.reduce = nn.Sequential(conv(fc, oc, 1), relu(), conv(oc, oc, 1))
        self.transform = nn.Sequential(conv(nc, nc, 3), relu(), conv(nc, nc, 3), relu(), conv(nc, oc, 3), relu())

    def forward(self, ft, score, x=None):
        h = self.reduce(ft)
        hpool = F.adaptive_avg_pool2d(h, (1, 1)) if x is None else x
        h = adaptive_cat((h, score), dim=1, ref_tensor=0)
        h = self.transform(h)
        return h, hpool

class CAB(nn.Module):

    def __init__(self, oc, fc, deepest):
        super().__init__()

        self.convreluconv = nn.Sequential(conv(2 * oc, oc, 1), relu(), conv(oc, oc, 1))
        self.deepest = deepest

    def forward(self, deeper, shallower, att_vec=None):
        shallow_pool = F.adaptive_avg_pool2d(shallower, (1, 1))
        deeper_pool = deeper if self.deepest else F.adaptive_avg_pool2d(deeper, (1, 1))
        if att_vec is not None:
            global_pool = torch.cat([shallow_pool, deeper_pool, att_vec], dim=1)
        else:
            global_pool = torch.cat((shallow_pool, deeper_pool), dim=1)
        conv_1x1 = self.convreluconv(global_pool)
        inputs = shallower * torch.sigmoid(conv_1x1)
        out = inputs + interpolate(deeper, inputs.shape[-2:])
        return out


class RRB(nn.Module):

    def __init__(self, oc, fc, num, use_bn=False):
        super().__init__()
        self.conv1x1 = conv(oc, oc, 1)
        if use_bn:
            self.bblock = nn.Sequential(conv(oc, oc, 3), nn.BatchNorm2d(oc), relu(), conv(oc, oc, 3, bias=False))
        else:
            self.bblock = nn.Sequential(conv(oc, oc, 3), relu(), conv(oc, oc, 3, bias=False))  # Basic block
        self.fc = fc
        self.num = num

    def forward(self, x):
        h = self.conv1x1(x)
        return F.relu(h + self.bblock(h))


class Upsampler(nn.Module):

    def __init__(self, in_channels=64):
        super().__init__()

        self.conv1 = conv(in_channels, in_channels // 2, 3)
        self.conv2 = conv(in_channels // 2, 1, 3)

    def forward(self, x, image_size):
        x = F.interpolate(x, (2 * x.shape[-2], 2 * x.shape[-1]), mode='bicubic', align_corners=False)
        x = F.relu(self.conv1(x))
        x = F.interpolate(x, image_size[-2:], mode='bicubic', align_corners=False)
        x = self.conv2(x)
        return x


class PyrUpBicubic2d(nn.Module):

    def __init__(self, channels):
        super().__init__()

        self.channels = channels

        def kernel(d):
            x = d + torch.arange(-1, 3, dtype=torch.float32)
            x = torch.abs(x)
            a = -0.75
            f = (x < 1).float() * ((a + 2) * x * x * x - (a + 3) * x * x + 1) + \
                ((x >= 1) * (x < 2)).float() * (a * x * x * x - 5 * a * x * x + 8 * a * x - 4 * a)
            W = f.reshape(1, 1, 1, len(x)).float()
            Wt = W.permute(0, 1, 3, 2)
            return W, Wt

        We, We_t = kernel(-0.25)
        Wo, Wo_t = kernel(-0.25 - 0.5)

        # Building non-separable filters for now. It would make sense to
        # have separable filters if it proves to be faster.

        # .contiguous() is needed until a bug is fixed in nn.Conv2d.
        self.W00 = (We_t @ We).expand(channels, 1, 4, 4).contiguous()
        self.W01 = (We_t @ Wo).expand(channels, 1, 4, 4).contiguous()
        self.W10 = (Wo_t @ We).expand(channels, 1, 4, 4).contiguous()
        self.W11 = (Wo_t @ Wo).expand(channels, 1, 4, 4).contiguous()

    def forward(self, input):

        if input.device != self.W00.device:
            self.W00 = self.W00.to(input.device)
            self.W01 = self.W01.to(input.device)
            self.W10 = self.W10.to(input.device)
            self.W11 = self.W11.to(input.device)

        a = F.pad(input, (2, 2, 2, 2), 'replicate')

        I00 = F.conv2d(a, self.W00, groups=self.channels)
        I01 = F.conv2d(a, self.W01, groups=self.channels)
        I10 = F.conv2d(a, self.W10, groups=self.channels)
        I11 = F.conv2d(a, self.W11, groups=self.channels)

        n, c, h, w = I11.shape

        J0 = torch.stack((I00, I01), dim=-1).view(n, c, h, 2 * w)
        J1 = torch.stack((I10, I11), dim=-1).view(n, c, h, 2 * w)
        out = torch.stack((J0, J1), dim=-2).view(n, c, 2 * h, 2 * w)

        out = F.pad(out, (-1, -1, -1, -1))
        return out


class BackwardCompatibleUpsampler(nn.Module):
    """ Upsampler with bicubic interpolation that works with Pytorch 1.0.1 """

    def __init__(self, in_channels=64):
        super().__init__()

        self.conv1 = conv(in_channels, in_channels // 2, 3)
        self.up1 = PyrUpBicubic2d(in_channels)
        self.conv2 = conv(in_channels // 2, 1, 3)
        self.up2 = PyrUpBicubic2d(in_channels // 2)

    def forward(self, x, image_size):
        x = self.up1(x)
        x = F.relu(self.conv1(x))
        x = self.up2(x)
        x = F.interpolate(x, image_size[-2:], mode='bilinear', align_corners=False)
        x = self.conv2(x)
        return x

class SegNetwork(nn.Module):

    def __init__(self, in_channels=1, out_channels=32, ft_channels=None, use_bn=False): #(64,64,n)

        super().__init__()

        assert ft_channels is not None
        self.ft_channels = ft_channels

        self.TSE_layer5 = TSE(2048, 144, 64)
        self.TSE_layer4 = TSE(1024, 144, 64)
        self.TSE_layer3 = TSE(512, 144, 64)
        self.TSE_layer2 = TSE(256, 144, 64)

        self.RRB1_layer5 = RRB(64, 2048, 1, use_bn=use_bn)
        self.RRB1_layer4 = RRB(64, 1024, 1, use_bn=use_bn)
        self.RRB1_layer3 = RRB(64, 512, 1, use_bn=use_bn)
        self.RRB1_layer2 = RRB(64, 256, 1, use_bn=use_bn)

        self.CAB_layer5 = CAB(64, 2048, True)
        self.CAB_layer4 = CAB(64, 1024, False)
        self.CAB_layer3 = CAB(64, 512, False)
        self.CAB_layer2 = CAB(64, 256, False)

        self.RRB2_layer5 = RRB(64, 2048, 2, use_bn=use_bn)
        self.RRB2_layer4 = RRB(64, 1024, 2, use_bn=use_bn)
        self.RRB2_layer3 = RRB(64, 512, 2, use_bn=use_bn)
        self.RRB2_layer2 = RRB(64, 256, 2, use_bn=use_bn)


        self.project = BackwardCompatibleUpsampler(out_channels)


        #if torch.__version__ == '1.0.1'
    def forward_trt(self, scores, feature4, feature3, feature2, feature1):

        num_targets = scores.shape[0]
        num_fmaps = feature4.shape[0]
        # num_fmaps = 1
        if num_targets > num_fmaps:
            multi_targets = True
        else:
            multi_targets = False

        x = None

        ft = feature4
        s = interpolate(scores, ft.shape[-2:])
        h, hpool = self.TSE_layer5(ft, s, x)
        h = self.RRB1_layer5(h)
        h = self.CAB_layer5(hpool, h)
        x = self.RRB2_layer5(h)

        ft = feature3
        s = interpolate(scores, ft.shape[-2:])
        h, hpool = self.TSE_layer4(ft, s, x)
        h = self.RRB1_layer4(h)
        h = self.CAB_layer4(hpool, h)
        x = self.RRB2_layer4(h)

        ft = feature2
        s = interpolate(scores, ft.shape[-2:])
        h, hpool = self.TSE_layer3(ft, s, x)
        h = self.RRB1_layer3(h)
        h = self.CAB_layer3(hpool, h)
        x = self.RRB2_layer3(h)

        ft = feature1
        s = interpolate(scores, ft.shape[-2:])
        h, hpool = self.TSE_layer2(ft, s, x)
        h = self.RRB1_layer2(h)
        h = self.CAB_layer2(hpool, h)
        x = self.RRB2_layer2(h)
        return x


    def forward(self, scores, feature4, feature3, feature2, feature1):    #64
        num_targets = scores.shape[0]
        num_fmaps = feature4.shape[0]
        # num_fmaps = 1
        if num_targets > num_fmaps:
            multi_targets = True
        else:
            multi_targets = False

        x = None

        ft = feature4
        s = interpolate(scores, ft.shape[-2:])
        h, hpool = self.TSE_layer5(ft, s, x)
        h = self.RRB1_layer5(h)
        h = self.CAB_layer5(hpool, h)
        x = self.RRB2_layer5(h)

        ft = feature3
        s = interpolate(scores, ft.shape[-2:])
        h, hpool = self.TSE_layer4(ft, s, x)
        h = self.RRB1_layer4(h)
        h = self.CAB_layer4(hpool, h)
        x = self.RRB2_layer4(h)

        ft = feature2
        s = interpolate(scores, ft.shape[-2:])
        h, hpool = self.TSE_layer3(ft, s, x)
        h = self.RRB1_layer3(h)
        h = self.CAB_layer3(hpool, h)
        x = self.RRB2_layer3(h)

        ft = feature1
        s = interpolate(scores, ft.shape[-2:])
        h, hpool = self.TSE_layer2(ft, s, x)
        h = self.RRB1_layer2(h)
        h = self.CAB_layer2(hpool, h)
        x = self.RRB2_layer2(h)
        return x

    def forward_project(self, x):
        x = self.project(x, (384, 384))
        return x











