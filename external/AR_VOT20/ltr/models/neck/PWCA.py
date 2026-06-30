import torch.nn as nn
import torch
try:
    from torch2trt import torch2trt
    from torch2trt import TRTModule
except:
    print('no tensorrt installed')
import os

class PWCA(nn.Module):
    """
    2021.4.8 Removing PrPooling, Pointwise Correlation + Channel Attention
    args:
        input_dim:  Feature dimensionality of the two input backbone layers.
        pred_input_dim:  Dimensionality input the the prediction network.
        pred_inter_dim:  Intermediate dimensionality in the prediction network."""

    def __init__(self, target_sz=8):
        super().__init__()
        num_corr_channel = target_sz * target_sz
        self.CA = SEModule(num_corr_channel, reduction=4)
        self.f_min = target_sz // 2
        self.f_max = target_sz + self.f_min
        self.ref_kernel = None
        self.init_trt_flag = 0

    def forward(self, f_z, f_x):
        """Runs the ATOM IoUNet during training operation.
        This forward pass is mainly used for training. Call the individual functions during tracking instead.
        args:
            feat1:  Features from the reference frames (4 or 5 dims).
            feat2:  Features from the test frames (4 or 5 dims)."""
        # deal with dimension first
        if len(f_z) == 1:
            f_z = f_z[0]  # size (bs,C,H,W)
            f_x = f_x[0]  # size (bs,C,H,W)
        else:
            raise ValueError("Only supporting single-layer feature for now")
        self.get_ref_kernel(f_z)
        self.fuse_feat(f_x)

    def init_trt(self):
        self.AR_neck_CA_trt = TRTModule()

        self.AR_neck_CA_trt.load_state_dict(torch.load(os.path.dirname(os.path.abspath(__file__))+'/../../../checkpoints/trt_models/AR_neck_CA_trt.pth'))

    def get_ref_kernel(self, f_z):
        if len(f_z) == 1:
            f_z = f_z[0]
        self.ref_kernel = f_z[:, :, self.f_min: self.f_max, self.f_min: self.f_max]

    def fuse_feat(self, f_x, trt='false'):
        if len(f_x) == 1:
            f_x = f_x[0]
        feat_corr = self.pw_corr(self.ref_kernel, f_x)
        if trt == 'false' or trt == 'tansform_backbone':
            feat_ca = self.CA(feat_corr)
            return feat_ca
        elif trt == 'transform':
            self.CA_trt = torch2trt(self.CA, [feat_corr], fp16_mode=True)
            feat_ca = self.CA_trt(feat_corr)
            torch.save(self.CA_trt.state_dict(), os.path.dirname(os.path.abspath(__file__))+'/../../../checkpoints/trt_models/AR_neck_CA_trt.pth')
            return feat_ca
        elif trt == 'true':
            if self.init_trt_flag == 0:
                self.init_trt()
            self.init_trt_flag = 1
            feat_ca = self.AR_neck_CA_trt(feat_corr)
            return feat_ca
        else:
            raise ValueError('neck error!')
        """fuse features from reference and test branch"""
        # Step1: pixel-wise correlation
        # Step2: channel attention: Squeeze and Excitation
        #zxh:2021.4.19 transform CA to trt
            # feat_ca = self.CA(feat_corr)
            # feat_ca = self.AR_neck_CA_trt(feat_corr)

            # return feat_ca

    def pw_corr(self, f_z, f_x):
        """Simple implementation of point-wise correlation"""
        bs, c, h_z, w_z = f_z.size()
        _, _, h_x, w_x = f_x.size()
        fz_m = f_z.reshape(bs, c, h_z * w_z).transpose(-1, -2)  # (bs, h_z * w_z, c)
        fx_m = f_x.reshape(bs, c, h_x * w_x)  # (bs, c, h_x * w_x)
        corr = torch.matmul(fz_m, fx_m)  # (bs, h_z * w_z, h_x * w_x)
        return corr.view(bs, h_z * w_z, h_x, w_x)


class SEModule(nn.Module):
    """Channel attention module"""
    def __init__(self, channels, reduction=4):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,
                             padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1,
                             padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x
