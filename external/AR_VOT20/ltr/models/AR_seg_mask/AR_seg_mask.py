import torch.nn as nn
from ltr.models.neck import PWCA
from ltr import model_constructor
import torch
import ltr.models.backbone.resnet_seg as resnet_seg

from ltr.models.head import corner, seg_network
from easydict import EasyDict as edict
try:
    from torch2trt import torch2trt
    from torch2trt import TRTModule
except:
    print('no tensorrt/torch2trt installed')
import os
import sys


from torch.autograd import Variable

'''2020.4.14 Replacing the mask branch with the structure in the frtm for preciser segmentation'''
'''2020.4.22 Only use the mask branch'''


class ARnet_seg_mask(nn.Module):
    """ AlphaRefine with a single mask branch. """

    def __init__(self, feature_extractor, neck_module, head_module, used_layers,
                 extractor_grad=True, output_size=(256, 256)):

        super(ARnet_seg_mask, self).__init__()

        self.feature_extractor = feature_extractor
        self.neck = neck_module
        self.refiner = head_module
        self.used_layers = used_layers
        self.output_size = output_size
        self.trt_flag = 0
        # self.project = seg_network.BackwardCompatibleUpsampler(64)



        if not extractor_grad:
            for p in self.feature_extractor.parameters():
                p.requires_grad_(False)

    def forward(self, train_imgs, test_imgs, mode='train'):
        """ Forward pass
        Note: If the training is done in sequence mode, that is, test_imgs.dim() == 5, then the batch dimension
        corresponds to the first dimensions. test_imgs is thus of the form [sequence, batch, feature, row, col]
        """

        self.forward_ref(train_imgs)
        pred_dict = self.forward_test(test_imgs, mode)
        return pred_dict

    def forward_ref(self, train_imgs, trt='false'):
        filepath = os.path.abspath(__file__)
        AR_dir = os.path.join(os.path.dirname(filepath), "..", "..", "..")
        sys.path.append(AR_dir)
        checkpoints_dir = os.path.join(AR_dir, "checkpoints")
        if not os.path.exists(checkpoints_dir + '/' + 'trt_models'):
            os.mkdir(checkpoints_dir + '/' + 'trt_models')
        if trt == 'false' or trt == 'transform_backbone' or trt == 'transform':
            train_feat_dict = self.extract_backbone_features(train_imgs.view(-1, *train_imgs.shape[-3:]))
            train_feat_list = [feat for feat in train_feat_dict]
            self.neck.get_ref_kernel(train_feat_list)
        if trt == 'true':
            if self.trt_flag == 0:
                self.init_tensorrt()
            self.trt_flag = 1
            train_feat_dict = self.test_AR_mask_backbone_trt(train_imgs.view(-1, *train_imgs.shape[-3:]))
            train_feat_list = [feat for feat in [train_feat_dict[3]]]
            self.neck.get_ref_kernel(train_feat_list)
        """ Forward pass of reference branch.
        size of train_imgs is (1,batch,3,H,W), train_bb is (1,batch,4)"""

    def forward_test(self, test_imgs, trt='false', mode='train'):
        """ Forward pass of test branch. size of test_imgs is (1,batch,3,256,256)"""
        #zxh:2021.4.19  extract backbone(layer4) feature[list]
        # Extract backbone features
        if trt == 'false':
            test_feat_dict = self.extract_backbone_features_test(
                test_imgs.view(-1, *test_imgs.shape[-3:]))  # 输入size是(batch,3,256,256)
            fusion_feat = self.neck.fuse_feat([test_feat_dict[3]], trt=trt)
            if mode == 'train':
                output = {'mask': torch.sigmoid(self.refiner(fusion_feat, test_feat_dict, self.output_size))}
            elif mode == 'mask':
                refiner_feat_module = self.refiner(fusion_feat, test_feat_dict[4], test_feat_dict[3], test_feat_dict[2], test_feat_dict[1])
                refiner_feat = self.refiner.forward_project(refiner_feat_module)
                output = torch.sigmoid(refiner_feat)
            else:
                raise ValueError("mode should be train or test")
            return output

        if trt == 'transform':
            test_feat_dict = self.extract_backbone_features_test(test_imgs.view(-1, *test_imgs.shape[-3:]))  # 输入size是(batch,3,256,256)
            fusion_feat = self.neck.fuse_feat([test_feat_dict[3]], trt=trt)
            if mode == 'train':
                output = {'mask': torch.sigmoid(self.refiner(fusion_feat, test_feat_dict, self.output_size))}
            elif mode == 'mask':  ###
                # refiner_feat = self.refiner(fusion_feat, test_feat_dict, self.output_size, trt=trt)
                # self.refiner_trt = torch2trt(self.refiner.forward_trt, [fusion_feat, test_feat_dict[4], test_feat_dict[3],
                #                                             test_feat_dict[2], test_feat_dict[1]])
                self.refiner_trt = torch2trt(self.refiner, [fusion_feat, test_feat_dict[4], test_feat_dict[3],
                                                            test_feat_dict[2], test_feat_dict[1]], fp16_mode=True)
                torch.save(self.refiner_trt.state_dict(), os.path.dirname(os.path.abspath(__file__))+'/../../../checkpoints/trt_models/refinenet_trt.pth')
                refiner_feat = self.refiner_trt(fusion_feat, test_feat_dict[4], test_feat_dict[3],
                                                            test_feat_dict[2], test_feat_dict[1])
                exit()
                output = torch.sigmoid(refiner_feat)

            else:
                raise ValueError("mode should be train or test")
            return output
        if trt == 'transform_backbone':
            # self.test_backbone_trt = torch2trt(self.extract_backbone_features_test, [test_imgs.view(-1, *test_imgs.shape[-3:])])
            self.test_backbone_trt = torch2trt(self.feature_extractor, [test_imgs.view(-1, *test_imgs.shape[-3:])], fp16_mode=True)
            test_feat_dict = self.test_backbone_trt(test_imgs.view(-1, *test_imgs.shape[-3:]))
            torch.save(self.test_backbone_trt.state_dict(), os.path.dirname(os.path.abspath(__file__))+'/../../../checkpoints/trt_models/test_AR_backbone_trt.pth')
            exit()
            fusion_feat = self.neck.fuse_feat([test_feat_dict[3]], trt=trt)
            if mode == 'train':
                output = {'mask': torch.sigmoid(self.refiner(fusion_feat, test_feat_dict, self.output_size))}
            elif mode == 'mask':  ###
                refiner_feat = self.refiner(fusion_feat, test_feat_dict, self.output_size, trt=trt)
                output = torch.sigmoid(refiner_feat)
            else:
                raise ValueError("mode should be train or test")
            return output
        if trt == 'true':
            test_feat_dict = self.test_AR_mask_backbone_trt(test_imgs.view(-1, *test_imgs.shape[-3:]))
        # fuse feature from two branches
            fusion_feat = self.neck.fuse_feat([test_feat_dict[3]], trt=trt)
        # fusion_feat = self.neck.fuse_feat([test_feat_dict['layer4']])
        # Obtain bbox prediction
            if mode == 'train':
                output = {'mask': torch.sigmoid(self.refiner(fusion_feat, test_feat_dict, self.output_size))}
            elif mode == 'mask':   ###
                refiner_feat_module = self.refinenet_trt(fusion_feat, test_feat_dict[4], test_feat_dict[3], test_feat_dict[2], test_feat_dict[1])
                refiner_feat = self.refiner.forward_project(refiner_feat_module)
                output = torch.sigmoid(refiner_feat)
            # output = torch.sigmoid(self.refiner(fusion_feat, test_feat_dict, self.output_size))
            else:
                raise ValueError("mode should be train or test")
            return output

    def forward_ego(self, imgs):
        """ Forward pass of test branch. size of test_imgs is (1,batch,3,256,256)"""
        # Extract backbone features
        feat_dict = self.extract_backbone_features(imgs.view(-1, *imgs.shape[-3:]),
                                                   layers=['layer1', 'layer2', 'layer3', 'layer4', 'layer5'])
        feat_list = [feat_dict["layer4"]]
        # get reference feature
        self.neck.get_ref_kernel(feat_list)
        # fuse feature from two branches
        fusion_feat = self.neck.fuse_feat([feat_dict['layer4']])
        # Obtain bbox prediction
        output = torch.sigmoid(self.refiner(fusion_feat, feat_dict, self.output_size))
        return output

    def extract_backbone_features(self, im, layers=None):
        if layers is None:
            layers = self.used_layers
        return self.feature_extractor(im, layers)

    def extract_backbone_features_test(self, im, layers=['layer1', 'layer2', 'layer3', 'layer4', 'layer5']):
        if layers is None:
            layers = self.used_layers
        return self.feature_extractor(im, layers)

    def extract_features(self, im, layers):
        return self.feature_extractor(im, layers)

    def init_tensorrt(self):
        self.test_AR_mask_backbone_trt = TRTModule()
        self.refinenet_trt = TRTModule()

        self.test_AR_mask_backbone_trt.load_state_dict(torch.load(os.path.dirname(os.path.abspath(__file__))+'/../../../checkpoints/trt_models/test_AR_backbone_trt.pth'))
        self.refinenet_trt.load_state_dict(torch.load(os.path.dirname(os.path.abspath(__file__))+'/../../../checkpoints/trt_models/refinenet_trt.pth'))


@model_constructor
def ARnet_seg_mask_resnet50(backbone_pretrained=True, used_layers=['layer4'], target_sz=None):
    # backbone
    backbone_net = resnet_seg.resnet50(pretrained=backbone_pretrained)
    # neck
    neck_net = PWCA.PWCA(target_sz=target_sz)
    # multiple heads
    '''create segnet'''
    in_channels = 1024

    '''2020.4.22 change "out_channels" to target_sz * target_sz'''
    disc_params = edict(layer="layer4", in_channels=in_channels, c_channels=96,
                        out_channels=target_sz * target_sz)  # non-local feat (64 channels rather than 1)
    refnet_params = edict(
        layers=("layer5", "layer4", "layer3", "layer2"),
        nchannels=64, use_batch_norm=True)
    p = refnet_params
    disc_params.in_channels = backbone_net.get_out_channels()[disc_params.layer]

    #p.layers = ("layer5", "layer4", "layer3", "layer2")
    refinement_layers_channels = {L: nch for L, nch in backbone_net.get_out_channels().items() if L in p.layers}
    # refinement_layers_channels = {nch for L, nch in backbone_net.get_out_channels().items() if L in p.layers}
    refiner = seg_network.SegNetwork(disc_params.out_channels, p.nchannels, refinement_layers_channels,
                                     p.use_batch_norm)

    # refiner = seg_network.SegNetwork(out_channels=target_sz * target_sz, nchannels=64, refinement_layers_channels,
    #                                  use_batch_norm=True)
    '''create Alpha-Refine'''
    net = ARnet_seg_mask(feature_extractor=backbone_net, neck_module=neck_net,
                         head_module=refiner,
                         used_layers=used_layers, extractor_grad=True,
                         output_size=(int(target_sz * 2 * 16), int(target_sz * 2 * 16)))
    return net
