import torch.nn as nn
from ltr.models.neck import CorrNL
from ltr import model_constructor
'''2020.4.14 newly added'''
import torch
import ltr.models.backbone.resnet_seg as resnet_seg
# '''2020.4.17 newly added'''
# from resnest.torch.resnest_seg import resnest101,resnest50
from ltr.models.head import corner,seg_network
from easydict import EasyDict as edict
'''2020.4.14 将mask分支替换成frtm中的结构，实现更加精细的分割'''
'''2020.4.24 加入距离变换图'''
class ARcmnet_seg_dtm(nn.Module):
    """ Scale Estimation network module with three branches: bbox, coner and mask. """
    def __init__(self, feature_extractor, neck_module, head_module, used_layers,
                 extractor_grad=True):
        """
        args:
            feature_extractor - backbone feature extractor
            bb_regressor - IoU prediction module
            bb_regressor_layer - List containing the name of the layers from feature_extractor, which are input to
                                    bb_regressor
            extractor_grad - Bool indicating whether backbone feature extractor requires gradients
        """
        super(ARcmnet_seg_dtm, self).__init__()

        self.feature_extractor = feature_extractor
        self.neck = neck_module
        self.corner_head, self.refiner = head_module
        self.used_layers = used_layers

        if not extractor_grad:
            for p in self.feature_extractor.parameters():
                p.requires_grad_(False)

    def forward(self, train_imgs, test_imgs, train_bb, test_dtm, mode='train'):
        """ Forward pass
        Note: If the training is done in sequence mode, that is, test_imgs.dim() == 5, then the batch dimension
        corresponds to the first dimensions. test_imgs is thus of the form [sequence, batch, feature, row, col]
        """
        self.forward_ref(train_imgs, train_bb)
        pred_dict = self.forward_test(test_imgs, test_dtm, mode)
        return pred_dict

    def forward_ref(self, train_imgs, train_bb):
        """ Forward pass of reference branch.
        size of train_imgs is (1,batch,3,H,W), train_bb is (1,batch,4)"""
        num_sequences = train_imgs.shape[-4] # batch
        num_train_images = train_imgs.shape[0] if train_imgs.dim() == 5 else 1 # 1

        # Extract backbone features
        '''train_feat的数据类型都是OrderedDict,字典的键为'layer4' '''
        train_feat_dict = self.extract_backbone_features(train_imgs.view(-1, *train_imgs.shape[-3:])) # 输入size是(batch,3,256,256)

        train_feat_list = [feat for feat in train_feat_dict.values()] #list,其中每个元素对应一层输出的特征(tensor)

        # get reference feature
        self.neck.get_ref_kernel(train_feat_list, train_bb.view(num_train_images, num_sequences, 4))


    def forward_test(self, test_imgs, test_dtm, mode='train'):
        """ Forward pass of test branch. size of test_imgs is (1,batch,3,256,256)"""
        # test_dtm: (1,batch,1,16,16) when train, (batch,1,16,16) when test
        output = {}
        # Extract backbone features
        test_feat_dict = self.extract_backbone_features(test_imgs.view(-1, *test_imgs.shape[-3:]),
                                                        layers=['layer1','layer2','layer3','layer4','layer5'])# 输入size是(batch,3,256,256)
        '''list,其中每个元素对应一层输出的特征(tensor)'''
        # Save low-level feature list
        # Lfeat_list = [feat for name, feat in test_feat_dict.items() if name != 'layer3']

        # fuse feature from two branches
        fusion_feat = self.neck.fuse_feat([test_feat_dict['layer4']])
        '''2020.4.24 concat test_dtm'''
        test_dtm_4d = test_dtm.view(-1,*test_dtm.shape[-3:]) # (batch,1,H/16,W/16)
        fusion_feat = torch.cat([fusion_feat,test_dtm_4d],dim=1) # (batch,(H/32)*(W/32)+1,H/16,W/16)
        # Obtain bbox prediction
        if mode=='train':
            output['corner'] = self.corner_head(fusion_feat)
            output['mask'] = torch.sigmoid(self.refiner(fusion_feat, test_feat_dict, test_imgs.shape[-2:]))
        elif mode=='mask':
            output = torch.sigmoid(self.refiner(fusion_feat, test_feat_dict, test_imgs.shape[-2:]))
        else:
            raise ValueError("mode should be train or mask")
        return output

    def extract_backbone_features(self, im, layers=None):
        if layers is None:
            layers = self.used_layers
        return self.feature_extractor(im, layers)

    def extract_features(self, im, layers):
        return self.feature_extractor(im, layers)


'''2020.4.24 Use ResNet-50 and introduce distance transformation map'''
@model_constructor
def ARcmseg_dtm_resnet50(backbone_pretrained=True,used_layers=('layer4',),pool_size=None):
    # backbone
    backbone_net = resnet_seg.resnet50(pretrained=backbone_pretrained)
    # neck
    neck_net = CorrNL.CorrNL(pool_size=pool_size)
    # multiple heads
    # non-local(64)+ dt map(1)
    corner_head = corner.Corner_Predictor(inplanes=pool_size*pool_size+1,
                                          output_sz=int(pool_size*2*16))
    '''create segnet'''
    in_channels = 1024
    # non-local feat(64 channels) + dt map(1 channel)
    disc_params = edict(layer="layer4", in_channels=in_channels,
                        c_channels=96, out_channels=pool_size*pool_size+1)
    refnet_params = edict(
        layers=("layer5", "layer4", "layer3", "layer2"),
        nchannels=64, use_batch_norm=True)
    disc_params.in_channels = backbone_net.get_out_channels()[disc_params.layer]

    p = refnet_params
    refinement_layers_channels = {L: nch for L, nch in backbone_net.get_out_channels().items() if L in p.layers}
    refiner = seg_network.SegNetwork(disc_params.out_channels, p.nchannels, refinement_layers_channels, p.use_batch_norm)
    '''create Alpha-Refine'''
    net = ARcmnet_seg_dtm(feature_extractor=backbone_net, neck_module=neck_net,
                   head_module=(corner_head, refiner),
                   used_layers=used_layers, extractor_grad=True)
    return net

