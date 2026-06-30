import os

import torch

# loss function related
from lib.utils.box_ops import giou_loss
from torch.nn.functional import l1_loss
from torch.nn import BCEWithLogitsLoss, MSELoss, CrossEntropyLoss
# train pipeline related
from lib.train.trainers import LTRTrainer,TestTrainer,LTR_PCGrad_Trainer
# distributed training related
from torch.nn.parallel import DistributedDataParallel as DDP
# some more advanced functions
from .base_functions import *
# network related
#from lib.models.mcitrack import build_mcitrack

from lib.train.actors import (
                              ODONET_V1_Actor,
                              )
from lib.utils.focal_loss import FocalLoss
from lib.utils.offset_loss import masked_offset_l1_loss, masked_cosine_similarity_loss,compute_offset_supervision_loss
from lib.utils.InfoNCE_loss import multi_view_contrastive_loss, multi_view_contrastive_loss_matrix,multi_view_contrastive_loss_matrix_tokenwise
from lib.utils.function_loss import compute_causal_input_nce_loss,compute_causal_regularization_loss
import lib.utils.causal_former_loss as causal_former_loss

'''first used in chotrack v2'''
from lib.utils.tracking_InfoNCE_loss import TrackingInfoNCELoss

# for import modules
import importlib


def run(settings):
    settings.description = 'Training script for Goku series'

    # update the default configs with config file
    if not os.path.exists(settings.cfg_file):
        raise ValueError("%s doesn't exist." % settings.cfg_file)
    config_module = importlib.import_module("lib.config.%s.config" % settings.script_name)
    cfg = config_module.cfg # generate cfg from lib.config
    config_module.update_config_from_file(settings.cfg_file) #update cfg from experiments
    if settings.local_rank in [-1, 0]:
        print("New configuration is shown below.")
        for key in cfg.keys():
            print("%s configuration:" % key, cfg[key])
            print('\n')

    # update settings based on cfg
    update_settings(settings, cfg)

    # Record the training log
    log_dir = os.path.join(settings.save_dir, 'logs')
    if settings.local_rank in [-1, 0]:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
    settings.log_file = os.path.join(log_dir, "%s-%s.log" % (settings.script_name, settings.config_name))

    # Build dataloaders
    loader_type = getattr(cfg.DATA, "LOADER", "tracking")
    if loader_type == "tracking":
        loader_train = build_dataloaders(cfg, settings)
    elif loader_type =='prev_tracking':
        loader_train,loader_val = build_train_prev_dataloaders(cfg,settings)
    elif loader_type =='prev_tracking_v3':
        loader_train,loader_val = build_train_prev_dataloaders_v3(cfg,settings)
    elif loader_type =='tracking_test':
        loader_train = build_test_dataloaders(cfg, settings)
    else:
        raise ValueError("illegal DATA LOADER")

    #


    # Create network
    if settings.script_name in ["mcitrack","mcitrack_hiera","mcitrack_ttt","mcitrack_vqd","mcitrack_vqd_vqh","mcitrack_vqdh_manba","mcitrack_no_manba","mcitrack_h_deform","mcitrack_h_deform_online",'mcitrack_h_deform_online_v1',"mcitrack_h_deform_online_v2","mcitrack_h_deform_online_v3","mcitrack_h_deform_online_v4","mcitrack_h_deform_online_v5"]:
        build_mcitrack_module = importlib.import_module("lib.models.%s.mcitrack" %settings.script_name)
        build_mcitrack = getattr(build_mcitrack_module, "build_mcitrack")
        net = build_mcitrack(cfg)
    elif settings.script_name in ['odonet_v1','odonet_wyq_v1','odonet_hn_v1','odonet_hn_v2','odonet_v1_odop','odonet_v1_pcgrad']:
        build_odonet_module = importlib.import_module("lib.models.%s.odonet" % settings.script_name)
        build_odonet = getattr(build_odonet_module, "build_odonet")
        net = build_odonet(cfg)
    elif settings.script_name in ['chotrack_v1',"chotrack_v2",'chotrack_v3','chotrack_v4','chotrack_v5','chotrack_v6','chotrack_v7','chotrack_v8','chotrack_v9','chotrack_v10','chotrack_v11','chotrack_v12','chotrack_v13','chotrack_v14','chotrack_v15','ovtrack_v1']:
        build_chotrack_module = importlib.import_module("lib.models.%s.chotrack" % settings.script_name)
        build_chotrack = getattr(build_chotrack_module, "build_chotrack")
        net = build_chotrack(cfg)
    else:
        raise ValueError("illegal script name")
    # freeze some of nets
    if settings.script_name in ["mcitrack_h_deform_online_v4"]:
        pass
    # if settings.script_name in ["mcitrack_vqd","mcitrack_vqdh_manba",]:
    #     for name,param in net.other_model_dict['vqvae'].named_parameters():
    #         param.requires_grad = False
    # if settings.script_name in ["mcitrack_vqd_vqh",]:
    #     for name,param in net.neck.named_parameters():
    #         param.requires_grad = False
        # for name,param in net.other_model_dict['vqvae_head'].named_parameters():
        #     param.requires_grad = False
        # for name,param in net.other_model_dict['down'].named_parameters():
        #     param.requires_grad = False
        # for name,param in net.other_model_dict['search_prompt_encoder'].named_parameters():
        #     param.requires_grad = False
        # for name,param in net.other_model_dict['search_mask_decoder'].named_parameters():
        #     param.requires_grad = False

    # wrap networks to distributed one
    net.cuda()

    #use torch.comnpile acc
    # torch._dynamo.config.optimize_ddp=False
    # net =torch.compile(net,mode="reduce-overhead")

    if settings.local_rank != -1:
        net = DDP(net, broadcast_buffers=False, device_ids=[settings.local_rank], find_unused_parameters=False)
        settings.device = torch.device("cuda:%d" % settings.local_rank)
    else:
        settings.device = torch.device("cuda:0")
    '''use torch.comnpile acc'''
    # torch._dynamo.config.optimize_ddp=False
    # net =torch.compile(net,mode='reduce-overhead',dynamic=True)
    '''see debug'''
    # torch._inductor.config.debug=True
    '''acc by tensor core'''
    torch.set_float32_matmul_precision('high')

    # Loss functions and Actors
    if settings.script_name in ["mcitrack","mcitrack_hiera","mcitrack_ttt","mcitrack_vqd","mcitrack_vqd_vqh","mcitrack_vqdh_manba","mcitrack_no_manba","mcitrack_h_deform",]:
        focal_loss = FocalLoss()
        objective = {'giou': giou_loss, 'l1': l1_loss, 'focal': focal_loss, 'cls': BCEWithLogitsLoss()}
        loss_weight = {'giou': cfg.TRAIN.GIOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT, 'focal': 1.,
                       'cls': cfg.TRAIN.CE_WEIGHT}
        if settings.script_name=="mcitrack":
            actor = MCITrackActor(net=net, objective=objective, loss_weight=loss_weight,
                                settings=settings, cfg=cfg)
        elif settings.script_name=="mcitrack_hiera":
            actor = MCITrack_Hiera_Actor(net=net, objective=objective, loss_weight=loss_weight,
                                settings=settings, cfg=cfg)
        elif settings.script_name=="mcitrack_ttt":
            actor = MCITrack_TTT_Actor(net=net, objective=objective, loss_weight=loss_weight,
                                settings=settings, cfg=cfg)
        elif settings.script_name=="mcitrack_vqd":
            actor = MCITrack_Vqd_Actor(net=net, objective=objective, loss_weight=loss_weight,
                                settings=settings, cfg=cfg)
        elif settings.script_name=="mcitrack_vqd_vqh":
            actor = MCITrack_Vqd_Vqh_Actor(net=net, objective=objective, loss_weight=loss_weight,
                                settings=settings, cfg=cfg)
        elif settings.script_name=="mcitrack_vqdh_manba":
            actor = MCITrack_Vqdh_Manba_Actor(net=net, objective=objective, loss_weight=loss_weight,
                                settings=settings, cfg=cfg)
        elif settings.script_name=="mcitrack_no_manba":
            actor = MCITrack_NoManba_Actor(net=net, objective=objective, loss_weight=loss_weight,
                                settings=settings, cfg=cfg)
        elif settings.script_name=="mcitrack_h_deform":
            actor = MCITrack_H_Deform_Actor(net=net, objective=objective, loss_weight=loss_weight,
                                settings=settings, cfg=cfg)
        else:
            raise ValueError("illegal script name")
    elif settings.script_name in ["mcitrack_h_deform_online",'mcitrack_h_deform_online_v1']:
        focal_loss = FocalLoss()
        objective = {'giou': giou_loss, 'l1': l1_loss, 'focal': focal_loss, 'cls': BCEWithLogitsLoss(),'offset_l1':masked_offset_l1_loss,'offset_cosine':masked_cosine_similarity_loss}
        loss_weight = {'giou': cfg.TRAIN.GIOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT, 'focal': 1.,
                       'cls': cfg.TRAIN.CE_WEIGHT,'offset_l1':cfg.TRAIN.OFFSET_L1_WEIGHT,'offset_cosine':cfg.TRAIN.OFFSET_COSINE_WEIGHT}

        if settings.script_name == "mcitrack_h_deform_online":
            actor = MCITrack_H_Deform_Online_Actor(net=net, objective=objective, loss_weight=loss_weight,
                                               settings=settings, cfg=cfg)
        elif settings.script_name == "mcitrack_h_deform_online_v1":
            actor = MCITrack_H_Deform_Online_V1_Actor(net=net, objective=objective, loss_weight=loss_weight,
                                               settings=settings, cfg=cfg)
    elif settings.script_name in ["mcitrack_h_deform_online_v2"]:
        focal_loss = FocalLoss()
        objective = {'giou': giou_loss, 'l1': l1_loss, 'focal': focal_loss, 'cls': BCEWithLogitsLoss(),
                     'offset_l1': masked_offset_l1_loss, 'offset_cosine': masked_cosine_similarity_loss}
        loss_weight = {'giou': cfg.TRAIN.GIOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT, 'focal': 1.,
                       'cls': cfg.TRAIN.CE_WEIGHT, 'offset_l1': cfg.TRAIN.OFFSET_L1_WEIGHT,
                       'offset_cosine': cfg.TRAIN.OFFSET_COSINE_WEIGHT}

        if settings.script_name == "mcitrack_h_deform_online_v2":
            actor = MCITrack_H_Deform_Online_V2_Actor(net=net, objective=objective, loss_weight=loss_weight,
                                               settings=settings, cfg=cfg)
    elif settings.script_name in ["mcitrack_h_deform_online_v3",'mcitrack_h_deform_online_v5']:
        focal_loss = FocalLoss()
        objective = {'giou': giou_loss, 'l1': l1_loss, 'focal': focal_loss, 'cls': BCEWithLogitsLoss(),
                     'offset_l1': masked_offset_l1_loss, 'offset_cosine': masked_cosine_similarity_loss,'pos_l1':compute_offset_supervision_loss,'simclr':multi_view_contrastive_loss_matrix_tokenwise}
        loss_weight = {'giou': cfg.TRAIN.GIOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT, 'focal': 1.,
                       'cls': cfg.TRAIN.CE_WEIGHT, 'offset_l1': cfg.TRAIN.OFFSET_L1_WEIGHT,'pos_l1': cfg.TRAIN.POS_L1_WEIGHT,
                       'offset_cosine': cfg.TRAIN.OFFSET_COSINE_WEIGHT,'simclr':cfg.TRAIN.SIMCLR_WEIGHT}

        if settings.script_name == "mcitrack_h_deform_online_v3":
            actor = MCITrack_H_Deform_Online_V3_Actor(net=net, objective=objective, loss_weight=loss_weight,
                                               settings=settings, cfg=cfg)
        elif settings.script_name == "mcitrack_h_deform_online_v5":
            actor = MCITrack_H_Deform_Online_V5_Actor(net=net, objective=objective, loss_weight=loss_weight,
                                               settings=settings, cfg=cfg)

    elif settings.script_name in ["mcitrack_h_deform_online_v4"]:
        focal_loss = FocalLoss()
        objective = {'giou': giou_loss, 'l1': l1_loss, 'focal': focal_loss, 'cls': BCEWithLogitsLoss(),
                     'offset_l1': masked_offset_l1_loss, 'offset_cosine': masked_cosine_similarity_loss,'simclr':multi_view_contrastive_loss_matrix}
        loss_weight = {'giou': cfg.TRAIN.GIOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT, 'focal': 1.,
                       'cls': cfg.TRAIN.CE_WEIGHT, 'offset_l1': cfg.TRAIN.OFFSET_L1_WEIGHT,
                       'offset_cosine': cfg.TRAIN.OFFSET_COSINE_WEIGHT,'simclr':cfg.TRAIN.SIMCLR_WEIGHT}

        if settings.script_name == "mcitrack_h_deform_online_v4":
            actor = MCITrack_H_Deform_Online_V4_Actor(net=net, objective=objective, loss_weight=loss_weight,
                                               settings=settings, cfg=cfg)
    elif settings.script_name in ['odonet_v1', 'odonet_wyq_v1', 'odonet_hn_v1', 'odonet_hn_v2']:
        focal_loss = FocalLoss()
        objective = {'giou': giou_loss, 'l1': l1_loss, 'focal': focal_loss, 'cls': BCEWithLogitsLoss(),
                     'offset_l1': masked_offset_l1_loss, 'offset_cosine': masked_cosine_similarity_loss,
                     'pos_l1': compute_offset_supervision_loss, 'simclr': multi_view_contrastive_loss_matrix_tokenwise}
        loss_weight = {'giou': cfg.TRAIN.GIOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT, 'focal': 1.,
                       'cls': cfg.TRAIN.CE_WEIGHT, 'offset_l1': cfg.TRAIN.OFFSET_L1_WEIGHT,
                       'pos_l1': cfg.TRAIN.POS_L1_WEIGHT,
                       'offset_cosine': cfg.TRAIN.OFFSET_COSINE_WEIGHT, 'simclr': cfg.TRAIN.SIMCLR_WEIGHT}

        if settings.script_name in ["odonet_v1", "odonet_wyq_v1"]:
            actor = ODONET_V1_Actor(net=net, objective=objective, loss_weight=loss_weight,
                                                      settings=settings, cfg=cfg)
        elif settings.script_name == "odonet_hn_v1":
            actor = ODONET_HN_V1_Actor(net=net, objective=objective, loss_weight=loss_weight,
                                                      settings=settings, cfg=cfg)
        elif settings.script_name == "odonet_hn_v2":
            actor = ODONET_HN_V2_Actor(net=net, objective=objective, loss_weight=loss_weight,
                                                      settings=settings, cfg=cfg)
    elif settings.script_name in ['odonet_v1_odop']:
        focal_loss = FocalLoss()
        objective = {'giou': giou_loss, 'l1': l1_loss, 'focal': focal_loss, 'cls': BCEWithLogitsLoss(),
                     'offset_l1': masked_offset_l1_loss, 'offset_cosine': masked_cosine_similarity_loss,
                     'pos_l1': compute_offset_supervision_loss, 'simclr': multi_view_contrastive_loss_matrix_tokenwise}
        loss_weight = {'giou': cfg.TRAIN.GIOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT, 'focal': 1.,
                       'cls': cfg.TRAIN.CE_WEIGHT, 'offset_l1': cfg.TRAIN.OFFSET_L1_WEIGHT,
                       'pos_l1': cfg.TRAIN.POS_L1_WEIGHT,
                       'offset_cosine': cfg.TRAIN.OFFSET_COSINE_WEIGHT, 'simclr': cfg.TRAIN.SIMCLR_WEIGHT}

        if settings.script_name == "odonet_v1_odop":
            actor = ODONET_V1_ODOP_Actor(net=net, objective=objective, loss_weight=loss_weight,
                                                      settings=settings, cfg=cfg)
    elif settings.script_name in ['odonet_v1_pcgrad']:
        focal_loss = FocalLoss()
        objective = {'giou': giou_loss, 'l1': l1_loss, 'focal': focal_loss, 'cls': BCEWithLogitsLoss(),
                     'offset_l1': masked_offset_l1_loss, 'offset_cosine': masked_cosine_similarity_loss,
                     'pos_l1': compute_offset_supervision_loss, 'simclr': multi_view_contrastive_loss_matrix_tokenwise}
        loss_weight = {'giou': cfg.TRAIN.GIOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT, 'focal': 1.,
                       'cls': cfg.TRAIN.CE_WEIGHT, 'offset_l1': cfg.TRAIN.OFFSET_L1_WEIGHT,
                       'pos_l1': cfg.TRAIN.POS_L1_WEIGHT,
                       'offset_cosine': cfg.TRAIN.OFFSET_COSINE_WEIGHT, 'simclr': cfg.TRAIN.SIMCLR_WEIGHT}

        if settings.script_name == "odonet_v1_pcgrad":
            actor = ODONET_V1_PCGrad_Actor(net=net, objective=objective, loss_weight=loss_weight,
                                                      settings=settings, cfg=cfg)
    elif settings.script_name in ['chotrack_v1']:
        focal_loss = FocalLoss()
        objective = {'giou': giou_loss, 'l1': l1_loss, 'focal': focal_loss, 'cls': BCEWithLogitsLoss(),
                     'offset_l1': masked_offset_l1_loss, 'offset_cosine': masked_cosine_similarity_loss,
                     'pos_l1': compute_offset_supervision_loss, 'simclr': multi_view_contrastive_loss_matrix_tokenwise}
        loss_weight = {'giou': cfg.TRAIN.GIOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT, 'focal': 1.,
                       'cls': cfg.TRAIN.CE_WEIGHT, 'offset_l1': cfg.TRAIN.OFFSET_L1_WEIGHT,
                       'pos_l1': cfg.TRAIN.POS_L1_WEIGHT,
                       'offset_cosine': cfg.TRAIN.OFFSET_COSINE_WEIGHT, 'simclr': cfg.TRAIN.SIMCLR_WEIGHT}

        if settings.script_name == "chotrack_v1":
            actor = CHOTrack_V1_Actor(net=net, objective=objective, loss_weight=loss_weight,
                                    settings=settings, cfg=cfg)
    elif settings.script_name in ['chotrack_v2']:
        focal_loss = FocalLoss()
        objective = {'giou': giou_loss, 'l1': l1_loss, 'focal': focal_loss, 'cls': BCEWithLogitsLoss(),
                     'offset_l1': masked_offset_l1_loss, 'offset_cosine': masked_cosine_similarity_loss,
                     'pos_l1': compute_offset_supervision_loss, 'trackingInfoNCE': TrackingInfoNCELoss(temperature=0.07).cuda()}
        loss_weight = {'giou': cfg.TRAIN.GIOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT, 'focal': 1.,
                       'cls': cfg.TRAIN.CE_WEIGHT, 'offset_l1': cfg.TRAIN.OFFSET_L1_WEIGHT,
                       'pos_l1': cfg.TRAIN.POS_L1_WEIGHT,
                       'offset_cosine': cfg.TRAIN.OFFSET_COSINE_WEIGHT, 'trackingInfoNCE': cfg.TRAIN.SIMCLR_WEIGHT}

        if settings.script_name == "chotrack_v2":
            actor = CHOTrack_V2_Actor(net=net, objective=objective, loss_weight=loss_weight,
                                    settings=settings, cfg=cfg)
    elif settings.script_name in ['chotrack_v3']:
        focal_loss = FocalLoss()
        objective = {'giou': giou_loss, 'l1': l1_loss, 'focal': focal_loss, 'cls': BCEWithLogitsLoss(),
                     'offset_l1': masked_offset_l1_loss, 'offset_cosine': masked_cosine_similarity_loss,
                     'pos_l1': compute_offset_supervision_loss, 'trackingInfoNCE': TrackingInfoNCELoss(temperature=0.07).cuda()}
        loss_weight = {'giou': cfg.TRAIN.GIOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT, 'focal': 1.,
                       'cls': cfg.TRAIN.CE_WEIGHT, 'offset_l1': cfg.TRAIN.OFFSET_L1_WEIGHT,
                       'pos_l1': cfg.TRAIN.POS_L1_WEIGHT,
                       'offset_cosine': cfg.TRAIN.OFFSET_COSINE_WEIGHT, 'trackingInfoNCE': cfg.TRAIN.SIMCLR_WEIGHT}

        if settings.script_name == "chotrack_v3":
            actor = CHOTrack_V3_Actor(net=net, objective=objective, loss_weight=loss_weight,
                                    settings=settings, cfg=cfg)
    elif settings.script_name in ['chotrack_v4']:
        focal_loss = FocalLoss()
        objective = {'giou': giou_loss, 'l1': l1_loss, 'focal': focal_loss, 'cls': BCEWithLogitsLoss(),
                     'offset_l1': masked_offset_l1_loss, 'offset_cosine': masked_cosine_similarity_loss,
                     'pos_l1': compute_offset_supervision_loss, 'trackingInfoNCE': TrackingInfoNCELoss(temperature=0.07).cuda()}
        loss_weight = {'giou': cfg.TRAIN.GIOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT, 'focal': 1.,
                       'cls': cfg.TRAIN.CE_WEIGHT, 'offset_l1': cfg.TRAIN.OFFSET_L1_WEIGHT,
                       'pos_l1': cfg.TRAIN.POS_L1_WEIGHT,
                       'offset_cosine': cfg.TRAIN.OFFSET_COSINE_WEIGHT, 'trackingInfoNCE': cfg.TRAIN.SIMCLR_WEIGHT}

        if settings.script_name == "chotrack_v4":
            actor = CHOTrack_V4_Actor(net=net, objective=objective, loss_weight=loss_weight,
                                    settings=settings, cfg=cfg)
    elif settings.script_name in ['chotrack_v5']:
        focal_loss = FocalLoss()
        objective = {'giou': giou_loss, 'l1': l1_loss, 'focal': focal_loss, 'cls': BCEWithLogitsLoss(),
                     'offset_l1': masked_offset_l1_loss, 'offset_cosine': masked_cosine_similarity_loss,
                     'pos_l1': compute_offset_supervision_loss,
                     'trackingInfoNCE': TrackingInfoNCELoss(temperature=0.07).cuda()}
        loss_weight = {'giou': cfg.TRAIN.GIOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT, 'focal': 1.,
                       'cls': cfg.TRAIN.CE_WEIGHT, 'offset_l1': cfg.TRAIN.OFFSET_L1_WEIGHT,
                       'pos_l1': cfg.TRAIN.POS_L1_WEIGHT,
                       'offset_cosine': cfg.TRAIN.OFFSET_COSINE_WEIGHT, 'trackingInfoNCE': cfg.TRAIN.SIMCLR_WEIGHT}

        if settings.script_name == "chotrack_v5":
            actor = CHOTrack_V5_Actor(net=net, objective=objective, loss_weight=loss_weight,
                                      settings=settings, cfg=cfg)
    elif settings.script_name in ['chotrack_v6']:
        focal_loss = FocalLoss()
        objective = {'giou': giou_loss, 'l1': l1_loss, 'focal': focal_loss, 'cls': BCEWithLogitsLoss(),
                     'offset_l1': masked_offset_l1_loss, 'offset_cosine': masked_cosine_similarity_loss,
                     'pos_l1': compute_offset_supervision_loss,
                     'trackingInfoNCE': TrackingInfoNCELoss(temperature=0.07).cuda()}
        loss_weight = {'giou': cfg.TRAIN.GIOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT, 'focal': 1.,
                       'cls': cfg.TRAIN.CE_WEIGHT, 'offset_l1': cfg.TRAIN.OFFSET_L1_WEIGHT,
                       'pos_l1': cfg.TRAIN.POS_L1_WEIGHT,
                       'offset_cosine': cfg.TRAIN.OFFSET_COSINE_WEIGHT, 'trackingInfoNCE': cfg.TRAIN.SIMCLR_WEIGHT}

        if settings.script_name == "chotrack_v6":
            actor = CHOTrack_V6_Actor(net=net, objective=objective, loss_weight=loss_weight,
                                      settings=settings, cfg=cfg)
    elif settings.script_name in ['chotrack_v7']:
        focal_loss = FocalLoss()
        objective = {'giou': giou_loss, 'l1': l1_loss, 'focal': focal_loss, 'cls': BCEWithLogitsLoss(),
                     'offset_l1': masked_offset_l1_loss, 'offset_cosine': masked_cosine_similarity_loss,
                     'pos_l1': compute_offset_supervision_loss,
                     'trackingInfoNCE': TrackingInfoNCELoss(temperature=0.07).cuda()}
        loss_weight = {'giou': cfg.TRAIN.GIOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT, 'focal': 1.,
                       'cls': cfg.TRAIN.CE_WEIGHT, 'offset_l1': cfg.TRAIN.OFFSET_L1_WEIGHT,
                       'pos_l1': cfg.TRAIN.POS_L1_WEIGHT,
                       'offset_cosine': cfg.TRAIN.OFFSET_COSINE_WEIGHT, 'trackingInfoNCE': cfg.TRAIN.SIMCLR_WEIGHT}

        if settings.script_name == "chotrack_v7":
            actor = CHOTrack_V7_Actor(net=net, objective=objective, loss_weight=loss_weight,
                                      settings=settings, cfg=cfg)
    elif settings.script_name in ['chotrack_v8','chotrack_v9','chotrack_v10','chotrack_v11',]:
        focal_loss = FocalLoss()
        objective = {'giou': giou_loss, 'l1': l1_loss, 'focal': focal_loss, 'cls': BCEWithLogitsLoss(),
                     'offset_l1': masked_offset_l1_loss, 'offset_cosine': masked_cosine_similarity_loss,
                     'pos_l1': compute_offset_supervision_loss,
                     'causal':getattr(causal_former_loss,'masked_mse_torch'),
                     'trackingInfoNCE': TrackingInfoNCELoss(temperature=getattr(cfg.TRAIN, "NCE_TEMPERATURE", 0.07)).cuda()}
        loss_weight = {'giou': cfg.TRAIN.GIOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT, 'focal': 1.,
                       'cls': cfg.TRAIN.CE_WEIGHT, 'offset_l1': cfg.TRAIN.OFFSET_L1_WEIGHT,
                       'pos_l1': cfg.TRAIN.POS_L1_WEIGHT,
                       'causal': getattr(cfg.TRAIN, "CAUSAL_WEIGHT", 1.0),
                       'causal_reg': getattr(cfg.TRAIN, "CAUSAL_REG_WEIGHT", 0.0),
                       'offset_cosine': cfg.TRAIN.OFFSET_COSINE_WEIGHT, 'trackingInfoNCE': cfg.TRAIN.SIMCLR_WEIGHT}

        if settings.script_name == "chotrack_v8":
            actor = CHOTrack_V8_Actor(net=net, objective=objective, loss_weight=loss_weight,
                                      settings=settings, cfg=cfg)
        elif settings.script_name == "chotrack_v9":
            actor = CHOTrack_V9_Actor(net=net, objective=objective, loss_weight=loss_weight,
                                      settings=settings, cfg=cfg)
        elif settings.script_name == "chotrack_v10":
            actor = CHOTrack_V10_Actor(net=net, objective=objective, loss_weight=loss_weight,
                                      settings=settings, cfg=cfg)
        elif settings.script_name == "chotrack_v11":
            actor = CHOTrack_V11_Actor(net=net, objective=objective, loss_weight=loss_weight,
                                      settings=settings, cfg=cfg)

    elif settings.script_name in ['chotrack_v12']:
        focal_loss = FocalLoss()
        objective = {'giou': giou_loss, 'l1': l1_loss, 'focal': focal_loss, 'cls': BCEWithLogitsLoss(),
                     'offset_l1': masked_offset_l1_loss, 'offset_cosine': masked_cosine_similarity_loss,
                     'pos_l1': compute_offset_supervision_loss,
                     'causal':getattr(causal_former_loss,'masked_mse_torch'),
                     'trackingInfoNCE': TrackingInfoNCELoss(temperature=getattr(cfg.TRAIN, "NCE_TEMPERATURE", 0.07)).cuda(),
                     'causal_regularization_loss':compute_causal_regularization_loss,
                     'causal_input_nce_loss':compute_causal_input_nce_loss
                     }
        loss_weight = {'giou': cfg.TRAIN.GIOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT, 'focal': 1.,
                       'cls': cfg.TRAIN.CE_WEIGHT, 'offset_l1': cfg.TRAIN.OFFSET_L1_WEIGHT,
                       'pos_l1': cfg.TRAIN.POS_L1_WEIGHT,
                       'causal': getattr(cfg.TRAIN, "CAUSAL_WEIGHT", 1.0),
                       'causal_reg': getattr(cfg.TRAIN, "CAUSAL_REG_WEIGHT", 0.0),
                       'offset_cosine': cfg.TRAIN.OFFSET_COSINE_WEIGHT, 'trackingInfoNCE': cfg.TRAIN.SIMCLR_WEIGHT}

        if settings.script_name == "chotrack_v12":
            actor = CHOTrack_V12_Actor(net=net, objective=objective, loss_weight=loss_weight,
                                      settings=settings, cfg=cfg)

    elif settings.script_name in ['chotrack_v13', 'chotrack_v14', 'chotrack_v15', 'ovtrack_v1']:
        focal_loss = FocalLoss()
        objective = {'giou': giou_loss, 'l1': l1_loss, 'focal': focal_loss, 'cls': BCEWithLogitsLoss(),
                     'offset_l1': masked_offset_l1_loss, 'offset_cosine': masked_cosine_similarity_loss,
                     'pos_l1': compute_offset_supervision_loss,
                     'causal':getattr(causal_former_loss,'masked_mse_torch')
                     }
        loss_weight = {'giou': cfg.TRAIN.GIOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT, 'focal': 1.,
                       'cls': cfg.TRAIN.CE_WEIGHT, 'offset_l1': cfg.TRAIN.OFFSET_L1_WEIGHT,
                       'pos_l1': cfg.TRAIN.POS_L1_WEIGHT,
                       'causal': getattr(cfg.TRAIN, "CAUSAL_WEIGHT", 1.0),
                       'partial_l1': getattr(cfg.TRAIN, "PARTIAL_L1_WEIGHT", 0.0),
                       'partial_giou': getattr(cfg.TRAIN, "PARTIAL_GIOU_WEIGHT", 0.0),
                       'offset_cosine': cfg.TRAIN.OFFSET_COSINE_WEIGHT}

        if settings.script_name == 'ovtrack_v1':
            actor_cls = OVTrack_V1_Actor
        elif settings.script_name == 'chotrack_v14':
            actor_cls = CHOTrack_V14_Actor
        elif settings.script_name == 'chotrack_v15':
            actor_cls = CHOTrack_V15_Actor
        else:
            actor_cls = CHOTrack_V13_Actor
        actor = actor_cls(net=net, objective=objective, loss_weight=loss_weight,
                          settings=settings, cfg=cfg)


    else:
        raise ValueError("illegal script name")

    # Optimizer, parameters, and learning rates
    optimizer, lr_scheduler = get_optimizer_scheduler(net, cfg)
    use_amp = getattr(cfg.TRAIN, "AMP", False)

    if settings.script_name=="mcitrack_ttt":
        trainer = TestTrainer(actor, [loader_train], optimizer, settings, lr_scheduler, use_amp=use_amp)
    elif settings.script_name in ['mcitrack_h_deform_online_v2','mcitrack_h_deform_online_v4']:
        trainer = LTRTrainer(actor, [loader_train,loader_val], optimizer, settings, lr_scheduler, use_amp=use_amp)
    elif settings.script_name in ['odonet_v1_pcgrad']:
        trainer = LTR_PCGrad_Trainer(actor, [loader_train], optimizer, settings, lr_scheduler, use_amp=use_amp)
    else:
        trainer = LTRTrainer(actor, [loader_train], optimizer, settings, lr_scheduler, use_amp=use_amp)
    # train process
    trainer.train(cfg.TRAIN.EPOCH, load_latest=True, fail_safe=True)
