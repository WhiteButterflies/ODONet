"""Copy from ARcm_coco_seg_only_mask_384"""
# import built-in library
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms
from torch.utils.data.distributed import DistributedSampler
# import our coded modules
# from ltr.dataset import Youtube_VOS, Saliency, MSCOCOSeq17, Got10k_mask
from ltr.dataset import MSCOCOSeq17_lmdb, Saliency_lmdb, Youtube_VOS_lmdb, Got10k_mask_lmdb
from ltr.data import SEprocessing, SEsampler, LTRLoader
import ltr.data.transforms as dltransforms
from ltr import actors
from ltr.trainers import LTRTrainer
# import ltr.models.AR_seg.ARcm_seg as ARcm_seg
import ltr.models.AR_seg_mask.AR_seg_mask as AR_seg_mask


def run(settings):
    # Most common settings are assigned in the settings struct
    settings.batch_size = 32  # Batch size
    settings.search_area_factor = 2.0  # Image patch size relative to target size
    settings.feature_sz = 24  # Size of feature map
    settings.output_sz = settings.feature_sz * 16  # Size of input image patches
    settings.used_layers = ['layer4']  # 2020.4.14 change name
    # Settings for the image sample and proposal generation
    settings.center_jitter_factor = {'train': 0, 'test': 0.25}
    settings.scale_jitter_factor = {'train': 0, 'test': 0.25}
    settings.max_gap = 50
    settings.sample_per_epoch = 512000  #
    '''others'''
    settings.print_interval = 100  # How often to print loss and other info
    settings.num_workers = 8  # Number of workers for image loading
    settings.normalize_mean = [0.485, 0.456, 0.406]  # Normalize mean (default pytorch ImageNet values)
    settings.normalize_std = [0.229, 0.224, 0.225]  # Normalize std (default pytorch ImageNet values)

    '''##### Prepare data for training and validation #####'''
    transform_train = torchvision.transforms.Compose([dltransforms.ToTensorAndJitter(0.2),
                                                      torchvision.transforms.Normalize(mean=settings.normalize_mean,
                                                                                       std=settings.normalize_std)])
    # Data processing to do on the training pairs
    '''Data_process class. In SEMaskProcessing, we use zero-padding for images and masks.'''
    data_processing_train = SEprocessing.SEMaskProcessing(search_area_factor=settings.search_area_factor,
                                                          output_sz=settings.output_sz,
                                                          center_jitter_factor=settings.center_jitter_factor,
                                                          scale_jitter_factor=settings.scale_jitter_factor,
                                                          mode='sequence',
                                                          transform=transform_train)
    # Train datasets
    # mask datasets

    coco17_train = MSCOCOSeq17_lmdb('train')
    saliency = Saliency_lmdb()
    youtube_vos = Youtube_VOS_lmdb()
    got10k_mask = Got10k_mask_lmdb(split="vottrain")
    # coco17_train = MSCOCOSeq17('train')
    # saliency = Saliency()
    # youtube_vos = Youtube_VOS()
    # got10k_mask = Got10k_mask(split="vottrain")

    # The sampler for training
    '''Build training dataset. focus "__getitem__" and "__len__"'''
    dataset_train = SEsampler.SEMaskSampler([youtube_vos, saliency, coco17_train, got10k_mask],
                                            [1, 1, 1, 1],
                                            samples_per_epoch=settings.sample_per_epoch,
                                            max_gap=settings.max_gap,
                                            processing=data_processing_train)

    # The loader for training
    '''using distributed sampler'''
    train_sampler = DistributedSampler(dataset_train)
    '''"sampler" is exclusive with "shuffle"'''
    loader_train = LTRLoader('train', dataset_train, training=True, batch_size=settings.batch_size,
                             num_workers=settings.num_workers,
                             drop_last=True, stack_dim=1, sampler=train_sampler, pin_memory=False)

    '''2. build validation dataset and dataloader'''
    coco17_val = MSCOCOSeq17_lmdb('val')
    # coco17_val = MSCOCOSeq17('val')
    # # The augmentation transform applied to the validation set (individually to each image in the pair)
    transform_val = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Normalize(mean=settings.normalize_mean,
                                                                                     std=settings.normalize_std)])
    # Data processing to do on the validation pairs
    data_processing_val = SEprocessing.SEMaskProcessing(search_area_factor=settings.search_area_factor,
                                                        output_sz=settings.output_sz,
                                                        center_jitter_factor=settings.center_jitter_factor,
                                                        scale_jitter_factor=settings.scale_jitter_factor,
                                                        mode='sequence',
                                                        transform=transform_val)
    # The sampler for validation
    dataset_val = SEsampler.SEMaskSampler([coco17_val], [1], samples_per_epoch=500 * settings.batch_size, max_gap=50,
                                          processing=data_processing_val)
    # The loader for validation
    loader_val = LTRLoader('val', dataset_val, training=False, batch_size=settings.batch_size,
                           num_workers=settings.num_workers,
                           shuffle=False, drop_last=True, epoch_interval=5, stack_dim=1)

    '''##### prepare network and other stuff for optimization #####'''
    # Create network (target size is half of the feature sz)
    net = AR_seg_mask.ARnet_seg_mask_resnet50(backbone_pretrained=True, used_layers=settings.used_layers,
                                              target_sz=int(settings.feature_sz / 2))
    # wrap network to distributed one
    net.cuda()
    net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[settings.local_rank], find_unused_parameters=True)
    # Set objective
    objective = {'mask': nn.BCELoss()}

    # Create actor, which wraps network and objective
    actor = actors.ARmask_Actor(net=net, objective=objective)

    # Optimizer
    optimizer = optim.Adam(net.parameters(), lr=1e-3)

    # Learning rate scheduler
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

    # Create trainer
    trainer = LTRTrainer(actor, [loader_train, loader_val], optimizer, settings, lr_scheduler)
    # Run training (set fail_safe=False if you are debugging)
    trainer.train(80, load_latest=True, fail_safe=False)
