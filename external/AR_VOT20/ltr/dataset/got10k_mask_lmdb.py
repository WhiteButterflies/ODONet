import os
import os.path
import numpy as np
import torch
import csv
import pandas
import random
from collections import OrderedDict
from .base_dataset import BaseDataset
from ltr.data.image_loader import default_image_loader
from ltr.admin.environment import env_settings
import cv2
from lib.utils.lmdb_utils import *


class Got10k_mask_lmdb(BaseDataset):
    """ GOT-10k mask dataset lmdb"""

    def __init__(self, root=None, image_loader=default_image_loader, split=None, seq_ids=None, data_fraction=None):
        """
        args:
            root - path to the got-10k training data. Note: This should point to the 'train' folder inside GOT-10k
            image_loader (default_image_loader) -  The function to read the images. If installed,
                                                   jpeg4py (https://github.com/ajkxyz/jpeg4py) is used by default. Else,
                                                   opencv's imread is used.
            split - 'train' or 'val'. Note: The validation split here is a subset of the official got-10k train split,
                    not NOT the official got-10k validation split. To use the official validation split, provide that as
                    the root folder instead.
            seq_ids - List containing the ids of the videos to be used for training. Note: Only one of 'split' or 'seq_ids'
                        options can be used at the same time.
            data_fraction (None) - Fraction of videos to be used. The videos are selected randomly. If None, all the videos
                                   will be used
        """
        print("building got10k mask dataset from lmdb")
        root = env_settings().got10k_lmdb_dir if root is None else root
        super().__init__(root, image_loader)
        self.mask_root = env_settings().got10k_mask_lmdb_dir

        # all folders inside the root
        self.sequence_list = self._get_sequence_list()

        # seq_id is the index of the folder inside the got10k root path
        if split is not None:
            if seq_ids is not None:
                raise ValueError('Cannot set both split_name and seq_ids.')
            ltr_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
            if split == 'train':
                file_path = os.path.join(ltr_path, 'data_specs', 'got10k_train_split.txt')
            elif split == 'val':
                file_path = os.path.join(ltr_path, 'data_specs', 'got10k_val_split.txt')
            elif split == 'vottrain':
                file_path = os.path.join(ltr_path, 'data_specs', 'got10k_vot_train_split.txt')
            elif split == 'votval':
                file_path = os.path.join(ltr_path, 'data_specs', 'got10k_vot_val_split.txt')
            else:
                raise ValueError('Unknown split name.')
            seq_ids = pandas.read_csv(file_path, header=None, squeeze=True, dtype=np.int64).values.tolist()
        elif seq_ids is None:
            seq_ids = list(range(0, len(self.sequence_list)))
        # self.seq_ids = seq_ids

        self.sequence_list = [self.sequence_list[i] for i in seq_ids]  # original
        # self.sequence_list = [self.sequence_list[i-1] for i in seq_ids] # my changed version

        if data_fraction is not None:
            self.sequence_list = random.sample(self.sequence_list, int(len(self.sequence_list) * data_fraction))

        self.sequence_meta_info = self._load_meta_info()

    def get_name(self):
        return 'got10k_mask_lmdb'

    def _load_meta_info(self):
        def _read_meta(meta_info):

            object_meta = OrderedDict({'object_class_name': meta_info[5].split(': ')[-1],
                                       'motion_class': meta_info[6].split(': ')[-1],
                                       'major_class': meta_info[7].split(': ')[-1],
                                       'root_class': meta_info[8].split(': ')[-1],
                                       'motion_adverb': meta_info[9].split(': ')[-1]})

            return object_meta
        sequence_meta_info = {}
        for s in self.sequence_list:
            try:
                meta_str = decode_str(self.root, "train/%s/meta_info.ini" %s)
                sequence_meta_info[s] = _read_meta(meta_str.split('\n'))
            except:
                sequence_meta_info[s] = OrderedDict({'object_class_name': None,
                                                     'motion_class': None,
                                                     'major_class': None,
                                                     'root_class': None,
                                                     'motion_adverb': None})
        return sequence_meta_info

    def _get_sequence_list(self):
        dir_str = decode_str(self.root, 'train/list.txt')
        dir_list = dir_str.split('\n')
        return dir_list

    def _read_bb_anno(self, seq_path):
        bb_anno_file = os.path.join(seq_path, "groundtruth.txt")
        gt_str_list = decode_str(self.root, bb_anno_file).split('\n')[:-1]  # the last line in got10k is empty
        gt_list = [list(map(float, line.split(','))) for line in gt_str_list]
        gt_arr = np.array(gt_list).astype(np.float32)

        return torch.tensor(gt_arr)

    def _read_target_visible(self, seq_path):
        # full occlusion and out_of_view files
        occlusion_file = os.path.join(seq_path, "absence.label")
        cover_file = os.path.join(seq_path, "cover.label")
        # we also consider whether the mask is valid
        mask_valid_file = os.path.join(seq_path, "valid.txt")
        # Read these files
        occ_list = list(map(int, decode_str(self.root, occlusion_file).split('\n')[:-1]))  # the last line in got10k is empty
        occlusion = torch.ByteTensor(occ_list)
        cover_list = list(map(int, decode_str(self.root, cover_file).split('\n')[:-1]))  # the last line in got10k is empty
        cover = torch.ByteTensor(cover_list)
        mask_valid_list = list(map(int, decode_str(self.mask_root, mask_valid_file).split('\n')[:-1]))
        mask_valid = torch.ByteTensor(mask_valid_list)

        target_visible = ~occlusion & (cover>0).byte() & mask_valid

        visible_ratio = cover.float() / 8
        return target_visible, visible_ratio

    def _get_sequence_path(self, seq_id):
        return os.path.join("train", self.sequence_list[seq_id])

    def get_sequence_info(self, seq_id):
        seq_path = self._get_sequence_path(seq_id)
        bbox = self._read_bb_anno(seq_path)

        valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        visible, visible_ratio = self._read_target_visible(seq_path)
        visible = visible & valid.byte()

        return {'bbox': bbox, 'valid': valid, 'visible': visible, 'visible_ratio': visible_ratio}

    def _get_frame_path(self, seq_path, frame_id):
        return os.path.join(seq_path, '{:08}.jpg'.format(frame_id + 1))  # frames start from 1

    def _get_frame(self, seq_path, frame_id):
        return decode_img(self.root, self._get_frame_path(seq_path, frame_id))

    def _get_mask(self, seq_id, frame_id):
        mask_img_path = os.path.join("train", self.sequence_list[seq_id], "%08d.jpg" % (frame_id + 1))
        mask = cv2.cvtColor(decode_img(self.mask_root, mask_img_path), cv2.COLOR_RGB2GRAY)
        mask_img = mask[..., np.newaxis]  # (H, W, 1)
        mask_bin = (mask_img == 255).astype(np.uint8)  # binary mask # (H, W, 1)
        return mask_bin

    def has_mask(self):
        return True

    def get_frames(self, seq_id, frame_ids, anno=None):
        seq_path = self._get_sequence_path(seq_id)
        obj_meta = self.sequence_meta_info[self.sequence_list[seq_id]]

        frame_list = [self._get_frame(seq_path, f_id) for f_id in frame_ids]
        mask_list = [self._get_mask(seq_id, f_id) for f_id in frame_ids]
        if anno is None:
            anno = self.get_sequence_info(seq_id)

        # Create anno dict
        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]

        return frame_list, mask_list, anno_frames, obj_meta
