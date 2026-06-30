import os
import os.path
import torch
import numpy as np
import pandas
import csv
import random
from collections import OrderedDict
from .base_dataset import BaseDataset
from ltr.data.image_loader import default_image_loader
from ltr.admin.environment import env_settings
import cv2
from lib.utils.lmdb_utils import *


class Lasot_mask_lmdb(BaseDataset):
    """ LaSOT dataset with masks"""

    def __init__(self, root=None, image_loader=default_image_loader, vid_ids=None, split=None, data_fraction=None):
        """
        args:
            root - path to the lasot dataset.
            image_loader (default_image_loader) -  The function to read the images. If installed,
                                                   jpeg4py (https://github.com/ajkxyz/jpeg4py) is used by default. Else,
                                                   opencv's imread is used.
            vid_ids - List containing the ids of the videos (1 - 20) used for training. If vid_ids = [1, 3, 5], then the
                    videos with subscripts -1, -3, and -5 from each class will be used for training.
            split - If split='train', the official train split (protocol-II) is used for training. Note: Only one of
                    vid_ids or split option can be used at a time.
            data_fraction (None) - Fraction of videos to be used. The videos are selected randomly. If None, all the
                                   videos will be used
        """
        print("building lasot mask dataset from lmdb")
        root = env_settings().lasot_lmdb_dir if root is None else root
        super().__init__(root, image_loader)
        self.mask_root = env_settings().lasot_mask_lmdb_dir

        self.sequence_list = self._build_sequence_list(vid_ids, split)

        if data_fraction is not None:
            self.sequence_list = random.sample(self.sequence_list, int(len(self.sequence_list) * data_fraction))

    def _build_sequence_list(self, vid_ids=None, split=None):
        if split is not None:
            if vid_ids is not None:
                raise ValueError('Cannot set both split_name and vid_ids.')
            ltr_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
            if split == 'train':
                file_path = os.path.join(ltr_path, 'data_specs', 'lasot_train_split.txt')
            elif split == 'test':
                file_path = os.path.join(ltr_path, 'data_specs', 'lasot_test_split.txt')
            else:
                raise ValueError('Unknown split name.')
            sequence_list = pandas.read_csv(file_path, header=None, squeeze=True).values.tolist()
        elif vid_ids is not None:
            sequence_list = [c + '-' + str(v) for c in self.class_list for v in vid_ids]
        else:
            raise ValueError('Set either split_name or vid_ids.')

        return sequence_list

    def get_name(self):
        return 'lasot_mask_lmdb'

    def get_num_sequences(self):
        return len(self.sequence_list)

    def _read_bb_anno(self, seq_path):
        bb_anno_file = os.path.join(seq_path, "groundtruth.txt")
        gt_str_list = decode_str(self.root, bb_anno_file).split('\n')[:-1]  # the last line is empty
        gt_list = [list(map(float, line.split(','))) for line in gt_str_list]
        gt_arr = np.array(gt_list).astype(np.float32)
        return torch.tensor(gt_arr)

    def _read_target_visible(self, seq_path):
        # Read full occlusion and out_of_view
        occlusion_file = os.path.join(seq_path, "full_occlusion.txt")
        out_of_view_file = os.path.join(seq_path, "out_of_view.txt")
        # we also consider whether the mask is valid
        mask_valid_file = os.path.join(seq_path, "valid.txt")

        occ_list = list(map(int, decode_str(self.root, occlusion_file).split(',')))
        occlusion = torch.ByteTensor(occ_list)
        out_view_list = list(map(int, decode_str(self.root, out_of_view_file).split(',')))
        out_of_view = torch.ByteTensor(out_view_list)
        mask_valid_list = list(map(int, decode_str(self.mask_root, mask_valid_file).split('\n')[:-1]))
        mask_valid = torch.ByteTensor(mask_valid_list)

        target_visible = ~occlusion & ~out_of_view & mask_valid

        return target_visible

    def _get_sequence_path(self, seq_id):
        seq_name = self.sequence_list[seq_id]
        class_name = seq_name.split('-')[0]
        vid_id = seq_name.split('-')[1]

        return os.path.join(class_name, class_name + '-' + vid_id)

    def get_sequence_info(self, seq_id):
        seq_path = self._get_sequence_path(seq_id)
        bbox = self._read_bb_anno(seq_path)

        valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        visible = self._read_target_visible(seq_path) & valid.byte()

        return {'bbox': bbox, 'valid': valid, 'visible': visible}

    def _get_frame_path(self, seq_path, frame_id):
        return os.path.join(seq_path, 'img', '{:08}.jpg'.format(frame_id + 1))  # frames start from 1

    def _get_frame(self, seq_path, frame_id):
        return decode_img(self.root, self._get_frame_path(seq_path, frame_id))

    def _get_class(self, seq_path):
        obj_class = seq_path.split('/')[-2]
        return obj_class

    def _get_mask(self, seq_id, frame_id):
        seq_name = self.sequence_list[seq_id]
        class_name = seq_name.split('-')[0]
        vid_id = seq_name.split('-')[1]
        mask_path = os.path.join(class_name, class_name + '-' + vid_id, "%08d.jpg" % (frame_id + 1))
        mask = cv2.cvtColor(decode_img(self.mask_root, mask_path), cv2.COLOR_RGB2GRAY)
        mask_img = mask[..., np.newaxis]  # (H,W,1)
        mask_ins = (mask_img == 255).astype(np.uint8)  # binary mask # (H,W,1)
        return mask_ins

    def has_mask(self):
        return True

    def get_frames(self, seq_id, frame_ids, anno=None):
        seq_path = self._get_sequence_path(seq_id)

        obj_class = self._get_class(seq_path)
        frame_list = [self._get_frame(seq_path, f_id) for f_id in frame_ids]
        mask_list = [self._get_mask(seq_id, f_id) for f_id in frame_ids]
        if anno is None:
            anno = self.get_sequence_info(seq_id)

        # Create anno dict
        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]

        object_meta = OrderedDict({'object_class': obj_class,
                                   'motion_class': None,
                                   'major_class': None,
                                   'root_class': None,
                                   'motion_adverb': None})

        return frame_list, mask_list, anno_frames, object_meta
