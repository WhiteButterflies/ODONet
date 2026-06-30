import os
from .base_dataset import BaseDataset
from ltr.data.image_loader import default_image_loader
import json
import torch
from ltr.admin.environment import env_settings

'''newly added'''
import cv2
from os.path import join
import numpy as np
from lib.utils.lmdb_utils import decode_img, decode_json


def xyxy_to_xywh(xyxy):
    """Convert [x1 y1 x2 y2] box format to [x1 y1 w h] format."""
    if isinstance(xyxy, (list, tuple)):
        # Single box given as a list of coordinates
        assert len(xyxy) == 4
        x1, y1 = xyxy[0], xyxy[1]
        w = xyxy[2] - x1 + 1
        h = xyxy[3] - y1 + 1
        return (x1, y1, w, h)
    elif isinstance(xyxy, np.ndarray):
        # Multiple boxes given as a 2D ndarray
        return np.hstack((xyxy[:, 0:2], xyxy[:, 2:4] - xyxy[:, 0:2] + 1))
    else:
        raise TypeError('Argument xyxy must be a list, tuple, or numpy array.')


def polys_to_boxes(polys):
    """Convert a list of polygons into an array of tight bounding boxes."""
    boxes_from_polys = np.zeros((len(polys), 4), dtype=np.float32)
    for i in range(len(polys)):
        poly = polys[i]
        x0 = min(min(p[::2]) for p in poly)
        x1 = max(max(p[::2]) for p in poly)
        y0 = min(min(p[1::2]) for p in poly)
        y1 = max(max(p[1::2]) for p in poly)
        boxes_from_polys[i, :] = [x0, y0, x1, y1]
    return boxes_from_polys


class Instance(object):
    instID = 0
    pixelCount = 0

    def __init__(self, imgNp, instID):
        if (instID == 0):
            return
        self.instID = int(instID)  # 1
        self.pixelCount = int(self.getInstancePixels(imgNp, instID))  # 目标占据的像素个数

    def getInstancePixels(self, imgNp, instLabel):
        return (imgNp == instLabel).sum()

    def toDict(self):
        buildDict = {}
        buildDict["instID"] = self.instID
        buildDict["pixelCount"] = self.pixelCount
        return buildDict

    def __str__(self):
        return "(" + str(self.instID) + ")"


def get_target_to_image_ratio(seq):
    init_frame = seq[0]
    H, W = init_frame['h'], init_frame['w']
    # area = init_frame['area']
    bbox = init_frame['bbox']  # list length=4
    anno = torch.Tensor(bbox)
    img_sz = torch.Tensor([H, W])
    # return (area / (img_sz.prod())).sqrt()
    '''边界框面积与图像面积算比值,再开方'''
    return (anno[2:4].prod() / (img_sz.prod())).sqrt()


class Youtube_VOS_lmdb(BaseDataset):
    """ Youtube_VOS dataset.
    """

    def __init__(self, root=None, image_loader=default_image_loader, min_length=0, max_target_area=1):
        """
        args:
            root - path to the imagenet vid dataset.
            image_loader (default_image_loader) -  The function to read the images. If installed,
                                                   jpeg4py (https://github.com/ajkxyz/jpeg4py) is used by default. Else,
                                                   opencv's imread is used.
            min_length - Minimum allowed sequence length.
            max_target_area - max allowed ratio between target area and image area. Can be used to filter out targets
                                which cover complete image.
        """
        root = env_settings().youtubevos_lmdb_dir if root is None else root
        super().__init__(root, image_loader)
        print("building youtubevos from lmdb")
        self.sequence_list = decode_json(root, "cache.json")
        # Filter the sequences based on min_length and max_target_area in the first frame
        '''youtube-vos中某些视频的目标很大,以至于用边界框来框的话就直接是整张图,我暂时去掉了这些实例(数量不多,感觉影响不大)'''
        self.sequence_list = [x for x in self.sequence_list if len(x) >= min_length and
                              get_target_to_image_ratio(x) < max_target_area]
        # print(len(self.sequence_list))
        # for x in self.sequence_list:
        #     if len(x) < min_length:
        #         print('小于最小长度:',x)
        #     if get_target_to_image_ratio(x) >= max_target_area:
        #         print('超出图像范围',x[0]['file_name'])

    def get_name(self):
        return 'youtube_vos_lmdb'

    def has_mask(self):
        return True

    def get_num_sequences(self):
        return len(self.sequence_list)

    def get_sequence_info(self, seq_id):
        """根据seq_id得到被选中的视频的信息,其中visible属性最重要.它被用来判断当前视频是否可以拿来训练"""
        '''需要得到“每一帧是否valid”'''
        '''由于数据存在很多字典里,需要转换一下才行'''
        cur_seq = self.sequence_list[seq_id]
        bbox_list = []
        for idx, info_dict in enumerate(cur_seq):
            bbox_list.append(info_dict['bbox'])
        bbox_arr = np.array(bbox_list).astype(np.float32)  # (N,4)
        bbox = torch.from_numpy(bbox_arr)  # torch tensor (N,4)
        valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        visible = valid.clone().byte()
        return {'bbox': bbox, 'valid': valid, 'visible': visible}

    def _get_frame(self, sequence, frame_id):
        """已知一个序列,以及待使用的帧在序列中的序号 返回对应的帧以及二值化mask"""
        frame_name = sequence[frame_id]['file_name']
        '''image RGB 3 channels'''
        frame_img = decode_img(self.root, os.path.join('train/JPEGImages', frame_name + '.jpg'))
        '''mask 1 channel'''
        mask_img = cv2.cvtColor(decode_img(self.root, os.path.join('train/Annotations/', frame_name + '.png')),
                                cv2.COLOR_RGB2GRAY)
        mask_img = mask_img[..., np.newaxis]  # (H,W,1)
        mask_ins = (mask_img == sequence[frame_id]['id']).astype(np.uint8)  # binary mask # (H,W,1)
        '''返回一个元组,第一个元素是RGB格式的图像,第二个元素是单通道的mask(只有0,1两种取值,但是是uint8类型)'''
        return frame_img, mask_ins

    def get_frames(self, seq_id, frame_ids, anno=None):
        sequence = self.sequence_list[seq_id]

        frame_mask_list = [self._get_frame(sequence, f) for f in frame_ids]

        if anno is None:
            anno = self.get_sequence_info(seq_id)

        # Create anno dict
        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]
        '''return both frame and mask'''
        frame_list = [f for f, m in frame_mask_list]
        mask_list = [m for f, m in frame_mask_list]
        return frame_list, mask_list, anno_frames, None
