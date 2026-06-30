import torch
import math
import cv2 as cv
import torch.nn.functional as F
import numpy as np

'''modified from the original test implementation
Replace cv.BORDER_REPLICATE with cv.BORDER_CONSTANT
Add a variable called att_mask for computing attention and positional encoding later'''


def sample_target(im, target_bb, search_area_factor, output_sz=None, mask=None):
    """ Extracts a square crop centered at target_bb box, of area search_area_factor^2 times target_bb area

    args:
        im - cv image
        target_bb - target box [x, y, w, h]
        search_area_factor - Ratio of crop size to target size
        output_sz - (float) Size to which the extracted crop is resized (always square). If None, no resizing is done.

    returns:
        cv image - extracted crop
        float - the factor by which the crop has been resized to make the crop size equal output_size
    """
    if not isinstance(target_bb, list):
        x, y, w, h = target_bb.tolist()
    else:
        x, y, w, h = target_bb
    # Crop image
    crop_sz = math.ceil(math.sqrt(w * h) * search_area_factor)

    if crop_sz < 1:
        raise Exception('Too small bounding box.')

    x1 = round(x + 0.5 * w - crop_sz * 0.5)
    x2 = x1 + crop_sz

    y1 = round(y + 0.5 * h - crop_sz * 0.5)
    y2 = y1 + crop_sz

    x1_pad = max(0, -x1)
    x2_pad = max(x2 - im.shape[1] + 1, 0)

    y1_pad = max(0, -y1)
    y2_pad = max(y2 - im.shape[0] + 1, 0)

    # Crop target
    im_crop = im[y1 + y1_pad:y2 - y2_pad, x1 + x1_pad:x2 - x2_pad, :]
    if mask is not None:
        mask_crop = mask[y1 + y1_pad:y2 - y2_pad, x1 + x1_pad:x2 - x2_pad]

    # Pad
    im_crop_padded = cv.copyMakeBorder(im_crop, y1_pad, y2_pad, x1_pad, x2_pad, cv.BORDER_CONSTANT)
    # deal with attention mask
    H, W, _ = im_crop_padded.shape
    att_mask = np.ones((H,W))
    end_x, end_y = -x2_pad, -y2_pad
    if y2_pad == 0:
        end_y = None
    if x2_pad == 0:
        end_x = None
    att_mask[y1_pad:end_y, x1_pad:end_x] = 0
    if mask is not None:
        mask_crop_padded = F.pad(mask_crop, pad=(x1_pad, x2_pad, y1_pad, y2_pad), mode='constant', value=0)

    if output_sz is not None:
        resize_factor = output_sz / crop_sz
        im_crop_padded = cv.resize(im_crop_padded, (output_sz, output_sz))
        att_mask = cv.resize(att_mask, (output_sz, output_sz)).astype(np.bool_)
        if mask is None:
            return im_crop_padded, resize_factor, att_mask
        mask_crop_padded = \
        F.interpolate(mask_crop_padded[None, None], (output_sz, output_sz), mode='bilinear', align_corners=False)[0, 0]
        return im_crop_padded, resize_factor, att_mask, mask_crop_padded

    else:
        if mask is None:
            return im_crop_padded, att_mask.astype(np.bool_), 1.0
        return im_crop_padded, 1.0, att_mask.astype(np.bool_), mask_crop_padded

def resize_sample_target(im, target_bb, output_sz=None, mask=None):
    """ Resize the image

    args:
        im - cv image
        target_bb - target box [x, y, w, h]
        output_sz - (float) Size to which the extracted crop is resized (always square). If None, no resizing is done.

    returns:
        cv image - extracted crop
        float - the factor by which the crop has been resized to make the crop size equal output_size
    """

    # Resize image
    # deal with attention mask
    H, W, _ = im.shape
    att_mask = np.zeros((H,W))

    if output_sz is not None:
        resize_factor = (output_sz / W, output_sz / H)  # (w,h) rather than (h,w)
        im_resized = cv.resize(im, (output_sz, output_sz))
        att_mask = cv.resize(att_mask, (output_sz, output_sz)).astype(np.bool_)
        if mask is None:
            return im_resized, resize_factor, att_mask
        mask_resized = \
        F.interpolate(mask[None, None], (output_sz, output_sz), mode='bilinear', align_corners=False)[0, 0]
        return im_resized, resize_factor, att_mask, mask_resized

    else:
        if mask is None:
            return im, att_mask.astype(np.bool_), 1.0
        return im, 1.0, att_mask.astype(np.bool_), mask

def transform_image_to_crop(box_in: torch.Tensor, box_extract: torch.Tensor, resize_factor: float,
                            crop_sz: torch.Tensor, normalize=False) -> torch.Tensor:
    """ Transform the box co-ordinates from the original image co-ordinates to the co-ordinates of the cropped image
    args:
        box_in - the box for which the co-ordinates are to be transformed
        box_extract - the box about which the image crop has been extracted.
        resize_factor - the ratio between the original image scale and the scale of the image crop
        crop_sz - size of the cropped image

    returns:
        torch.Tensor - transformed co-ordinates of box_in
    """
    box_extract_center = box_extract[0:2] + 0.5 * box_extract[2:4]

    box_in_center = box_in[0:2] + 0.5 * box_in[2:4]

    box_out_center = (crop_sz - 1) / 2 + (box_in_center - box_extract_center) * resize_factor
    box_out_wh = box_in[2:4] * resize_factor

    box_out = torch.cat((box_out_center - 0.5 * box_out_wh, box_out_wh))
    if normalize:
        #crop_sz[0] - 1, modified by chenxin from crop_sz[0],2022.7.15
        return box_out / (crop_sz[0]-1)
    else:
        return box_out

def transform_image_to_resize(box_in: torch.Tensor, resize_factor: float,
                            crop_sz: torch.Tensor, normalize=False) -> torch.Tensor:
    """ Transform the box co-ordinates from the original image co-ordinates to the co-ordinates of the cropped image
    args:
        box_in - the box for which the co-ordinates are to be transformed
        box_extract - the box about which the image crop has been extracted.
        resize_factor - the ratio between the original image scale and the scale of the image crop
        crop_sz - size of the cropped image

    returns:
        torch.Tensor - transformed co-ordinates of box_in
    """
    box_out_xy = box_in[:2] * torch.tensor(resize_factor)
    box_out_wh = box_in[2:4] * torch.tensor(resize_factor)

    box_out = torch.cat((box_out_xy, box_out_wh))
    if normalize:
        return box_out / (crop_sz[0]-1)
    else:
        return box_out

def jittered_center_crop(frames, box_extract, box_gt, search_area_factor, output_sz, masks=None):
    """ For each frame in frames, extracts a square crop centered at box_extract, of area search_area_factor^2
    times box_extract area. The extracted crops are then resized to output_sz. Further, the co-ordinates of the box
    box_gt are transformed to the image crop co-ordinates

    args:
        frames - list of frames
        box_extract - list of boxes of same length as frames. The crops are extracted using anno_extract
        box_gt - list of boxes of same length as frames. The co-ordinates of these boxes are transformed from
                    image co-ordinates to the crop co-ordinates
        search_area_factor - The area of the extracted crop is search_area_factor^2 times box_extract area
        output_sz - The size to which the extracted crops are resized

    returns:
        list - list of image crops
        list - box_gt location in the crop co-ordinates
        """

    if masks is None:
        crops_resize_factors = [sample_target(f, a, search_area_factor, output_sz)
                                for f, a in zip(frames, box_extract)]
        frames_crop, resize_factors, att_mask = zip(*crops_resize_factors)
        masks_crop = None
    else:
        crops_resize_factors = [sample_target(f, a, search_area_factor, output_sz, m)
                                for f, a, m in zip(frames, box_extract, masks)]
        frames_crop, resize_factors, att_mask, masks_crop = zip(*crops_resize_factors)
    # frames_crop: tuple of ndarray (128,128,3), att_mask: tuple of ndarray (128,128)
    crop_sz = torch.Tensor([output_sz, output_sz])

    # find the bb location in the crop
    '''Note that here we use normalized coord'''
    box_crop = [transform_image_to_crop(a_gt, a_ex, rf, crop_sz, normalize=True)
                for a_gt, a_ex, rf in zip(box_gt, box_extract, resize_factors)]  # (x1,y1,w,h) list of tensors
    '''added by liu for get box_extract 就像 self.state 的真实坐标，也就是裁切中心的'''
    box_extract_crop = [transform_image_to_crop(a_gt, a_ex, rf, crop_sz, normalize=True)
                for a_gt, a_ex, rf in zip(box_extract, box_extract, resize_factors)]  # (x1,y1,w,h) list of tensors
    '''ended by liu'''

    return frames_crop, box_crop, att_mask, masks_crop,box_extract_crop

def pstb_jittered_center_crop(frames, box_extract, box_gt, box_frame, search_area_factor, output_sz, masks=None):
    """ For each frame in frames, extracts a square crop centered at box_extract, of area search_area_factor^2
    times box_extract area. The extracted crops are then resized to output_sz. Further, the co-ordinates of the box
    box_gt are transformed to the image crop co-ordinates

    args:
        frames - list of frames
        box_extract - list of boxes of same length as frames. The crops are extracted using anno_extract
        box_gt - list of boxes of same length as frames. The co-ordinates of these boxes are transformed from
                    image co-ordinates to the crop co-ordinates
        search_area_factor - The area of the extracted crop is search_area_factor^2 times box_extract area
        output_sz - The size to which the extracted crops are resized

    returns:
        list - list of image crops
        list - box_gt location in the crop co-ordinates
        """

    if masks is None:
        crops_resize_factors = [sample_target(f, a, search_area_factor, output_sz)
                                for f, a in zip(frames, box_extract)]
        frames_crop, resize_factors, att_mask = zip(*crops_resize_factors)
        masks_crop = None
    else:
        crops_resize_factors = [sample_target(f, a, search_area_factor, output_sz, m)
                                for f, a, m in zip(frames, box_extract, masks)]
        frames_crop, resize_factors, att_mask, masks_crop = zip(*crops_resize_factors)
    # frames_crop: tuple of ndarray (128,128,3), att_mask: tuple of ndarray (128,128)
    crop_sz = torch.Tensor([output_sz, output_sz])

    # find the bb location in the crop
    '''Note that here we use normalized coord'''
    box_crop = [transform_image_to_crop(a_gt, a_ex, rf, crop_sz, normalize=True)
                for a_gt, a_ex, rf in zip(box_gt, box_extract, resize_factors)]  # (x1,y1,w,h) list of tensors
    box_frame_crop = [transform_image_to_crop(a_gt, box_extract[-1], resize_factors[-1], crop_sz, normalize=True)
                      for a_gt in box_frame]

    return frames_crop, box_crop, box_frame_crop, att_mask, masks_crop

def resize(frames, box, output_sz, masks=None):
    """ For each frame in frames, extracts a square crop centered at box_extract, of area search_area_factor^2
    times box_extract area. The extracted crops are then resized to output_sz. Further, the co-ordinates of the box
    box_gt are transformed to the image crop co-ordinates

    args:
        frames - list of frames
        box_extract - list of boxes of same length as frames. The crops are extracted using anno_extract
        box_gt - list of boxes of same length as frames. The co-ordinates of these boxes are transformed from
                    image co-ordinates to the crop co-ordinates
        search_area_factor - The area of the extracted crop is search_area_factor^2 times box_extract area
        output_sz - The size to which the extracted crops are resized

    returns:
        list - list of image crops
        list - box_gt location in the crop co-ordinates
        """

    if masks is None:
        crops_resize_factors = [resize_sample_target(f, a, output_sz)
                                for f, a in zip(frames, box)]
        frames_crop, resize_factors, att_mask = zip(*crops_resize_factors)
        masks_crop = None
    else:
        crops_resize_factors = [resize_sample_target(f, a, output_sz, m)
                                for f, a, m in zip(frames, box, masks)]
        frames_crop, resize_factors, att_mask, masks_crop = zip(*crops_resize_factors)
    # frames_crop: tuple of ndarray (128,128,3), att_mask: tuple of ndarray (128,128)
    crop_sz = torch.Tensor([output_sz, output_sz])

    # find the bb location in the crop
    '''Note that here we use normalized coord'''
    box_crop = [transform_image_to_resize(bb, rf, crop_sz, normalize=True)
                for bb, rf in zip(box, resize_factors)]  # (x1,y1,w,h) list of tensors

    return frames_crop, box_crop, att_mask, masks_crop


def transform_box_to_crop(box: torch.Tensor, crop_box: torch.Tensor, crop_sz: torch.Tensor, normalize=False) -> torch.Tensor:
    """ Transform the box co-ordinates from the original image co-ordinates to the co-ordinates of the cropped image
    args:
        box - the box for which the co-ordinates are to be transformed
        crop_box - bounding box defining the crop in the original image
        crop_sz - size of the cropped image

    returns:
        torch.Tensor - transformed co-ordinates of box_in
    """

    box_out = box.clone()
    box_out[:2] -= crop_box[:2]

    scale_factor = crop_sz / crop_box[2:]

    box_out[:2] *= scale_factor
    box_out[2:] *= scale_factor
    if normalize:
        return box_out / crop_sz[0]
    else:
        return box_out

#added by liu for LK flow point get
def get_sparse_flow_keypoints_in_boxes(images, boxes_list, max_corners=150, max_points_per_box=5):
    """
    获取每个目标框内的稀疏光流特征点（归一化到框内坐标）

    Args:
        images: 图像元组/列表，每个元素是BGR图像 (H,W,3)
        boxes_list: 目标框列表，每个元素是该图像对应的目标框数组 (M,4)
        max_corners: 每张图像最多提取多少关键点
        max_points_per_box: 每个框内最多保留多少关键点

    Returns:
        results: 列表，每个元素是该图像各框内归一化关键点列表 [ (K1,2), (K2,2), ... ], K <= max_points_per_box
    """
    results = []

    for img, box in zip(images, boxes_list):
        # 1. 提取稀疏光流关键点
        gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        pts = cv.goodFeaturesToTrack(
            image=gray,
            maxCorners=max_corners,
            qualityLevel=0.05,
            minDistance=7,
            blockSize=7
        )
        keypoints = pts.reshape(-1, 2).astype(np.float32) if pts is not None else np.empty((0, 2), dtype=np.float32)

        # 2. 处理每个框
        img_h, img_w = img.shape[:2]

        # 转换框为像素坐标
        l, t, bw, bh = box
        x1, y1 = int(l * img_w), int(t * img_h)
        x2, y2 = int((l + bw) * img_w), int((t + bh) * img_h)

        # 找出框内关键点
        in_box = ((keypoints[:, 0] >= x1) & (keypoints[:, 0] <= x2) &
                  (keypoints[:, 1] >= y1) & (keypoints[:, 1] <= y2))
        box_kps = keypoints[in_box]

        if max_points_per_box ==999:
            # 处理无关键点情况
            if len(box_kps) < 2:
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                tl, tr, bl, br = [x1, y1], [x2, y1], [x1, y2], [x2, y2]
                box_kps = np.array([[cx, cy], tl, tr, bl, br], dtype=np.float32)
            else:
                box_kps = box_kps
        else:
            # 处理无关键点情况
            if len(box_kps) < 2:
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                tl, tr, bl, br = [x1, y1], [x2, y1], [x1, y2], [x2, y2]
                box_kps = np.array([[cx, cy], tl, tr, bl, br], dtype=np.float32)
            # 限制最大点数
            elif len(box_kps) > max_points_per_box:
                box_kps = box_kps[np.random.choice(len(box_kps), max_points_per_box, replace=False)]
            elif len(box_kps) < max_points_per_box:
                box_kps = box_kps[np.random.choice(len(box_kps), max_points_per_box, replace=True)]

        # 归一化到框内坐标 (0-1)
        norm_kps = np.zeros_like(box_kps)
        norm_kps[:, 0] = box_kps[:, 0] / img_w  # 避免除以0
        norm_kps[:, 1] = box_kps[:, 1] / img_h

        results.append(torch.tensor(norm_kps))

    return results

#增加此函数实现bbox_extret_center 到bbox_extract的光流特征点变换
import torch


def transform_keypoints_with_boxes(original_keypoints_list, original_boxes_list, transformed_boxes_list):
    """
    根据框的变换参数转换关键点坐标（PyTorch 张量版本）

    Args:
        original_keypoints_list: 原始关键点列表，每个元素是 (K,2) 的tensor，坐标归一化到图像尺寸
        original_boxes_list: 原始框列表，每个元素是归一化xywh格式的tensor
        transformed_boxes_list: 变换后的框列表，每个元素是归一化xywh格式的tensor

    Returns:
        transformed_keypoints_list: 变换后的关键点列表，每个元素是 (K,2) 的tensor
    """
    transformed_keypoints_list = []

    for kps, orig_box, trans_box in zip(original_keypoints_list, original_boxes_list, transformed_boxes_list):
        # 确保所有输入都是张量
        kps = kps if isinstance(kps, torch.Tensor) else torch.tensor(kps)
        orig_box = orig_box if isinstance(orig_box, torch.Tensor) else torch.tensor(orig_box)
        trans_box = trans_box if isinstance(trans_box, torch.Tensor) else torch.tensor(trans_box)

        # 解包原始框和变换后框
        orig_x, orig_y, orig_w, orig_h = orig_box.unbind()
        trans_x, trans_y, trans_w, trans_h = trans_box.unbind()

        # 1. 将关键点从图像归一化坐标转换到原始框内的归一化坐标 (0-1)
        box_kps_x = (kps[:, 0] - orig_x) / orig_w
        box_kps_y = (kps[:, 1] - orig_y) / orig_h

        # 2. 应用框的变换：平移和缩放
        new_kps_x = trans_x + box_kps_x * trans_w
        new_kps_y = trans_y + box_kps_y * trans_h

        # 组合成新的关键点
        new_kps = torch.stack([new_kps_x, new_kps_y], dim=1)
        transformed_keypoints_list.append(new_kps)

    return transformed_keypoints_list


#added by liu for calc offset point by box
def compute_point_offsets(original_boxes, new_boxes, points, flip_prob=0.0, is_flipped=None):
    """
    Compute keypoint offsets (Δx, Δy) in new boxes with batch support.
    Horizontal flipping is performed around the vertical axis at x=0.5.

    Args:
        original_boxes: Tensor of shape (B, 4) [x, y, w, h]
        new_boxes: Tensor of shape (B, 4) [x, y, w, h]
        points: Tensor of shape (B, N, 2) where N is number of points per sample
        flip_prob: Probability of flipping
        is_flipped: Optional tensor of shape (B,) indicating which samples to flip

    Returns:
        offsets: Tensor of shape (B, N, 2) containing (Δx, Δy) offsets
        is_flipped: Tensor indicating which samples were flipped
    """
    B = original_boxes.shape[0]

    # Reshape boxes for broadcasting
    x, y, w, h = original_boxes[:, 0], original_boxes[:, 1], original_boxes[:, 2], original_boxes[:, 3]
    x2, y2, w2, h2 = new_boxes[:, 0], new_boxes[:, 1], new_boxes[:, 2], new_boxes[:, 3]

    # Calculate relative coordinates
    x_rel = (points[..., 0] - x.unsqueeze(-1)) / w.unsqueeze(-1)
    y_rel = (points[..., 1] - y.unsqueeze(-1)) / h.unsqueeze(-1)

    # Handle flipping
    if is_flipped is None:
        if flip_prob > 0:
            is_flipped = torch.rand(B, device=original_boxes.device) < flip_prob
        else:
            is_flipped = torch.zeros(B, dtype=torch.bool, device=original_boxes.device)

    # Apply flipping to x_rel where is_flipped is True
    x_rel = torch.where(is_flipped.unsqueeze(-1), 1 - x_rel, x_rel)

    # Calculate new coordinates
    x_new = x2.unsqueeze(-1) + w2.unsqueeze(-1) * x_rel
    y_new = y2.unsqueeze(-1) + h2.unsqueeze(-1) * y_rel

    # Calculate offsets
    offsets = torch.stack([x_new - points[..., 0], y_new - points[..., 1]], dim=-1)

    return offsets, is_flipped


#仿照GvSeg
def compute_shape_position_descriptor_list(norm_kps_list, u=16, v=32, d_model=256):
    """
    norm_kps_list: list of (K_i, 2) torch.Tensor, 每个样本的关键点 (归一化到 [0,1])

    输出:
    H_batch: (B, u*v) torch.Tensor
    """
    B = len(norm_kps_list)
    device = norm_kps_list[0].device
    H_list = []

    for norm_kps in norm_kps_list:
        K = norm_kps.shape[0]

        cx, cy = norm_kps[:, 0].mean(), norm_kps[:, 1].mean()
        dx = norm_kps[:, 0] - cx
        dy = norm_kps[:, 1] - cy
        r = torch.sqrt(dx ** 2 + dy ** 2)
        theta = (torch.atan2(dy, dx) + 2 * torch.pi) % (2 * torch.pi)

        theta_bin_size = 2 * torch.pi / u
        r_bin_size = 1.0 / v

        theta_idx = torch.clamp((theta / theta_bin_size).long(), max=u - 1)
        r_idx = torch.clamp((r / r_bin_size).long(), max=v - 1)

        H = torch.zeros((u, v), dtype=torch.float32, device=device)
        for i in range(K):
            H[theta_idx[i], r_idx[i]] += 1.0 / torch.sqrt(torch.tensor(d_model, device=device))

        H_list.append(H.flatten())

    # H_batch = torch.stack(H_list, dim=0)  # (B, u*v)
    return H_list

def apply_horizontal_flip_to_keypoints(norm_kps_list, flip_flags):
    """
    norm_kps_list: list of (K_i, 2) torch.Tensor
    flip_flags: list of bool, 是否需要水平翻转

    返回:
    flipped_kps_list: list of (K_i, 2) torch.Tensor
    """
    flipped_kps_list = []

    for kps, flip in zip(norm_kps_list, flip_flags):
        if flip:
            flipped_kps = kps.clone()
            flipped_kps[:, 0] = 1.0 - flipped_kps[:, 0]  # 只翻x坐标
        else:
            flipped_kps = kps

        flipped_kps_list.append(flipped_kps)

    return flipped_kps_list


def compare_tensor_lists(list1, list2):
    #用x坐标是否变化判断是否进行水平翻转数据增强
    return [t1[0].item() != t2[0].item() for t1, t2 in zip(list1, list2)]

