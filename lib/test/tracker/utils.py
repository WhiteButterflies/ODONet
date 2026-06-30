import torch
import math
import numpy as np
import cv2 as cv
import torch.nn.functional as F
from lib.utils.misc import NestedTensor


class Preprocessor(object):
    def __init__(self):
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view((1, 3, 1, 1)).cuda()
        self.std = torch.tensor([0.229, 0.224, 0.225]).view((1, 3, 1, 1)).cuda()
        self.mm_mean = torch.tensor([0.485, 0.456, 0.406, 0.485, 0.456, 0.406]).view((1, 6, 1, 1)).cuda()
        self.mm_std = torch.tensor([0.229, 0.224, 0.225, 0.229, 0.224, 0.225]).view((1, 6, 1, 1)).cuda()

    def process(self, img_arr: np.ndarray):
        if img_arr.shape[-1] == 6:
            mean = self.mm_mean
            std = self.mm_std
        else:
            mean = self.mean
            std = self.std
        # Deal with the image patch
        img_tensor = torch.tensor(img_arr).cuda().float().permute((2,0,1)).unsqueeze(dim=0)
        # img_tensor = torch.tensor(img_arr).float().permute((2,0,1)).unsqueeze(dim=0)
        img_tensor_norm = ((img_tensor / 255.0) - mean) / std  # (1,3,H,W)
        return img_tensor_norm


def sample_target(im, target_bb, search_area_factor, output_sz=None):
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

    # Pad
    im_crop_padded = cv.copyMakeBorder(im_crop, y1_pad, y2_pad, x1_pad, x2_pad, cv.BORDER_CONSTANT)
    # deal with attention mask
    H, W, _ = im_crop_padded.shape

    if output_sz is not None:
        resize_factor = output_sz / crop_sz
        im_crop_padded = cv.resize(im_crop_padded, (output_sz, output_sz))

        return im_crop_padded, resize_factor

    else:
        return im_crop_padded, 1.0

def resize_sample_target(im, output_sz=None):
    """ Resize the image

    args:
        im - cv image
        output_sz - (float) Size to which the extracted crop is resized (always square). If None, no resizing is done.

    returns:
        cv image - extracted crop
        float - the factor by which the crop has been resized to make the crop size equal output_size
    """

    # Resize image
    # deal with attention mask
    H, W, _ = im.shape
    if output_sz is not None:
        resize_factor = (output_sz / W, output_sz / H)  # (w,h) rather than (h,w)
        im_resized = cv.resize(im, (output_sz, output_sz))
        return im_resized, resize_factor
    else:
        return im, 1.0

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
        return box_out / (crop_sz[0]-1)
    else:
        return box_out

#added by liu for LK flow point get
def get_sparse_flow_keypoints_in_boxes(images, boxes_list, max_corners=150, max_points_per_box=5,device='cuda'):
    """
    获取每个目标框内的稀疏光流特征点（归一化到框内坐标）

    Args:
        images: 图像元组/列表，每个元素是BGR图像 (H,W,3)
        boxes_list: 目标框列表，每个元素是该图像对应的目标框数组 (M,4) [lxywh归一化坐标]
        max_corners: 每张图像最多提取多少关键点
        max_points_per_box: 每个框内最多保留多少关键点

    Returns:
        results: 列表，每个元素是该图像各框内归一化关键点列表 [ (1,K1,2), (1,K2,2), ... ], K <= max_points_per_box
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
                box_kps=box_kps
        else:

            # 处理无关键点情况
            if len(box_kps) < 2:
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                tl,tr,bl,br =[x1,y1],[x2,y1],[x1,y2],[x2,y2]
                box_kps = np.array([[cx, cy],tl,tr,bl,br], dtype=np.float32)
            # 限制最大点数
            elif len(box_kps) > max_points_per_box:
                box_kps = box_kps[np.random.choice(len(box_kps), max_points_per_box, replace=False)]
            elif len(box_kps) < max_points_per_box:
                box_kps = box_kps[np.random.choice(len(box_kps), max_points_per_box, replace=True)]

        # 归一化到框内坐标 (0-1)
        norm_kps = np.zeros_like(box_kps)
        norm_kps[:, 0] = box_kps[:, 0] / img_w  # 避免除以0
        norm_kps[:, 1] = box_kps[:, 1] / img_h

        results.append(torch.tensor(norm_kps,device=device).unsqueeze(0)) #注意 与训练的同名函数不同

    return results
#这个方法貌似在测试时无用。
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

        H_list.append(H.flatten().unsqueeze(0))

    # H_batch = torch.stack(H_list, dim=0)  # (B, u*v)
    return H_list

'''suppress the peek by manual'''
# --- 辅助函数：创建抑制掩码 ---
# (与上一版代码相同)
def _create_suppression_spot(
    mask: torch.Tensor,
    b_idx: int,
    y_center: int,
    x_center: int,
    radius: int,
    factor: float
):
    B, _, H, W = mask.shape
    device = mask.device
    y, x = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing='ij'
    )
    dist_from_center = torch.sqrt((y - y_center).float()**2 + (x - x_center).float()**2)
    spot_mask = (dist_from_center <= radius)
    mask[b_idx, 0, :, :] = torch.where(
        spot_mask,
        factor,
        mask[b_idx, 0, :, :]
    )
def suppress_ambiguous_peaks_by_location(
    score_map: torch.Tensor,
    config: dict
) -> torch.Tensor:
    """
    当 ScoreMap 出现模糊时，硬编码地抑制“右下角”的峰值。
    这是一个非常特定于某个场景的策略。

    Args:
        score_map (torch.Tensor): 形状为 [B, 1, H, W] 的外观置信图。
        config (dict): 包含阈值的字典，例如：
                       'peak_thresh_factor': 0.5,  # 峰值必须高于 (最大值 * 因子)
                       'distance_threshold': 5.0,  # 峰值之间“相近”的距离阈值
                       'suppression_radius': 3,    # 抑制区域的半径
                       'suppression_factor': 0.1   # 将峰值抑制到原值的 10%
    Returns:
        torch.Tensor: 形状为 [B, 1, H, W] 的、抑制了模糊峰值后的 ScoreMap。
    """

    B, _, H, W = score_map.shape
    device = score_map.device

    # 1. 查找所有潜在峰值 (矢量化)
    local_max = F.max_pool2d(score_map, kernel_size=3, stride=1, padding=1)
    is_peak = (score_map == local_max)
    max_val_per_batch = torch.amax(score_map, dim=(2, 3), keepdim=True)
    is_above_thresh = (score_map > max_val_per_batch * config['peak_thresh_factor'])
    all_peaks_mask = is_peak & is_above_thresh

    peak_coords = torch.nonzero(all_peaks_mask)
    if peak_coords.shape[0] == 0:
        return score_map

    batch_indices = peak_coords[:, 0]

    # 2. 循环每个批次，应用抑制逻辑
    suppression_mask = torch.ones_like(score_map)

    for b in range(B):
        mask_in_batch = (batch_indices == b)
        num_peaks_in_batch = torch.sum(mask_in_batch)

        # 检查模糊性：只有当至少有两个峰值时才需要处理
        if num_peaks_in_batch < 2:
            continue

        b_coords = peak_coords[mask_in_batch] # [num_peaks_in_b, 4] (b, c, y, x)

        # 提取当前批次峰值的 y, x 坐标 [num_peaks_in_b, 2]
        b_yx_coords = b_coords[:, 2:].float()

        # 3. 找到分数最高的两个峰值 (Top-2)
        b_scores = score_map[b_coords[:, 0], b_coords[:, 1], b_coords[:, 2], b_coords[:, 3]]

        # 如果峰值少于2个，理论上不会发生，但以防万一
        if b_scores.shape[0] < 2:
            continue

        top2_scores, top2_indices = torch.topk(b_scores, 2)

        p1_coord = b_yx_coords[top2_indices[0]] # (y, x) of P1
        p2_coord = b_yx_coords[top2_indices[1]] # (y, x) of P2

        p1_full_coord = b_coords[top2_indices[0]] # (b, c, y, x) of P1
        p2_full_coord = b_coords[top2_indices[1]] # (b, c, y, x) of P2

        # 4. 检查是否“相近”
        dist = torch.dist(p1_coord, p2_coord)

        if dist < config['distance_threshold']:
            # **触发抑制**
            # 两个最强的峰值彼此靠近，存在模糊性

            # 5. 决策：哪个是“右下角”？
            # 我们用一个简单的启发式规则：y+x 更大的那个是“右下角”
            p1_loc_score = p1_coord[0] + p1_coord[1] # y + x
            p2_loc_score = p2_coord[0] + p2_coord[1] # y + x

            if p1_loc_score > p2_loc_score:
                # P1 在 P2 的右下方，抑制 P1
                peak_to_suppress = p1_full_coord
            else:
                # P2 在 P1 的右下方 (或同一对角线)，抑制 P2
                peak_to_suppress = p2_full_coord

            # 6. 执行抑制
            c_idx, y_idx, x_idx = int(peak_to_suppress[1]), int(peak_to_suppress[2]), int(peak_to_suppress[3])

            _create_suppression_spot(
                suppression_mask, b, y_idx, x_idx,
                config['suppression_radius'],
                config['suppression_factor']
            )

    # 7. 应用掩码并返回
    return score_map * suppression_mask

'''ended'''
