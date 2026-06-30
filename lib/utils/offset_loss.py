import torch
import torch.nn.functional as F

def masked_offset_l1_loss(pred_offset, gt_offset):
    """
    只对目标区域计算预测偏移和 GT 偏移的 L1 loss。

    Args:
        pred_offset (Tensor): (B, H, W, 2), 模型预测
        gt_offset (Tensor):   (B, H, W, 2), 标签偏移，目标框区域有值

    Returns:
        loss: scalar tensor
    """
    # 构造 mask，gt 非零区域视为目标区域
    mask = (gt_offset.abs().sum(dim=-1) > 0).float()  # (B, H, W)

    # L1 loss (element-wise)，但不做平均
    l1 = F.l1_loss(pred_offset, gt_offset, reduction='none')  # (B, H, W, 2)

    # 应用 mask，只对目标区域进行 loss 统计
    l1 = l1.sum(dim=-1) * mask  # (B, H, W)

    # 平均：只对有效区域平均
    loss = l1.sum() / (mask.sum() + 1e-6)

    return loss



def masked_cosine_similarity_loss(pred_offset, gt_offset, eps=1e-6):
    """
    计算 offset 的方向一致性损失（余弦距离），仅在目标区域进行约束。

    Args:
        pred_offset: (B, H, W, 2)
        gt_offset:   (B, H, W, 2)
        eps: small value to avoid division by zero

    Returns:
        loss: scalar
    """
    B, H, W, _ = pred_offset.shape

    # 构建 mask：标签非零表示目标区域
    mask = (gt_offset.norm(dim=-1) > 0).float()  # (B, H, W)

    # 单位化
    pred_norm = F.normalize(pred_offset, dim=-1, eps=eps)
    gt_norm = F.normalize(gt_offset, dim=-1, eps=eps)

    # cos_sim ∈ [-1, 1]，1 表示方向完全一致
    cos_sim = (pred_norm * gt_norm).sum(dim=-1)  # (B, H, W)

    # 余弦距离 = 1 - cos_sim
    cos_dist = 1 - cos_sim  # (B, H, W)

    # 只在目标区域求平均
    loss = (cos_dist * mask).sum() / (mask.sum() + eps)

    return loss

#用来轮廓偏移一致性的
def compute_offset_supervision_loss(pos, keypoints_norm):
    """
    Args:
        pos: Tensor (B, H, W, 2), offset + reference, 归一化到 [-1, 1]
        keypoints_norm: Tensor (B, N, 2), 关键点 (0~1) 范围

    Returns:
        loss: scalar Tensor
    """
    B, H, W, _ = pos.shape
    device = pos.device

    # 1. 归一化关键点坐标到 [-1,1]
    keypoints_grid = keypoints_norm * 2.0 - 1.0  # (B, N, 2)

    # 2. 随机采样每个位置的目标 keypoint
    N = keypoints_grid.shape[1]
    rand_idx = torch.randint(0, N, (B, H, W), device=device)  # 每个位置一个索引

    # 3. 用 gather 获取目标关键点坐标
    target_kps = torch.gather(
        keypoints_grid.unsqueeze(1).unsqueeze(1).expand(-1, H, W, -1, 2),  # -> (B, H, W, N, 2)
        dim=3,
        index=rand_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, H, W, 1, 2)  # (B, H, W, 1, 2)
    ).squeeze(3)  # -> (B, H, W, 2)

    # 4. L1 supervision loss
    loss = F.l1_loss(pos, target_kps)
    return loss
