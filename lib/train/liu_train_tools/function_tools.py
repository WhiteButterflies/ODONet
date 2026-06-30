import torch
import torch.nn.functional as F
from torchvision.ops import roi_align

def cxcywh_to_xyxy_norm(boxes_cxcywh: torch.Tensor) -> torch.Tensor:
    """
    boxes_cxcywh: (B, N, 4) in normalized [0,1] cx,cy,w,h
    returns: (B, N, 4) normalized [0,1] x1,y1,x2,y2
    """
    cx, cy, w, h = boxes_cxcywh.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=-1)

def roi_token_from_tokens(
    tokens: torch.Tensor,          # (B, HW, C)
    boxes_cxcywh: torch.Tensor,     # (B, N, 4) normalized cx,cy,w,h
    H: int,
    W: int,
    roi_out_size=(7, 7),#（ori 7，7）
    spatial_scale: float = 1.0,
    aligned: bool = True,
) -> torch.Tensor:
    """
    return: (B, N, C)  one token per ROI using AdaptiveAvgPool2d((1,1))
    """
    B, HW, C = tokens.shape
    assert HW == H * W, f"HW({HW}) must equal H*W({H*W})"
    assert boxes_cxcywh.shape[0] == B and boxes_cxcywh.shape[-1] == 4

    # (B, HW, C) -> (B, C, H, W)
    feat = tokens.transpose(1, 2).contiguous().view(B, C, H, W)

    # cxcywh -> xyxy (normalized), then clamp
    boxes_xyxy = cxcywh_to_xyxy_norm(boxes_cxcywh).clamp(0, 1)

    # roi_align expects boxes in absolute coords: x1,y1,x2,y2 in feature-map scale
    # If your boxes are normalized w.r.t image, and feat is at same scale as image,
    # multiply by (W,H). If feat is downsampled, adapt accordingly.
    # Here we assume boxes are normalized w.r.t feat map size.
    boxes_abs = boxes_xyxy.clone()
    boxes_abs[..., 0] *= W
    boxes_abs[..., 2] *= W
    boxes_abs[..., 1] *= H
    boxes_abs[..., 3] *= H

    # Build RoIs tensor: (B*N, 5) with batch_idx
    N = boxes_abs.shape[1]
    batch_idx = torch.arange(B, device=tokens.device)[:, None].expand(B, N).reshape(-1, 1).float()
    rois = torch.cat([batch_idx, boxes_abs.reshape(-1, 4)], dim=1)  # (B*N, 5)

    # ROI Align: (B*N, C, oh, ow)
    roi_feat = roi_align(
        feat, rois,
        output_size=roi_out_size,
        spatial_scale=spatial_scale,
        sampling_ratio=-1,
        aligned=aligned,
    )

    # AdaptiveAvgPool2d -> (B*N, C, 1, 1) -> (B, N, C)
    roi_token = F.adaptive_avg_pool2d(roi_feat, (1, 1)).squeeze(-1).squeeze(-1)
    roi_token = roi_token.view(B, N, C)
    return roi_token
def expand_tensor_list_uniform(tensor_list, n):
    m = len(tensor_list)
    if m == 0:
        raise ValueError("tensor_list 不能为空")
    if n < m:
        raise ValueError("n 必须大于等于原长度 m")

    result = []
    for i, x in enumerate(tensor_list):
        repeat = round((i + 1) * n / m) - round(i * n / m)
        result.extend([x] * repeat)
    return result

##---------------lazy-------

def gaussian_kernel_1d(kernel_size, sigma):
    kernel = torch.exp(-0.5 * (torch.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1).float() / sigma) ** 2)
    kernel = kernel / torch.max(kernel)
    return kernel

def roi_token_from_tokens_lazystrike(
        tokens: torch.Tensor,  # (B, HW, C)
        boxes_cxcywh: torch.Tensor,  # (B, N, 4) normalized cx,cy,w,h
        H: int,
        W: int,
        roi_out_size=(7, 7),
        aligned: bool = True,
        top_k: int = 1,  # K 值：从 7x7=49 个 token 中选出最稳定的 K 个
        sigma: float = 2.0,  # 高斯核的宽度
        cached_kernel: torch.Tensor = None  # 建议传入预先计算好的高斯核
) -> torch.Tensor:
    """
    结合 LazyStrike 频域过滤思想的 ROI Token 提取。
    return: (B, N, C)
    """
    B, HW, C = tokens.shape
    N = boxes_cxcywh.shape[1]
    assert HW == H * W, f"HW({HW}) must equal H*W({H * W})"

    # 1. 重构特征图并提取 ROI 特征 (B*N, C, 7, 7)
    feat = tokens.transpose(1, 2).contiguous().view(B, C, H, W)
    boxes_xyxy = cxcywh_to_xyxy_norm(boxes_cxcywh).clamp(0, 1)

    boxes_abs = boxes_xyxy.clone()
    boxes_abs[..., 0] *= W
    boxes_abs[..., 2] *= W
    boxes_abs[..., 1] *= H
    boxes_abs[..., 3] *= H

    boxes_list = list(torch.unbind(boxes_abs, dim=0))

    # roi_feat shape: (B*N, C, 7, 7)
    roi_feat = roi_align(
        feat, boxes_list,
        output_size=roi_out_size,
        spatial_scale=1.0,
        sampling_ratio=-1,
        aligned=aligned,
    )

    # 2. 将空间维度展平，适配 LazyStrike 计算
    # (B*N, C, 7, 7) -> (B*N, C, 49) -> (B*N, 49, C)
    roi_tokens = roi_feat.flatten(start_dim=2).transpose(1, 2)
    num_roi_tokens = roi_tokens.shape[1]  # 通常是 49

    # 如果 top_k 超过了网格内的 token 总数，限制为最大 token 数
    k = min(top_k, num_roi_tokens)

    # 3. 获取或生成高斯核
    gs_k = cached_kernel if cached_kernel is not None else gaussian_kernel_1d(C, C ** 0.5).to(roi_tokens.device).unsqueeze(0).unsqueeze(0)
    # gs_k shape: (C,), PyTorch 会自动广播到 (B*N, 49, C)

    # ==========================================
    # 4. LazyStrike 核心逻辑 (频域稳定度过滤)
    # ==========================================
    x_detach = roi_tokens  # (B*N, 49, C)

    # 对通道维度 (dim=-1) 进行 FFT
    x_fft = torch.fft.fft(x_detach, dim=-1)

    # 频域中心化 -> 乘以高斯核(低通滤波) -> 逆中心化
    x_fft_shifted = torch.fft.fftshift(x_fft, dim=-1)
    x_fft_filtered = x_fft_shifted * gs_k
    x_fft_ishifted = torch.fft.ifftshift(x_fft_filtered, dim=-1)

    # 逆 FFT 返回时域，并提取实部，得到平滑后的 tokens
    x_smoothed = torch.fft.ifft(x_fft_ishifted, dim=-1).real

    # 计算稳定度分数 (加上 1e-6 防止分母为 0)
    diff = x_detach / (torch.abs(x_smoothed - x_detach) + 1e-6)  # (B*N, 49, C)

    # 在 token 维度 (dim=1) 上，为每个通道选出最稳定的 top-k 个 token
    _, indices = torch.topk(diff, k=k, dim=1, largest=True)  # (B*N, K, C)

    # 提取这些 top-k 的 token
    sel_p = torch.gather(x_detach, 1, indices)  # (B*N, K, C)

    # 在选出的 K 个 token 上进行平均，作为该 ROI 最终的特征
    roi_cls_token = torch.mean(sel_p, dim=1)  # (B*N, C)

    # ==========================================
    # 5. 还原维度
    # ==========================================
    # (B*N, C) -> (B, N, C)
    roi_cls_token = roi_cls_token.view(B, N, C)

    return roi_cls_token


import torch


def roi_token_from_tokens_lazystrike_no_interpolate(
        tokens: torch.Tensor,  # (B, HW, C)
        boxes_cxcywh: torch.Tensor,  # (B, N, 4) normalized cx,cy,w,h
        H: int,
        W: int,
        top_k_ratio: float = 0.01,  # 动态 Top-K 比例（例如取最稳定的 30%）
        min_k: int = 1,  # 防止框极小，保底至少取 1 个 Token
        sigma: float = 2.0,  # 高斯核的宽度
        cached_kernel: torch.Tensor = None
) -> torch.Tensor:
    """
    不使用 roi_align 插值，直接在特征图上进行坐标截断，
    并配合 LazyStrike 频域过滤思想提取目标特征。
    返回: (B, N, C) 保证每个 ROI 只吐出唯一 1 个特征向量
    """
    B, HW, C = tokens.shape
    N = boxes_cxcywh.shape[1]
    assert HW == H * W, f"HW({HW}) must equal H*W({H * W})"

    # 1. 坐标转换：cxcywh -> xyxy (normalized)
    cx, cy, w, h = boxes_cxcywh.unbind(-1)
    x1_norm = (cx - 0.5 * w).clamp(0, 1)
    y1_norm = (cy - 0.5 * h).clamp(0, 1)
    x2_norm = (cx + 0.5 * w).clamp(0, 1)
    y2_norm = (cy + 0.5 * h).clamp(0, 1)

    # 映射到特征图的网格坐标上
    x1_grid = (x1_norm * W).floor().long().clamp(0, W - 1)
    y1_grid = (y1_norm * H).floor().long().clamp(0, H - 1)
    x2_grid = (x2_norm * W).ceil().long().clamp(0, W)  # 注意上限取 W 和 H，方便直接切片
    y2_grid = (y2_norm * H).ceil().long().clamp(0, H)

    # (B, HW, C) -> (B, H, W, C) 方便使用物理坐标切片提取
    tokens_2d = tokens.view(B, H, W, C)

    # 准备存储结果的空张量，形状与你的要求完美对齐 (B, N, C)
    out_tokens = torch.zeros((B, N, C), device=tokens.device, dtype=tokens.dtype)

    # 获取或生成高斯核 (C,) -> (1, 1, C) 用于后续广播
    if cached_kernel is not None:
        gs_k = cached_kernel
    else:
        # 这里复用了你原代码中的高斯核生成方式
        kernel = torch.exp(-0.5 * (torch.arange(-C // 2 + 1, C // 2 + 1).float() / sigma) ** 2)
        gs_k = kernel / torch.max(kernel)
        gs_k = gs_k.to(tokens.device)

    if gs_k.dim() == 1:
        gs_k = gs_k.unsqueeze(0).unsqueeze(0)  # 适配 (1, num_tokens, C) 的计算

    # 2. 遍历 Batch 和 Box，直接硬截取并应用 LazyStrike
    # (因为 B 和 N 在推理时极小，通常是 1x1 或 1x3，使用 for 循环对速度几乎无影响)
    for b in range(B):
        for n in range(N):
            x1, x2 = x1_grid[b, n].item(), x2_grid[b, n].item()
            y1, y2 = y1_grid[b, n].item(), y2_grid[b, n].item()

            # 安全保护：如果框极度小（例如特征图上不到一个像素），强行保证至少能切出 1x1 的网格
            if x1 == x2: x2 = min(x1 + 1, W)
            if y1 == y2: y2 = min(y1 + 1, H)

            # 直接提取框内的原版纯净 Token，形状为 (h_box, w_box, C)
            box_tokens = tokens_2d[b, y1:y2, x1:x2, :]

            # 展平空间维度 -> (1, num_tokens, C) 以适配 LazyStrike
            x_detach = box_tokens.reshape(1, -1, C)
            num_tokens = x_detach.shape[1]

            # 根据实际圈出来的 Token 数量，动态计算需要保留的 Top-K 数量
            k = max(min_k, int(num_tokens * top_k_ratio))
            k = min(k, num_tokens)  # 防止溢出

            # ==========================================
            # 3. LazyStrike 核心逻辑 (频域稳定度过滤)
            # ==========================================
            # 对通道维度 (dim=-1) 进行 FFT
            x_fft = torch.fft.fft(x_detach, dim=-1)

            # 频域中心化 -> 乘以高斯核(低通滤波) -> 逆中心化
            x_fft_shifted = torch.fft.fftshift(x_fft, dim=-1)
            x_fft_filtered = x_fft_shifted * gs_k
            x_fft_ishifted = torch.fft.ifftshift(x_fft_filtered, dim=-1)

            # 逆 FFT 返回时域，并提取实部
            x_smoothed = torch.fft.ifft(x_fft_ishifted, dim=-1).real

            # 计算稳定度分数 (加上 1e-6 防止分母为 0)
            diff = x_detach / (torch.abs(x_smoothed - x_detach) + 1e-6)  # (1, num_tokens, C)

            # 在 token 维度 (dim=1) 上，为每个通道独立选出最稳定的 top-k 个 token
            _, indices = torch.topk(diff, k=k, dim=1, largest=True)  # (1, K, C)

            # 提取这些 top-k 的纯净 token
            sel_p = torch.gather(x_detach, 1, indices)  # (1, K, C)

            # 融合压缩：在选出的 K 个 token 上进行平均，生成代表该目标的最终唯一特征向量
            roi_cls_token = torch.mean(sel_p, dim=1).squeeze(0)  # (C,)

            # 存入结果张量
            out_tokens[b, n] = roi_cls_token

    return out_tokens


def build_2d_sincos_pe( cx, cy, C, temperature=10000.0):
    """
    为 ROI box 的中心坐标生成 2D 正弦余弦位置编码
    Args:
        cx: (B, N) 归一化中心 x 坐标，∈ [0, 1]
        cy: (B, N) 归一化中心 y 坐标，∈ [0, 1]
        C:  int, 特征维度（必须被 4 整除）
        temperature: float, 频率缩放温度
    Returns:
        pe: (B, N, C) 位置编码
    """
    assert C % 4 == 0, f"C={C} 必须被 4 整除"
    half_C = C // 2
    quarter_C = C // 4
    # 频率序列：低频到高频，覆盖从粗定位到细定位
    # shape: (quarter_C,)
    dim_t = torch.arange(quarter_C, dtype=torch.float32, device=cx.device)
    dim_t = temperature ** (2 * dim_t / half_C)  # 几何级数分布
    # cx, cy: (B, N) → (B, N, 1)
    cx = cx.unsqueeze(-1)  # (B, N, 1)
    cy = cy.unsqueeze(-1)  # (B, N, 1)
    # 编码：坐标 / 频率 → sin, cos 交替
    # (B, N, 1) / (quarter_C,) → (B, N, quarter_C)
    pe_cx_sin = torch.sin(cx / dim_t)  # (B, N, quarter_C)
    pe_cx_cos = torch.cos(cx / dim_t)  # (B, N, quarter_C)
    pe_cy_sin = torch.sin(cy / dim_t)  # (B, N, quarter_C)
    pe_cy_cos = torch.cos(cy / dim_t)  # (B, N, quarter_C)
    # 拼接：[sin_x, cos_x, sin_y, cos_y]，总维度 = 4 × quarter_C = C
    pe = torch.cat([pe_cx_sin, pe_cx_cos, pe_cy_sin, pe_cy_cos], dim=-1)  # (B, N, C)
    return pe