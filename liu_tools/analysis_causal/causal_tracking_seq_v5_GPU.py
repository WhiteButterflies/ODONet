#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Causal Analysis for Visual Tracking: Offset -> HeatMap (GPU Accelerated)
基于 ICCV 2015 Lebeda et al. 方法，分析目标跟踪中 Offset 场对 HeatMap 响应的因果关系。

修改说明:
- 引入 PyTorch 进行 GPU 加速 (KDE 估计)。
- 增加 USE_GPU 开关。

输入: 包含 heatmap, offset, box 的 .npz 文件序列
输出: TE 曲线、显著性检验结果、最佳时空参数 (dt, n)
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, ttest_ind
import time

# 尝试导入 PyTorch
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not found. GPU acceleration is disabled.")

# ==========================
#    配置 & 工具函数
# ==========================

# 全局开关，可在 main 中修改
USE_GPU = True 

def get_device():
    if USE_GPU and TORCH_AVAILABLE and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def gaussian_kde_entropy_gpu(samples, bw_method="scott"):
    """
    使用 PyTorch (Float64) 计算微分熵。
    改进点：
    1. 使用 float64 避免精度误差。
    2. 优化奇异值处理逻辑。
    """
    device = get_device()
    
    # 强制使用 float64 (Double) 以保证与 Scipy 精度一致
    X = torch.as_tensor(samples, dtype=torch.float64, device=device)
    if X.ndim == 1:
        X = X.unsqueeze(1)
    
    N, D = X.shape
    
    # 1. 预处理：标准化防止数值溢出 (均值归0，方差不归一化以免改变熵的相对值，但在计算协方差时数值更稳)
    # 这里只做中心化，不缩放，因为缩放会改变熵值 (H(aX) = H(X) + log(a))
    # 但由于我们计算的是 TE (差值)，只要统一处理是可以的。为了严谨，我们仅中心化。
    X_mean = X.mean(dim=0, keepdim=True)
    X = X - X_mean

    # 2. 计算协方差矩阵
    if N > 1:
        # torch.cov 默认归一化因子是 N-1 (unbiased)，与 scipy 一致
        X_t = X.T
        cov = torch.cov(X_t)
    else:
        cov = torch.eye(D, device=device, dtype=torch.float64) * 1e-6

    # 处理 1D 情况 torch.cov 返回 0维 tensor
    if cov.ndim == 0:
        cov = cov.view(1, 1)

    # 3. 带宽计算 (Scott's Rule)
    if bw_method == "scott":
        factor = N ** (-1.0 / (D + 4))
    else:
        factor = 0.5 
        
    kde_cov = cov * (factor ** 2)
    
    # 4. Cholesky 分解与稳定性处理
    # 使用极小的 jitter (1e-8) 仅用于防止崩溃，尽量不影响结果
    jitter = 1e-8
    eye = torch.eye(D, device=device, dtype=torch.float64) * jitter
    
    try:
        # 尝试 Cholesky
        L = torch.linalg.cholesky(kde_cov + eye)
    except RuntimeError:
        # 如果失败（矩阵严重奇异），回退到对角阵近似 (Diagonal fallback)
        # 这相当于假设各维度独立，虽然不完美，但比返回 0 或崩溃要好
        # 增加 jitter 强度以确保通过
        L = torch.diag(torch.sqrt(torch.diagonal(kde_cov) + 1e-6))

    # Log determinant: 2 * sum(log(diag(L)))
    log_det = 2 * torch.sum(torch.log(torch.diagonal(L)))

    # 5. 白化与距离计算
    # Y = X @ inv(L).T
    # 求解 L * Y.T = X.T -> Y.T = L \ X.T -> Y = (L \ X.T).T
    try:
        Y_t = torch.linalg.solve_triangular(L, X.T, upper=False)
        Y = Y_t.T
    except RuntimeError:
        # 如果求解失败，使用伪逆
        L_inv = torch.linalg.pinv(L)
        Y = X @ L_inv.T

    # 计算成对距离矩阵 (Squared Euclidean)
    # ||y_i - y_j||^2
    # 使用 float64 精度计算
    dists = torch.cdist(Y, Y, p=2) ** 2

    # 6. 计算熵
    # H(X) = -mean( log( 1/N * sum( exp(-0.5 * dists) ) * const ) )
    #      = -mean( logsumexp(-0.5 * dists) - log(N) + log_const )
    
    const_term = -0.5 * D * np.log(2 * np.pi) - 0.5 * log_det - np.log(N)
    
    # logsumexp
    log_probs = torch.logsumexp(-0.5 * dists, dim=1) + const_term
    
    entropy = -torch.mean(log_probs)
    
    return entropy.item()

def differential_entropy_kde(samples, bw_method="scott"):
    # 自动切换
    if USE_GPU and TORCH_AVAILABLE and torch.cuda.is_available():
        return gaussian_kde_entropy_gpu(samples, bw_method)
    else:
        # CPU 回退
        from scipy.stats import gaussian_kde
        samples = np.asarray(samples)
        if samples.ndim == 1: samples = samples[:, None]
        try:
            kde = gaussian_kde(samples.T, bw_method=bw_method)
            return -np.mean(kde.logpdf(samples.T))
        except Exception:
            return 0.0

def transfer_entropy(X_past, Y_past, Y_future, bw_method="scott"):
    X_past = np.asarray(X_past)
    Y_past = np.asarray(Y_past)
    Y_future = np.asarray(Y_future)

    if X_past.ndim == 1: X_past = X_past[:, None]
    if Y_past.ndim == 1: Y_past = Y_past[:, None]
    if Y_future.ndim == 1: Y_future = Y_future[:, None]

    Z_yf_yp = np.hstack([Y_future, Y_past])
    Z_yp = Y_past
    Z_yf_yp_xp = np.hstack([Y_future, Y_past, X_past])
    Z_yp_xp = np.hstack([Y_past, X_past])

    H_yf_yp = differential_entropy_kde(Z_yf_yp, bw_method=bw_method)
    H_yp = differential_entropy_kde(Z_yp, bw_method=bw_method)
    H_yf_yp_xp = differential_entropy_kde(Z_yf_yp_xp, bw_method=bw_method)
    H_yp_xp = differential_entropy_kde(Z_yp_xp, bw_method=bw_method)

    TE = (H_yf_yp - H_yp) - (H_yf_yp_xp - H_yp_xp)
    return max(0, TE)


# ==========================
#     数据处理函数 (保持不变)
# ==========================
def topk_peaks(heatmap, k=3):
    """
    返回 top-k 峰的 (y, x, conf) 列表，按 conf 从大到小排序。
    """
    H, W = heatmap.shape
    flat = heatmap.reshape(-1)
    k = min(k, flat.size)
    idxs = np.argpartition(flat, -k)[-k:]
    idxs = idxs[np.argsort(-flat[idxs])]  # 从大到小排序

    peaks = []
    for idx in idxs:
        y = idx // W
        x = idx % W
        conf = heatmap[y, x]
        peaks.append((y, x, conf))
    return peaks

def load_tracking_signals(seq_dir, K=1):
    """
    从 .npz 文件序列中提取 X (Offset 全场统计) 和 Y (HeatMap top-K 峰) 信号
    """
    file_list = sorted(glob.glob(os.path.join(seq_dir, "*.npz")))
    if not file_list:
        raise FileNotFoundError(f"No .npz files found in {seq_dir}")

    Y_seq = []  # HeatMap 多峰特征
    X_seq = []  # Offset 全场方向/强度特征

    print(f"Loading {len(file_list)} frames from {seq_dir}...")

    for file_path in file_list:
        with np.load(file_path) as data:
            heatmap = data['heatmap']  # (H, W) or (1, H, W)
            offset = data['offset']    # (H, W, 2) or (1, H, W, 2)

            if heatmap.ndim == 3:
                heatmap = heatmap.squeeze()
            if offset.ndim == 4 and offset.shape[0] == 1:
                offset = offset.squeeze(0)
            if offset.ndim == 3 and offset.shape[0] == 1:
                offset = offset.squeeze(0)

            H, W = heatmap.shape

            # ===== Y: top-K 峰 (主峰 + 次峰) =====
            peaks = topk_peaks(heatmap, k=K)

            # 如果不足 K 个，就 pad
            if len(peaks) < K:
                for _ in range(K - len(peaks)):
                    peaks.append((0, 0, 0.0))

            feats_y = []
            for (y_idx, x_idx, conf) in peaks:
                y_norm = y_idx / max(H - 1, 1)
                x_norm = x_idx / max(W - 1, 1)
                feats_y.extend([conf, y_norm, x_norm])

            Y_seq.append(feats_y)  # 每帧 (3K,)

            # ===== X: 整个 offset 场的统计特征 =====
            off_vecs = offset.reshape(-1, offset.shape[-1])  # (HW, 2)
            mean_vec = off_vecs.mean(axis=0)                 # (2,)
            mean_mag = np.linalg.norm(off_vecs, axis=1).mean()

            # X 可以选 [mean_u, mean_v, mean_mag]
            feats_x = np.concatenate([mean_vec, [mean_mag]])  # (3,)

            X_seq.append(feats_x)

    X = np.array(X_seq)   # (T, 3)
    Y = np.array(Y_seq)   # (T, 3K)

    # ===== 归一化 =====
    X_mean = X.mean(axis=0, keepdims=True)
    X_std  = X.std(axis=0, keepdims=True) + 1e-6
    X = (X - X_mean) / X_std

    Y_mean = Y.mean(axis=0, keepdims=True)
    Y_std  = Y.std(axis=0, keepdims=True) + 1e-6
    Y = (Y - Y_mean) / Y_std

    return X, Y


def build_windows(x, y, yr, n, dt):
    """
    x: (T, Dx)
    y, yr: (T, Dy)
    构造滑动窗口数据
    """
    length = len(x)
    times = []
    data = {k: [] for k in ["X_fut", "X_past", "Y_fut", "Y_past", "Yr_fut", "Yr_past"]}

    for t in range(n + dt, length):
        x_past = x[t - n - dt: t - dt]  # (n, Dx) 或 (n,)
        y_fut  = y[t]                   # (Dy,)
        y_past = y[t - n: t]            # (n, Dy)

        yr_fut  = yr[t]
        yr_past = yr[t - n: t]

        if len(x_past) != n:
            continue

        # ---- 展平时间维 ----
        x_past_flat  = x_past.reshape(-1)   # (n * Dx,)
        y_past_flat  = y_past.reshape(-1)   # (n * Dy,)
        yr_past_flat = yr_past.reshape(-1)

        times.append(t)
        data["X_past"].append(x_past_flat)
        data["Y_fut"].append(y_fut)
        data["Y_past"].append(y_past_flat)
        data["Yr_fut"].append(yr_fut)
        data["Yr_past"].append(yr_past_flat)

    for k in data:
        data[k] = np.array(data[k])

    return np.array(times), data


def relative_improvement(TEgrid):
    """ 计算相对提升矩阵 """
    TEgrid = np.asarray(TEgrid)
    if np.all(np.isnan(TEgrid)): return np.zeros_like(TEgrid)

    Tmax = np.nanmax(TEgrid)
    if Tmax == 0: return np.zeros_like(TEgrid)

    ri = np.zeros_like(TEgrid)
    rows, cols = TEgrid.shape

    for i in range(rows):
        for j in range(1, cols):
            prev_vals = TEgrid[i, :j]
            if len(prev_vals) > 0 and not np.all(np.isnan(prev_vals)):
                best_prev = np.nanmax(prev_vals)
                ri[i, j] = (TEgrid[i, j] - best_prev) / Tmax
    return ri


# ==========================
#           主流程
# ==========================

def analyze_tracking_causality(seq_dir, output_dir="results5"):
    os.makedirs(output_dir, exist_ok=True)

    # 1. 加载数据
    try:
        X, Y = load_tracking_signals(seq_dir)
        # 构造打乱的 Y 用于基准测试
        Yr = np.random.permutation(Y)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    length = len(X)
    frames = np.arange(length)

    # 画信号图
    plt.figure(figsize=(10, 4))
    plt.plot(frames, X, label="Offset (X)", alpha=0.7)
    plt.plot(frames, Y, label="HeatMap (Y)", alpha=0.7)
    plt.title(f"Normalized Signals: {os.path.basename(seq_dir)}")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "signals.png"))
    plt.close()

    # 2. 随时间计算 TE
    n_default = 1
    dt_default = 0

    times, win_data = build_windows(X, Y, Yr, n_default, dt_default)

    TEd = np.zeros(length)
    TEr = np.zeros(length)
    min_samples = 10
    
    # 扩大一点 window_N 以利用 GPU 并行优势
    window_N = 10 
    
    print(f"Calculating Transfer Entropy over time... (Mode: {'GPU' if USE_GPU and TORCH_AVAILABLE else 'CPU'})")
    start_time = time.time()
    
    TE_List = []

    for k in range(len(times)):
        t_idx = times[k]
        end = k + 1
        start = max(0, end - window_N)

        curr_N = end - start
        if curr_N < min_samples:
            continue

        x_past_batch = win_data["X_past"][start:end]
        y_past_batch = win_data["Y_past"][start:end]
        y_fut_batch = win_data["Y_fut"][start:end]

        yr_past_batch = win_data["Yr_past"][start:end]
        yr_fut_batch = win_data["Yr_fut"][start:end]

        TEd[t_idx] = transfer_entropy(x_past_batch, y_past_batch, y_fut_batch)
        TEr[t_idx] = transfer_entropy(x_past_batch, yr_past_batch, yr_fut_batch)

        if TEd[t_idx] > 0:
            TE_List.append(f"TE>0,frame{t_idx},TE={TEd[t_idx]:.4f}")
            # 减少 print 频率，防止 I/O 拖慢 GPU
            if len(TE_List) % 10 == 0:
                print(f"TE>0,frame{t_idx},TE={TEd[t_idx]:.4f}")

    print(f"TE Calculation Time: {time.time() - start_time:.2f}s")

    # storage TE result
    filename = os.path.join(output_dir, "te.txt")
    try:
        with open(filename, 'w', encoding='utf-8') as file:
            for line in TE_List:
                file.write(str(line) + '\n')
    except Exception as e:
        print(f"Error writing file: {e}")

    # 绘制 TE 曲线
    plt.figure(figsize=(10, 4))
    plt.plot(frames, TEd, label=r"$Offset \rightarrow HeatMap$", color='b')
    plt.plot(frames, TEr, label=r"$Offset \rightarrow Random$", color='gray', alpha=0.5)
    plt.xlabel("Frame")
    plt.ylabel("TE (bits)")
    plt.title("Causal Strength over Time")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "te_curve.png"))
    plt.close()

    # 3. 显著性检验
    p_values = np.ones(length)
    win_len = 50
    start_idx = np.where(TEd > 0)[0][0] if np.any(TEd > 0) else 0

    for i in range(start_idx + win_len, length):
        center = i
        left = max(0,center-win_len//2)
        right = center
        sample_te = TEd[left: right]
        sample_base = TEr[left: right]
        if np.std(sample_te) < 1e-6 or np.std(sample_base) < 1e-6:
            continue
        _, p = ttest_ind(sample_te, sample_base, equal_var=False)
        p_values[i] = p

    plt.figure(figsize=(10, 4))
    plt.semilogy(frames, p_values, label="p-value")
    plt.axhline(1e-4, color='r', linestyle='--', label="Threshold (1e-4)")
    plt.title("Statistical Significance of Causality")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "significance.png"))
    plt.close()

    sig_frames = np.where(p_values < 1e-4)[0]
    first_sig_frame = sig_frames[0] if len(sig_frames) > 0 else start_idx
    if len(sig_frames) > 0:
        print(f"Significant causality detected starting at frame {first_sig_frame}")
    else:
        print("No significant causality detected.")

    # 4. 参数网格搜索 (dt, n)
    print("Grid searching optimal parameters...")
    dt_range = np.arange(0, 6)
    n_range = np.arange(1, 6)
    TE_grid = np.zeros((len(dt_range), len(n_range)))

    # 网格搜索也使用 GPU 加速
    for i, dti in enumerate(dt_range):
        for j, ni in enumerate(n_range):
            _, g_data = build_windows(X, Y, Yr, ni, dti)
            x_p = np.array(g_data["X_past"])
            y_p = np.array(g_data["Y_past"])
            y_f = np.array(g_data["Y_fut"])
            
            # 使用较多样本进行稳定估计
            if len(y_f) > 50:
                # 可以选择只取显著性区间的数据，或者全量数据
                # 随机采样 1000 个样本防止显存爆炸（如果数据量极大）
                if len(y_f) > 2000 and USE_GPU:
                    idx = np.random.choice(len(y_f), 2000, replace=False)
                    TE_grid[i, j] = transfer_entropy(x_p[idx], y_p[idx], y_f[idx])
                else:
                    TE_grid[i, j] = transfer_entropy(x_p, y_p, y_f)
            else:
                TE_grid[i, j] = 0

    rel_impr = relative_improvement(TE_grid)

    plt.figure(figsize=(6, 5))
    plt.imshow(TE_grid, origin='lower', aspect='auto',
               extent=[n_range[0] - 0.5, n_range[-1] + 0.5, dt_range[0] - 0.5, dt_range[-1] + 0.5])
    plt.colorbar(label="Average TE")
    plt.xlabel("Window Size (n)")
    plt.ylabel("Time Lag (dt)")

    best_idx = np.unravel_index(np.argmax(TE_grid), TE_grid.shape)
    valid_opts = np.argwhere(rel_impr > 0.05)
    if len(valid_opts) > 0:
        vals = [TE_grid[r, c] for r, c in valid_opts]
        best_in_valid = valid_opts[np.argmax(vals)]
        best_dt_idx, best_n_idx = best_in_valid
    else:
        best_dt_idx, best_n_idx = best_idx

    plt.scatter(n_range[best_n_idx], dt_range[best_dt_idx], s=200, c='red', marker='*')
    plt.title(f"Optimal: dt={dt_range[best_dt_idx]}, n={n_range[best_n_idx]}")
    plt.savefig(os.path.join(output_dir, "param_grid.png"))
    plt.close()

    print(f"Analysis Complete. Optimal Parameters: dt={dt_range[best_dt_idx]}, n={n_range[best_n_idx]}")
    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    
    # === 配置区域 ===
    # True: 使用 GPU (需安装 torch 且有 CUDA)
    # False: 使用 CPU (原版 scipy 逻辑)
    USE_GPU = True  
    
    # seq_list = ['Basketball','Bird1','BlurCar2','BlurCar3','BlurCar4','Board','Bolt']
    seq_list = ['Basketball']
    
    for seq_name in seq_list:
        # 修改为实际路径
        seq_path = "/data2/lqh/workspace_pycharm/MCITrack/vis/chotrack_b224_got_a100/{}/causal_data".format(seq_name)
        
        if os.path.exists(seq_path):
            analyze_tracking_causality(seq_path, output_dir=str(seq_name))
        else:
            print(f"Path {seq_path} does not exist. Please generate data first.")