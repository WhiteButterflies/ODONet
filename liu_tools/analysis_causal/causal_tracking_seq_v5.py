#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Causal Analysis for Visual Tracking: Offset -> HeatMap
基于 ICCV 2015 Lebeda et al. 方法，分析目标跟踪中 Offset 场对 HeatMap 响应的因果关系。

输入: 包含 heatmap, offset, box 的 .npz 文件序列
输出: TE 曲线、显著性检验结果、最佳时空参数 (dt, n)
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, ttest_ind




# ==========================
#    工具函数：熵 & TE (保持不变)
# ==========================

def differential_entropy_kde(samples, bw_method="scott"):
    """
    使用 Gaussian KDE 估计连续随机变量的微分熵 H(Z)
    samples: (N, D) numpy array，每行一个样本
    """
    samples = np.asarray(samples)
    if samples.ndim == 1:
        samples = samples[:, None]

    # 添加微小噪声防止奇异矩阵，如果数据完全一样 KDE 会报错
    if np.std(samples) < 1e-6:
        samples += np.random.normal(0, 1e-6, samples.shape)

    try:
        kde = gaussian_kde(samples.T, bw_method=bw_method)
        log_p = kde.logpdf(samples.T)  # shape (N,)
        return -np.mean(log_p)
    except np.linalg.LinAlgError:
        return 0.0  # 处理数值不稳定情况


def transfer_entropy(X_past, Y_past, Y_future, bw_method="scott"):
    """
    估计 TE(X -> Y) = H(Y_future | Y_past) - H(Y_future | Y_past, X_past)
    """
    X_past = np.asarray(X_past)
    Y_past = np.asarray(Y_past)
    Y_future = np.asarray(Y_future)

    if X_past.ndim == 1: X_past = X_past[:, None]
    if Y_past.ndim == 1: Y_past = Y_past[:, None]
    if Y_future.ndim == 1: Y_future = Y_future[:, None]

    # 拼接各种联合变量
    Z_yf_yp = np.hstack([Y_future, Y_past])
    Z_yp = Y_past
    Z_yf_yp_xp = np.hstack([Y_future, Y_past, X_past])
    Z_yp_xp = np.hstack([Y_past, X_past])

    H_yf_yp = differential_entropy_kde(Z_yf_yp, bw_method=bw_method)
    H_yp = differential_entropy_kde(Z_yp, bw_method=bw_method)
    H_yf_yp_xp = differential_entropy_kde(Z_yf_yp_xp, bw_method=bw_method)
    H_yp_xp = differential_entropy_kde(Z_yp_xp, bw_method=bw_method)

    TE = (H_yf_yp - H_yp) - (H_yf_yp_xp - H_yp_xp)
    # return max(0, TE)  # 理论上 TE >= 0
    return TE  # 理论上 TE >= 0


# ==========================
#     数据处理函数
# ==========================
def topk_peaks(heatmap, k=3):
    """
    返回 top-k 峰的 (y, x, conf) 列表，按 conf 从大到小排序。
    不做局部极大值约束，如果有需要可以再加 NMS。
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
                # 用 0 填充
                for _ in range(K - len(peaks)):
                    peaks.append((0, 0, 0.0))

            feats_y = []
            for (y_idx, x_idx, conf) in peaks:
                y_norm = y_idx / max(H - 1, 1)
                x_norm = x_idx / max(W - 1, 1)
                feats_y.extend([conf, y_norm, x_norm])

            Y_seq.append(feats_y)  # 每帧 (3K,)

            # ===== X: 整个 offset 场的统计特征 =====
            # offset: (H, W, 2)
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
    构造滑动窗口数据，用于随时间计算 TE
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
        for j in range(1, cols):  # j=0 (n=1) 时无更短窗口，ri=0
            # 找到当前 n 之前 (n' < n) 的最大 TE
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

    # --------------------------
    # 1. 加载数据
    # --------------------------
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

    # --------------------------
    # 2. 随时间计算 TE
    # --------------------------
    n_default = 1  # 默认窗口
    dt_default = 0  # 默认延迟

    times, win_data = build_windows(X, Y, Yr, n_default, dt_default)

    TEd = np.zeros(length)  # Offset -> HeatMap
    TEr = np.zeros(length)  # Offset -> Randomized HeatMap

    # 最小样本数，太少无法准确估计 KDE
    min_samples = 10

    print("Calculating Transfer Entropy over time...")
    window_N = 10 # 例如在最近 100 个样本上估一个局部 TE

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

        # if k % 20 == 0:
        #     print(f"Frame {t_idx}: TE={TEd[t_idx]:.4f} (Base={TEr[t_idx]:.4f})")

        # if  TEd[t_idx]>0:
        if  True:
            TE_List.append(f"frame{t_idx},TE={TEd[t_idx]:.4f}")
            # print(f"TE>0,frame{t_idx},TE={TEd[t_idx]:.4f}")

    #storage TE result
    filename = os.path.join(output_dir, "te.txt")
    mode = 'w'
    try:
        with open(filename, mode, encoding='utf-8') as file:
            if isinstance(TE_List, list):
                # 如果是列表，写入多行
                for line in TE_List:
                    file.write(str(line) + '\n')
            else:
                # 如果是字符串，直接写入
                file.write(str(TE_List))
        print(f"成功写入文件: {filename}")
    except Exception as e:
        print(f"写入文件时出错: {e}")

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

    # --------------------------
    # 3. 显著性检验 (Welch t-test)
    # --------------------------
    # 寻找第一帧显著因果
    p_values = np.ones(length)
    win_len = 50  # 检验窗口大小

    start_idx = np.where(TEd > 0)[0][0] if np.any(TEd > 0) else 0

    for i in range(start_idx + win_len, length):
        center = i
        left = max(0,center-win_len//2)
        right = center

        sample_te = TEd[left: right]
        sample_base = TEr[left: right]

        # 如果方差极小，跳过
        if np.std(sample_te) < 1e-6 or np.std(sample_base) < 1e-6:
            continue

        _, p = ttest_ind(sample_te, sample_base, equal_var=False)
        p_values[i] = p

    # 绘制 P-value
    plt.figure(figsize=(10, 4))
    plt.semilogy(frames, p_values, label="p-value")
    plt.axhline(1e-4, color='r', linestyle='--', label="Threshold (1e-4)")
    plt.title("Statistical Significance of Causality")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "significance.png"))
    plt.close()

    sig_frames = np.where(p_values < 1e-4)[0]
    if len(sig_frames) > 0:
        first_sig_frame = sig_frames[0]
        print(f"Significant causality detected starting at frame {first_sig_frame}")
    else:
        print("No significant causality detected.")
        first_sig_frame = start_idx  # 如果没检测到，就用有数据的起点做网格搜索

    # --------------------------
    # 4. 参数网格搜索 (dt, n)
    # --------------------------
    print("Grid searching optimal parameters...")
    dt_range = np.arange(0, 6)  # lag: 0 to 5 frames
    n_range = np.arange(1, 6)  # window: 1 to 5 frames

    TE_grid = np.zeros((len(dt_range), len(n_range)))

    # 使用所有有效数据进行网格搜索（也可以仅使用 first_sig_frame 之后的数据）
    valid_data_start = first_sig_frame

    for i, dti in enumerate(dt_range):
        for j, ni in enumerate(n_range):
            # 构造当前参数的数据集
            _, g_data = build_windows(X, Y, Yr, ni, dti)

            # 截取有效部分
            x_p = np.array(g_data["X_past"])
            y_p = np.array(g_data["Y_past"])
            y_f = np.array(g_data["Y_fut"])

            # 仅使用显著帧之后的数据计算平均 TE
            # 注意需要对齐索引，这里简化处理，直接用全量数据的平均 TE
            if len(y_f) > 50:
                val = transfer_entropy(x_p, y_p, y_f)
                TE_grid[i, j] = val
            else:
                TE_grid[i, j] = 0

    # 计算相对提升
    rel_impr = relative_improvement(TE_grid)

    # 绘制网格
    plt.figure(figsize=(6, 5))
    plt.imshow(TE_grid, origin='lower', aspect='auto',
               extent=[n_range[0] - 0.5, n_range[-1] + 0.5, dt_range[0] - 0.5, dt_range[-1] + 0.5])
    plt.colorbar(label="Average TE")
    plt.xlabel("Window Size (n)")
    plt.ylabel("Time Lag (dt)")

    # 标出最佳点
    best_idx = np.unravel_index(np.argmax(TE_grid), TE_grid.shape)
    # 或者使用相对提升阈值筛选
    valid_opts = np.argwhere(rel_impr > 0.05)  # 阈值 5%
    if len(valid_opts) > 0:
        # 在满足提升阈值的点中选 TE 最大的
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
    # seq_list = ['Basketball','Bird1','BlurCar2','BlurCar3','BlurCar4','Board','Bolt']
    # seq_list = ['Bolt2']
    # seq_list = ['Car4','CarScale','Coupon','Deer','Dog']
    # seq_list = ['Football']
    seq_list = ['volleyball-13']
    # seq_list = ['Box']
    for seq_name in seq_list:

        # 示例调用
        # 假设你的 npz 文件在 "./causal_data/basketball"
        seq_path = "/data2/lqh/workspace_pycharm/MCITrack/vis/chotrackV1Mem_b224/{}/causal_data".format(seq_name)
        # seq_path = "/data2/lqh/workspace_pycharm/MCITrack/vis/chotrack_b224_got_a100/Biker/causal_data"
        if os.path.exists(seq_path):
            analyze_tracking_causality(seq_path,output_dir=str(seq_name))
        else:
            print(f"Path {seq_path} does not exist. Please generate data first.")