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
    return max(0, TE)  # 理论上 TE >= 0


# ==========================
#     数据处理函数
# ==========================

def load_tracking_signals(seq_dir):
    """
    从 .npz 文件序列中提取 X (Offset) 和 Y (HeatMap) 信号
    """
    file_list = sorted(glob.glob(os.path.join(seq_dir, "*.npz")))
    if not file_list:
        raise FileNotFoundError(f"No .npz files found in {seq_dir}")

    Y_seq = []  # HeatMap Confidence
    X_seq = []  # Offset Magnitude

    print(f"Loading {len(file_list)} frames from {seq_dir}...")

    for file_path in file_list:
        with np.load(file_path) as data:
            heatmap = data['heatmap']  # (H, W) or (1, H, W)
            offset = data['offset']  # (H, W, 2)

            if heatmap.ndim == 3: heatmap = heatmap.squeeze()
            if offset.ndim == 3 and offset.shape[0] == 1: offset = offset.squeeze()

            # 策略：取 HeatMap 全局最大值点作为目标位置
            # (实际应用中可能需要更复杂的 Top-k 或上一帧位置引导)
            peak_loc = np.unravel_index(np.argmax(heatmap), heatmap.shape)
            y_idx, x_idx = peak_loc

            # 提取 Y: 响应强度
            conf_val = heatmap[y_idx, x_idx]
            Y_seq.append(conf_val)

            # 提取 X: 偏移模长 (代表运动强度)
            off_vec = offset[y_idx, x_idx]
            off_mag = np.sqrt(off_vec[0] ** 2 + off_vec[1] ** 2)
            X_seq.append(off_mag)

    X = np.array(X_seq)
    Y = np.array(Y_seq)

    # 简单归一化，利于 KDE 计算
    X = (X - np.mean(X)) / (np.std(X) + 1e-6)
    Y = (Y - np.mean(Y)) / (np.std(Y) + 1e-6)

    return X, Y


def build_windows(x, y, yr, n, dt):
    """
    构造滑动窗口数据，用于随时间计算 TE
    """
    length = len(x)
    times = []
    data = {k: [] for k in ["X_fut", "X_past", "Y_fut", "Y_past", "Yr_fut", "Yr_past"]}

    for t in range(n + dt, length):
        # 截取窗口
        x_past = x[t - n - dt: t - dt]
        y_fut = y[t]
        y_past = y[t - n: t]

        yr_fut = yr[t]
        yr_past = yr[t - n: t]

        if len(x_past) != n: continue

        times.append(t)
        data["X_past"].append(x_past)
        data["Y_fut"].append(y_fut)
        data["Y_past"].append(y_past)
        data["Yr_fut"].append(yr_fut)
        data["Yr_past"].append(yr_past)
        # 此处省略反向因果所需的 X_fut，如需可添加

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

def analyze_tracking_causality(seq_dir, output_dir="results"):
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
    n_default = 3  # 默认窗口
    dt_default = 1  # 默认延迟

    times, win_data = build_windows(X, Y, Yr, n_default, dt_default)

    TEd = np.zeros(length)  # Offset -> HeatMap
    TEr = np.zeros(length)  # Offset -> Randomized HeatMap

    # 最小样本数，太少无法准确估计 KDE
    min_samples = 30

    print("Calculating Transfer Entropy over time...")
    for k in range(len(times)):
        t_idx = times[k]  # 原始帧索引

        # 使用 t 之前的所有历史数据来估计分布
        # 注意：这里是 accumulated history，模拟在线过程
        curr_N = k + 1
        if curr_N < min_samples: continue

        # 提取切片
        x_past_batch = win_data["X_past"][:curr_N]
        y_past_batch = win_data["Y_past"][:curr_N]
        y_fut_batch = win_data["Y_fut"][:curr_N]

        yr_past_batch = win_data["Yr_past"][:curr_N]
        yr_fut_batch = win_data["Yr_fut"][:curr_N]

        # 计算 TE: Offset -> HeatMap
        TEd[t_idx] = transfer_entropy(x_past_batch, y_past_batch, y_fut_batch)

        # 计算 TE: Offset -> Randomized (Baseline)
        TEr[t_idx] = transfer_entropy(x_past_batch, yr_past_batch, yr_fut_batch)

        if k % 20 == 0:
            print(f"Frame {t_idx}: TE={TEd[t_idx]:.4f} (Base={TEr[t_idx]:.4f})")

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
        sample_te = TEd[i - win_len: i]
        sample_base = TEr[i - win_len: i]

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
    # 示例调用
    # 假设你的 npz 文件在 "./causal_data/basketball"
    seq_path = "/data2/lqh/workspace_pycharm/MCITrack/vis/chotrack_b224_got_a100/Biker/causal_data"
    if os.path.exists(seq_path):
        analyze_tracking_causality(seq_path)
    else:
        print(f"Path {seq_path} does not exist. Please generate data first.")