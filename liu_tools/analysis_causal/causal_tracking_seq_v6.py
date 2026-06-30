#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Semantic Causal Analysis for Visual Tracking (v6)
改进点：
1. 信号提取：从“全图统计”改为“中心对抗模型” (Centrifugal Force vs Response Gain).
2. 引入冲突指数 (Conflict Index) 以捕捉行为突变节点.
3. 增强的可解释性可视化.
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, ttest_ind, entropy

# ==========================
#    数学工具函数 (TE & Entropy)
# ==========================

def differential_entropy_kde(samples, bw_method="scott"):
    """ 计算微分熵 H(Z) """
    samples = np.asarray(samples)
    if samples.ndim == 1:
        samples = samples[:, None]
    
    # 添加微小抖动防止奇异矩阵
    if np.std(samples) < 1e-6:
        samples += np.random.normal(0, 1e-6, samples.shape)

    try:
        kde = gaussian_kde(samples.T, bw_method=bw_method)
        # 为了速度，仅在样本点上评估
        log_p = kde.logpdf(samples.T)
        return -np.mean(log_p)
    except np.linalg.LinAlgError:
        return 0.0

def transfer_entropy(X_past, Y_past, Y_future):
    """ 计算 TE(X->Y) """
    # 维度调整
    X_past = np.atleast_2d(X_past).T if X_past.ndim == 1 else X_past
    Y_past = np.atleast_2d(Y_past).T if Y_past.ndim == 1 else Y_past
    Y_future = np.atleast_2d(Y_future).T if Y_future.ndim == 1 else Y_future

    # 联合分布构建
    Z_yf_yp = np.hstack([Y_future, Y_past])
    Z_yp = Y_past
    Z_yf_yp_xp = np.hstack([Y_future, Y_past, X_past])
    Z_yp_xp = np.hstack([Y_past, X_past])

    # 熵计算
    H_yf_yp = differential_entropy_kde(Z_yf_yp)
    H_yp = differential_entropy_kde(Z_yp)
    H_yf_yp_xp = differential_entropy_kde(Z_yf_yp_xp)
    H_yp_xp = differential_entropy_kde(Z_yp_xp)

    TE = (H_yf_yp - H_yp) - (H_yf_yp_xp - H_yp_xp)
    return max(0, TE)

# ==========================
#    新的信号提取逻辑
# ==========================

def create_hanning_window(shape):
    """ 生成与 HeatMap 同尺寸的汉宁窗 """
    H, W = shape
    hy = np.hanning(H)
    hx = np.hanning(W)
    return np.outer(hy, hx)

def extract_semantic_signals(seq_dir):
    """
    提取语义信号：
    X: 离心力 (Offset Centrifugal Force) - Offset 是否想逃离中心
    Y: 响应增益 (Response Gain) - Offset 指向位置的热度 vs 中心热度
    Aux: 冲突指数 (Conflict Index) - 捕捉异常节点
    """
    file_list = sorted(glob.glob(os.path.join(seq_dir, "*.npz")))
    if not file_list:
        raise FileNotFoundError(f"No .npz files found in {seq_dir}")

    X_seq = [] # 离心力
    Y_seq = [] # 响应增益
    
    # 辅助指标
    Conflict_seq = [] 
    Entropy_seq = []

    print(f"Processing {len(file_list)} frames...")

    for file_path in file_list:
        with np.load(file_path) as data:
            # 原始 HeatMap (无汉宁窗)
            heatmap = data['heatmap'] 
            offset = data['offset'] 

            if heatmap.ndim == 3: heatmap = heatmap.squeeze()
            if offset.ndim == 4: offset = offset.squeeze(0)
            if offset.ndim == 3 and offset.shape[0]==1: offset = offset.squeeze(0)

            H, W = heatmap.shape
            cy, cx = H // 2, W // 2 # 假设裁剪中心即为跟踪中心

            # 1. 提取中心区域的 Offset (代表跟踪器的运动意图)
            # 取中心 3x3 区域的平均，增加鲁棒性
            roi_slice_y = slice(max(0, cy-1), min(H, cy+2))
            roi_slice_x = slice(max(0, cx-1), min(W, cx+2))
            
            off_vectors = offset[roi_slice_y, roi_slice_x, :] # (3,3,2)
            avg_off_vec = np.mean(off_vectors.reshape(-1, 2), axis=0) # (dy, dx)
            
            # 2. 计算信号 X: 离心力 (Centrifugal Force)
            # 构建从中心向外的单位向量 (这里简化为 Offset 自身的模长，因为中心向外就是辐射状)
            # 更严谨的做法是：计算 Offset 向量 与 "中心指向Offset位置的向量" 的点积
            # 但在中心点，任何非零 Offset 实际上都是在"远离中心"
            
            # 这里我们使用：Offset 模长 * (Offset 指向边缘的程度)
            # 简化版：X = Offset 模长 (假设任何运动都是试图改变位置)
            # 增强版：X = Offset 模长，但如果 Offset 指回中心则为负? 
            # 考虑到跟踪通常是中心对其，Offset 只要大就是在"突围"。
            x_val = np.linalg.norm(avg_off_vec)

            # 3. 计算信号 Y: 响应增益 (Raw Response Gain)
            # Offset 指向的目标坐标
            target_y = int(np.clip(cy + avg_off_vec[0], 0, H-1))
            target_x = int(np.clip(cx + avg_off_vec[1], 0, W-1))

            val_center = heatmap[cy, cx] + 1e-6 # 防止除零
            val_target = heatmap[target_y, target_x] + 1e-6
            
            y_val = np.log(val_target / val_center) # Log ratio，使其更像高斯分布

            # 4. 计算辅助指标: 冲突指数 (Conflict Index)
            # 定义：Offset 越长 且 指向的位置汉宁窗惩罚越重 -> 冲突越大
            # 生成临时汉宁窗获取惩罚系数
            hanning_val_at_target = (np.hanning(H)[target_y] * np.hanning(W)[target_x])
            # 冲突 = 意图强度 * (1 - 汉宁支持度)
            conflict_val = x_val * (1.0 - hanning_val_at_target)

            # 5. 计算辅助指标: 空间熵 (Spatial Entropy)
            # 衡量 HeatMap 的混乱程度
            prob_map = np.abs(heatmap) / (np.sum(np.abs(heatmap)) + 1e-9)
            ent_val = entropy(prob_map.flatten())

            X_seq.append(x_val)
            Y_seq.append(y_val)
            Conflict_seq.append(conflict_val)
            Entropy_seq.append(ent_val)

    return np.array(X_seq), np.array(Y_seq), np.array(Conflict_seq), np.array(Entropy_seq)

# ==========================
#    分析与绘图
# ==========================

def analyze_behavior(seq_dir, output_dir="results_v6"):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 加载新信号
    try:
        X, Y, Conflict, Entropy = extract_semantic_signals(seq_dir)
    except Exception as e:
        print(f"Data load error: {e}")
        return

    frames = np.arange(len(X))

    # 标准化 X 和 Y 用于 TE 计算 (TE 对幅度敏感)
    X_norm = (X - np.mean(X)) / (np.std(X) + 1e-6)
    Y_norm = (Y - np.mean(Y)) / (np.std(Y) + 1e-6)

    # 2. 计算 TE (滑动窗口)
    te_vals = []
    window = 10 # 窗口稍大一点以获得稳定估计
    
    print("Calculating Semantic TE...")
    for t in range(len(frames)):
        if t < window:
            te_vals.append(0.0)
            continue
        
        # 提取局部窗口
        x_win = X_norm[t-window : t]
        y_win = Y_norm[t-window : t]
        
        # 简单的 lag=1 TE
        x_past = x_win[:-1]
        y_past = y_win[:-1]
        y_fut  = y_win[1:]
        
        val = transfer_entropy(x_past, y_past, y_fut)
        te_vals.append(val)
    
    te_vals = np.array(te_vals)

    # 3. 绘制综合分析图
    fig, axes = plt.subplots(4, 1, figsize=(12, 16), sharex=True)
    
    # 子图 1: TE (因果强度)
    axes[0].plot(frames, te_vals, color='blue', linewidth=1.5)
    axes[0].set_ylabel('TE (Bits)')
    axes[0].set_title('1. Causal Intervention Strength (Offset -> HeatMap Gain)')
    axes[0].grid(True, alpha=0.3)
    # 标记高 TE 区域
    high_te_mask = te_vals > np.percentile(te_vals, 90)
    axes[0].fill_between(frames, 0, te_vals, where=high_te_mask, color='blue', alpha=0.1, label="Strong Intervention")
    axes[0].legend()

    # 子图 2: 冲突指数 (Conflict Index - 捕捉节点)
    axes[1].plot(frames, Conflict, color='red', linewidth=1.5)
    axes[1].set_ylabel('Conflict Index')
    axes[1].set_title('2. Conflict Index (Offset vs. Hanning Window)')
    axes[1].grid(True, alpha=0.3)
    # 标记危险节点
    threshold = np.mean(Conflict) + 2 * np.std(Conflict)
    axes[1].axhline(threshold, color='orange', linestyle='--')
    axes[1].text(0, threshold, " Danger Zone", color='orange')

    # 子图 3: 原始信号 (离心力 vs 增益)
    axes[2].plot(frames, X, label='Offset Force (X)', color='green', alpha=0.7)
    axes[2].plot(frames, Y, label='Response Gain (Y)', color='purple', alpha=0.7)
    axes[2].set_ylabel('Signal Magnitude')
    axes[2].set_title('3. Semantic Signals: Force & Gain')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    # 子图 4: HeatMap 熵 (不确定性)
    axes[3].plot(frames, Entropy, color='gray', linewidth=1.5)
    axes[3].set_ylabel('Spatial Entropy')
    axes[3].set_title('4. HeatMap Uncertainty')
    axes[3].set_xlabel('Frame Index')
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "behavior_analysis.png"))
    plt.close()

    # 4. 输出关键节点日志
    # 寻找 "冲突高" 且 "TE高" 的时刻 (良性纠错) vs "冲突高" 且 "TE低" (恶性漂移)
    
    log_path = os.path.join(output_dir, "critical_events.txt")
    with open(log_path, "w") as f:
        f.write("Frame, Conflict, TE, Interpretation\n")
        
        # 简单的事件检测逻辑
        for t in range(window, len(frames)):
            is_high_conflict = Conflict[t] > threshold
            is_high_te = te_vals[t] > 0.1  # 阈值根据实际数据调整
            
            if is_high_conflict:
                status = "BENIGN CORRECTION" if is_high_te else "MALIGNANT DRIFT (RISK)"
                f.write(f"{t}, {Conflict[t]:.4f}, {te_vals[t]:.4f}, {status}\n")

    print(f"Analysis saved to {output_dir}")

if __name__ == "__main__":
    # 替换为你的数据路径
    # seq_name = "Basketball"
    # base_path = "./causal_data" 
    
    seq_list = ['Basketball']
    for seq_name in seq_list:
        # 修改这里的路径指向你的 .npz 文件夹
        seq_path = f"/data2/lqh/workspace_pycharm/MCITrack/vis/chotrack_b224_got_a100/{seq_name}/causal_data"
        
        if os.path.exists(seq_path):
            print(f"Analyzing {seq_name}...")
            analyze_behavior(seq_path, output_dir=f"results_v6_chotrackv2.2_{seq_name}")
        else:
            print(f"Path not found: {seq_path}")