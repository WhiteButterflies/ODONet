#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Hanning Window Conflict Analysis
功能：计算加汉宁窗前后的 HeatMap 峰值坐标距离。
逻辑：Distance = || Peak_Raw - Peak_Windowed ||
解读：
  - Distance == 0: 网络非常自信且目标在中心，状态稳定。
  - Distance > 0:  网络在边缘发现高响应，但被汉宁窗抑制。这是不稳定的前兆。
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt

def analyze_hanning_conflict(seq_dir, output_dir="results_conflict"):
    os.makedirs(output_dir, exist_ok=True)
    
    file_list = sorted(glob.glob(os.path.join(seq_dir, "*.npz")))
    if not file_list:
        raise FileNotFoundError(f"No .npz files found in {seq_dir}")

    print(f"Loading {len(file_list)} frames from {seq_dir}...")

    frames = []
    shifts = []
    peak_raw_vals = []
    peak_han_vals = []

    for i, file_path in enumerate(file_list):
        try:
            with np.load(file_path) as data:
                # 1. 加载数据 (根据你的截图，key 分别为 'heatmap' 和 'heatmap_han')
                # heatmap: 原始无窗 (Raw Network Output)
                # heatmap_han: 加窗后 (Final Tracking Score)
                hm_raw = data['heatmap']
                hm_han = data['heatmap_han']
                
                # 处理维度 (1, H, W) -> (H, W)
                if hm_raw.ndim == 3: hm_raw = hm_raw.squeeze()
                if hm_han.ndim == 3: hm_han = hm_han.squeeze()
                
                # 2. 寻找峰值坐标 (y, x)
                # 原始峰值
                raw_flat_idx = np.argmax(hm_raw)
                raw_y, raw_x = np.unravel_index(raw_flat_idx, hm_raw.shape)
                raw_val = hm_raw[raw_y, raw_x]
                
                # 加窗峰值
                han_flat_idx = np.argmax(hm_han)
                han_y, han_x = np.unravel_index(han_flat_idx, hm_han.shape)
                han_val = hm_han[han_y, han_x]

                # 3. 计算位移距离 (欧氏距离)
                dist = np.sqrt((raw_x - han_x)**2 + (raw_y - han_y)**2)
                
                frames.append(i)
                shifts.append(dist)
                peak_raw_vals.append(raw_val)
                peak_han_vals.append(han_val)
                
        except Exception as e:
            print(f"Error processing frame {i}: {e}")
            continue

    frames = np.array(frames)
    shifts = np.array(shifts)
    
    # ==========================
    #       可视化结果
    # ==========================
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # 图 1: 峰值位移距离 (Peak Shift)
    axes[0].plot(frames, shifts, color='red', linewidth=1.5, label='Peak Shift Distance')
    axes[0].set_ylabel('Shift Distance (pixels)')
    axes[0].set_title('Hanning Window Suppression Effect (Raw Peak vs. Windowed Peak)')
    axes[0].grid(True, alpha=0.3)
    
    # 标记冲突时刻
    threshold = 5.0 # 设定一个阈值，比如 5 像素
    danger_indices = np.where(shifts > threshold)[0]
    if len(danger_indices) > 0:
        axes[0].scatter(frames[danger_indices], shifts[danger_indices], color='orange', s=20, zorder=5)
        # 仅在图中标记前几个关键点避免拥挤
        for idx in danger_indices[:5]: 
            axes[0].text(frames[idx], shifts[idx], f" Fr{frames[idx]}", fontsize=9)

    # 图 2: 峰值强度对比
    axes[1].plot(frames, peak_raw_vals, label='Raw Peak Value', color='blue', alpha=0.6)
    axes[1].plot(frames, peak_han_vals, label='Windowed Peak Value', color='green', alpha=0.6)
    axes[1].set_ylabel('Response Score')
    axes[1].set_title('Confidence Drop due to Hanning Window')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.xlabel('Frame Index')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "hanning_conflict_analysis.png"))
    plt.close()
    
    # ==========================
    #       输出文本日志
    # ==========================
    log_path = os.path.join(output_dir, "unstable_frames.txt")
    with open(log_path, "w") as f:
        f.write("Frame, Shift_Distance, Raw_Val, Han_Val, Analysis\n")
        for i in range(len(frames)):
            shift = shifts[i]
            if shift > 0: # 只要有位移就记录，哪怕是 1 像素
                status = "Minor Correction"
                if shift > 10: status = "MAJOR CONFLICT (Drift/Fast Motion)"
                elif shift > 5: status = "Potential Instability"
                
                f.write(f"{frames[i]+1}, {shift:.2f}, {peak_raw_vals[i]:.4f}, {peak_han_vals[i]:.4f}, {status}\n")

    print(f"Analysis complete. Results saved to {output_dir}")

if __name__ == "__main__":
    # 配置你的序列路径
    seq_name = "Basketball"
    seq_name = "monkey-3"
    # seq_name = "Box"
    # 请修改为实际路径
    seq_path = f"/data2/lqh/workspace_pycharm/MCITrack/vis/chotrackV1Mem_b224/{seq_name}/causal_data"
    
    if os.path.exists(seq_path):
        analyze_hanning_conflict(seq_path, output_dir=f"results_simple_chotrackV1Mem_{seq_name}")
    else:
        print("Path not found.")