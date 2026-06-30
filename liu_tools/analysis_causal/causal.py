import numpy as np
import pandas as pd
import os
def load_causal_sequence(seq_dir, start_frame, window_size):
    """
    读取时间窗口内的数据，用于计算 TE
    对应论文中的滑动窗口 past := t - dt -> t - n
    """
    data_buffer = []
    # 读取从 start_frame 回溯 window_size 的数据
    for i in range(window_size):
        curr_id = start_frame - i
        if curr_id < 0: break

        path = os.path.join(seq_dir, f"{str(curr_id).zfill(5)}.npz")
        if os.path.exists(path):
            with np.load(path) as data:
                # 提取对应论文 [cite: 480] 的信号: Y=[Pos, conf], X=[Pos, offset]
                data_buffer.append({
                    'heatmap': data['heatmap'],
                    'offset': data['offset'],
                    'id': data['frame_id']
                })
    return data_buffer  # 返回列表，包含 t, t-1, t-2... 的数据

window_size = 10
start_frame = window_size +1
buffer = load_causal_sequence('/data2/lqh/workspace_pycharm/MCITrack/vis/chotrack_b224_got_a100/Basketball/causal_data',start_frame,window_size)
pass