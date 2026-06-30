import torch
import os
import torch.nn.functional as F # 导入 F 用于余弦相似性
from collections import OrderedDict

def load_ckpt_state_dict(pth_path):
    """
    加载 .ckpt 文件并提取 state_dict。
    考虑到 .ckpt 文件可能直接是 state_dict，也可能是一个包含 'state_dict' 键的字典。
    """
    if not os.path.exists(pth_path):
        raise FileNotFoundError(f"Checkpoint file not found: {pth_path}")

    print(f"Loading checkpoint from: {pth_path}")
    # map_location='cpu' 确保加载到CPU，避免GPU内存问题
    checkpoint = torch.load(pth_path, map_location='cpu', weights_only=False)

    # 尝试从常见的键中提取 state_dict
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            return checkpoint['state_dict']['net']
        elif 'model_state_dict' in checkpoint:
            return checkpoint['model_state_dict']
        elif 'net' in checkpoint:
            return checkpoint['net']
        # 如果checkpoint本身就是state_dict，或者不包含上述键，则直接返回
        # 假设如果它是一个字典，且没有'state_dict'/'model_state_dict'，那它就是state_dict本身
        # 或者是一个包含其他元数据的字典，我们需要过滤出tensor
        # 更好的做法是依赖于模型的结构来load，但这里我们直接处理state_dict
        # 遍历字典，如果是tensor就保留，否则可能需要更具体的处理
        filtered_state_dict = {k: v for k, v in checkpoint.items() if isinstance(v, torch.Tensor)}
        if filtered_state_dict: # 如果过滤后有tensor，就认为是state_dict
             print("Warning: Checkpoint is a dict but no 'state_dict' or 'model_state_dict' key found. Assuming top-level keys are model parameters.")
             return filtered_state_dict
        else:
             raise ValueError(f"Could not find a valid state_dict in {pth_path}. Checkpoint structure: {checkpoint.keys()}")
    elif isinstance(checkpoint, torch.nn.Module):
        return checkpoint.state_dict()
    else:
        # 如果直接是 OrderedDict 或类似结构
        return checkpoint

def get_decoder_weights(state_dict, decoder_prefix='decoder.', strip_prefix=True):
    """
    从完整的 state_dict 中提取 decoder 相关的权重。
    如果 strip_prefix 为 True，则去掉键名中的前缀，方便不同前缀的模型对比。
    """
    decoder_weights = {}
    for k, v in state_dict.items():
        if k.startswith(decoder_prefix):
            if strip_prefix:
                # 截取掉前缀部分，例如 'decoder.conv1.weight' -> 'conv1.weight'
                new_key = k[len(decoder_prefix):]
                decoder_weights[new_key] = v
            else:
                decoder_weights[k] = v
    return decoder_weights

def compare_state_dicts(sd1, sd2, name1="Checkpoint 1", name2="Checkpoint 2", tolerance=1e-6):
    """
    对比两个 state_dict 的差异。
    返回一个字典，记录所有差异。
    """
    global differences_order
    differences = {}
    keys1 = set(sd1.keys())
    keys2 = set(sd2.keys())

    # 1. 检查只存在于一个 state_dict 中的键
    only_in_sd1 = keys1 - keys2
    for k in only_in_sd1:
        differences[k] = f"Only in {name1}"

    only_in_sd2 = keys2 - keys1
    for k in only_in_sd2:
        differences[k] = f"Only in {name2}"

    # 2. 检查共同存在的键
    common_keys = keys1.intersection(keys2)
    for k in common_keys:
        if 'running' in k or 'batch' in k :
            continue
        t1 = sd1[k]
        t2 = sd2[k]

        # 确保它们都是 Tensor
        if not isinstance(t1, torch.Tensor) or not isinstance(t2, torch.Tensor):
            differences[k] = f"Non-tensor item, cannot compare directly: Type1={type(t1)}, Type2={type(t2)}"
            continue

        # 形状对比
        if t1.shape != t2.shape:
            differences[k] = f"Shape mismatch: {t1.shape} in {name1} vs {t2.shape} in {name2}"
        else:
            # 转换类型
            t1_for_comp = t1.to(torch.float32) if not t1.is_floating_point() else t1
            t2_for_comp = t2.to(torch.float32) if not t2.is_floating_point() else t2

            diff_tensor = t1_for_comp - t2_for_comp
            max_abs_diff = diff_tensor.abs().max().item()

            # --- 计算余弦相似性 ---
            cosine_sim = 1.0  # 默认为完全相同
            if t1_for_comp.numel() > 0:
                t1_flat = t1_for_comp.flatten()
                t2_flat = t2_for_comp.flatten()

                # 只有当不是两个全 0 向量时才计算
                if torch.norm(t1_flat, 2) > 1e-8 and torch.norm(t2_flat, 2) > 1e-8:
                    cosine_sim = F.cosine_similarity(t1_flat.unsqueeze(0), t2_flat.unsqueeze(0)).item()
                elif torch.all(t1_flat == 0) and torch.all(t2_flat == 0):
                    cosine_sim = 1.0
                else:
                    cosine_sim = 0.0

            # --- 逻辑修正：无论是否有差异，都记录相似度用于计算平均值 ---
            # 我们可以创建一个单独的 list 来存所有的 similarity
            if 'all_similarities' not in locals():
                all_similarities = []
            all_similarities.append(cosine_sim)

            # 只有超过容忍度时才记录到 differences 字典中用于打印明细
            if max_abs_diff > tolerance:
                differences[k] = {
                    "L1_diff": torch.norm(diff_tensor, 1).item(),
                    "Max_Abs_Diff": max_abs_diff,
                    "Shape": t1.shape,
                    "cos_diff": cosine_sim
                }

            # 计算全局平均相似度（应该使用刚才记录的 all_similarities）
        average_cos_diff = sum(all_similarities) / len(all_similarities) if all_similarities else 1.0
        differences['average'] = {"average_cos_diff": average_cos_diff}

    return differences

def report_differences(differences, comparison_type):
    """
    打印对比结果报告。
    """
    print(f"\n--- {comparison_type} Comparison Report ---")
    if not differences:
        print("No significant differences found.")
        return

    diff_count = 0
    for k, v in differences.items():
        if isinstance(v, dict) and 'average' not in k: # 实际的数值差异
            print(f"  Key '{k}': Values differ (Shape: {v['Shape']})")
            # print(f"    L1 Difference: {v['L1_diff']:.6f}")
            # print(f"    L2 Difference: {v['L2_diff']:.6f}")
            # print(f"    Max Absolute Difference: {v['Max_Abs_Diff']:.6f}")
            # print(f"    Mean Absolute Difference: {v['Mean_Abs_Diff']:.6f}")
            print(f"    Cos Difference: {v['cos_diff']:.6f}")
            diff_count += 1
        else: # 键缺失/类型不匹配等
            print('---------------')
            print(f"  Key '{k}': {v}")
            diff_count += 1
    print(f"\nTotal unique differences found: {diff_count}")

if __name__ == '__main__':
    # 定义你的 checkpoint 路径
    # 假设你的文件路径是这样的，请根据实际情况修改
    ckpt1_path = r'/data2/lqh/workspace_pycharm/MCITrack/checkpoints/train/chotrack_v7/chotrack_b224_got_double/CHOTRACK_ep0005.pth.tar'
    ckpt2_path = r'/data2/lqh/workspace_pycharm/MCITrack/checkpoints/train/chotrack_v7/chotrack_b224_got_double/CHOTRACK_ep0100.pth.tar'
    # 或者另一个路径，例如：
    # ckpt2_path = r'/data2/lqh/workspace_pycharm/MCITrack/checkpoints/train/chotrack_v6/chotrack_b224_got_double/CHOTRACK_ep0100.pth.tar'

    # 定义一个容忍度，用于浮点数比较。如果两个数值的绝对差小于此值，则认为它们相同。
    COMPARISON_TOLERANCE = 1e-6

    try:
        # 1. 加载完整的 state_dict
        sd1_full = load_ckpt_state_dict(ckpt1_path)
        sd2_full = load_ckpt_state_dict(ckpt2_path)

        print(f"\nCheckpoint 1 (Full) has {len(sd1_full)} keys.")
        print(f"Checkpoint 2 (Full) has {len(sd2_full)} keys.")

        # 2. 对比所有权重
        # print("\n" + "="*60)
        # print("Comparing ALL weights between Checkpoint 1 and Checkpoint 2")
        # print("="*60)
        # all_weights_diffs = compare_state_dicts(sd1_full, sd2_full,
        #                                         name1="Checkpoint 1", name2="Checkpoint 2",
        #                                         tolerance=COMPARISON_TOLERANCE)
        # report_differences(all_weights_diffs, "ALL Weights")

        # 3. 提取并对比 DECODER 权重
        print("\n" + "="*60)
        print("Extracting and Comparing DECODER weights")
        print("="*60)
        # sd1_decoder = get_decoder_weights(sd1_full, decoder_prefix='decoder.') # 根据你的模型实际的decoder键名修改前缀
        sd1_decoder = get_decoder_weights(sd1_full, decoder_prefix='other_model_dict.decoder.') # 根据你的模型实际的decoder键名修改前缀
        sd2_decoder = get_decoder_weights(sd2_full, decoder_prefix='other_model_dict.decoder.')
        # sd2_decoder = get_decoder_weights(sd2_full, decoder_prefix='decoder.')

        print(f"\nCheckpoint 1 (Decoder) has {len(sd1_decoder)} keys.")
        print(f"Checkpoint 2 (Decoder) has {len(sd2_decoder)} keys.")

        decoder_weights_diffs = compare_state_dicts(sd1_decoder, sd2_decoder,
                                                    name1="Checkpoint 1 Decoder", name2="Checkpoint 2 Decoder",
                                                    tolerance=COMPARISON_TOLERANCE)
        report_differences(decoder_weights_diffs, "DECODER Weights")

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

