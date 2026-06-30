import torch

# 加载 .pth 文件



import torch
def modify_ckpt(pth_path,new_epoch):
    # 加载 .pth 文件
    state_dict = torch.load(pth_path, map_location='cpu',weights_only=False)

    # 进入 'net'
    if 'net' in state_dict:
        net_dict = state_dict['net']
    else:
        raise KeyError("No 'net' key found in state_dict!")

    # 明确列出要删除的 keys
    keys_to_remove = [
        'encoder.body.pos_embed',
        "neck.layers.0.mixer.A_log", "neck.layers.0.mixer.D", "neck.layers.0.mixer.in_proj.weight",
        "neck.layers.0.mixer.conv1d.weight", "neck.layers.0.mixer.conv1d.bias", "neck.layers.0.mixer.x_proj.weight",
        "neck.layers.0.mixer.dt_proj.weight", "neck.layers.0.mixer.dt_proj.bias", "neck.layers.0.mixer.out_proj.weight",
        "neck.layers.0.norm.weight", "neck.layers.1.mixer.A_log", "neck.layers.1.mixer.D",
        "neck.layers.1.mixer.in_proj.weight", "neck.layers.1.mixer.conv1d.weight", "neck.layers.1.mixer.conv1d.bias",
        "neck.layers.1.mixer.x_proj.weight", "neck.layers.1.mixer.dt_proj.weight", "neck.layers.1.mixer.dt_proj.bias",
        "neck.layers.1.mixer.out_proj.weight", "neck.layers.1.norm.weight", "neck.layers.2.mixer.A_log",
        "neck.layers.2.mixer.D", "neck.layers.2.mixer.in_proj.weight", "neck.layers.2.mixer.conv1d.weight",
        "neck.layers.2.mixer.conv1d.bias", "neck.layers.2.mixer.x_proj.weight", "neck.layers.2.mixer.dt_proj.weight",
        "neck.layers.2.mixer.dt_proj.bias", "neck.layers.2.mixer.out_proj.weight", "neck.layers.2.norm.weight",
        "neck.layers.3.mixer.A_log", "neck.layers.3.mixer.D", "neck.layers.3.mixer.in_proj.weight",
        "neck.layers.3.mixer.conv1d.weight", "neck.layers.3.mixer.conv1d.bias", "neck.layers.3.mixer.x_proj.weight",
        "neck.layers.3.mixer.dt_proj.weight", "neck.layers.3.mixer.dt_proj.bias", "neck.layers.3.mixer.out_proj.weight",
        "neck.layers.3.norm.weight"
    ]

    # 删除这些 keys
    for key in keys_to_remove:
        if key in net_dict:
            print(f"Deleting key: {key}")
            del net_dict[key]
        else:
            print(f"Key not found, skipping: {key}")

    # 更新 'net' 回 state_dict
    state_dict['net'] = net_dict

    # 手动设置 epoch
    print(f"Setting epoch to {new_epoch}")
    state_dict['epoch'] = new_epoch

    # 清除优化器状态
    if 'optimizer' in state_dict:
        print("清除优化器状态...")
        del state_dict['optimizer']

    # 清除学习率调度器状态
    if 'scheduler' in state_dict:
        print("清除学习率调度器状态...")
        del state_dict['scheduler']

    # 保存新的文件
    # new_pth_path = 'MCITRACK_ep0240_cleaned_epoch{}.pth.tar'.format(new_epoch)
    new_pth_path = 'CHOTRACK_cleaned_epoch{}.pth.tar'.format(new_epoch)
    torch.save(state_dict, new_pth_path)

    print(f"Saved cleaned model with updated epoch to {new_pth_path}")

if __name__ == '__main__':
    pth_path = r'/data2/lqh/workspace_pycharm/MCITrack/liu_tools/modify_ckpt/ORI_MCITRACK_ep0240.pth.tar'
    # pth_path = r'/data2/lqh/workspace_pycharm/MCITrack/liu_tools/modify_ckpt/CHOTRACK_ep0100.pth.tar'
    # pth_path = r'/data2/lqh/workspace_pycharm/MCITrack/checkpoints/train/chotrack_v9/chotrack_b224/CHOTRACK_ep0240.pth.tar'
    modify_ckpt(pth_path,new_epoch=240)