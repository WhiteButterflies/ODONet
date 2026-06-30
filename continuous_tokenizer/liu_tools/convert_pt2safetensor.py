import safetensors.torch
import torch
from safetensors.torch import save_file,load_file

# 加载 PyTorch 模型
model = torch.load('model.pt', map_location='cpu')

#model2 = load_file('model2.safetensors')

# 如果 model 是一个 OrderedDict，直接使用它作为 state_dict
state_dict = model['model']

# 保存为 safetensors 格式
save_file(state_dict, 'model.safetensors')
