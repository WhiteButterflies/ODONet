import yaml
import json

# 加载YAML配置
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# 保存为JSON格式
with open('config.json', 'w') as f:
    json.dump(config, f, indent=4)
