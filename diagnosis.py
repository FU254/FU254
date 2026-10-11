import os
import sys
sys.path.insert(0, 'classification')

import torch
from models.models_eva import eva02_tiny_patch16_xattn_fusedLN_SwiGLU_preln_RoPE

print('=' * 60)
print('EVA-X 模型检测脚本')
print('=' * 60)

# 检查预训练权重
print('\n[1/3] 检查预训练权重...')
weight_path = '/tmp/pycharm_project_601/EVA-X-main/eva_x_tiny_patch16_merged520k_mim.pt'
if os.path.exists(weight_path):
    print(f'✅ 权重文件存在: {weight_path}')
    print(f'   文件大小: {os.path.getsize(weight_path) / 1024 / 1024:.2f} MB')
    
    try:
        checkpoint = torch.load(weight_path, map_location='cpu', weights_only=False)
        print('✅ 权重文件可以正常读取')
        print(f'   Checkpoint类型: {type(checkpoint).__name__}')
        if isinstance(checkpoint, dict):
            print(f'   Checkpoint键: {list(checkpoint.keys())}')
            
            # 检查是否有model键
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            print(f'   状态字典包含 {len(state_dict)} 个键')
            
            # 尝试加载模型
            model = eva02_tiny_patch16_xattn_fusedLN_SwiGLU_preln_RoPE(pretrained=False)
            msg = model.load_state_dict(state_dict, strict=False)
            print(f'✅ 模型加载成功')
            print(f'   缺失键: {len(msg.missing_keys)}')
            print(f'   意外键: {len(msg.unexpected_keys)}')
            if msg.missing_keys:
                print(f'   缺失键示例: {msg.missing_keys[:3]}')
            if msg.unexpected_keys:
                print(f'   意外键示例: {msg.unexpected_keys[:3]}')
                
    except Exception as e:
        print(f'❌ 权重文件读取失败: {e}')
else:
    print(f'❌ 权重文件不存在: {weight_path}')

# 检查数据路径
print('\n[2/3] 检查数据路径...')
data_paths = [
    '/home/cfu/.virtualenvs/DR/TEST/DR已筛选2019-2023.xlsx',
    '/home/cfu/.virtualenvs/DR/TEST/肺功能万能表.xlsx',
    '/home/cfu/.virtualenvs/DR/TEST/NEW.DR'
]

for path in data_paths:
    if os.path.exists(path):
        if os.path.isdir(path):
            files = os.listdir(path)[:5]
            print(f'✅ 目录存在: {path}')
            print(f'   目录下文件数: {len(os.listdir(path))}')
            if files:
                print(f'   文件示例: {files}')
        else:
            print(f'✅ 文件存在: {path}')
            print(f'   文件大小: {os.path.getsize(path) / 1024 / 1024:.2f} MB')
    else:
        print(f'❌ 路径不存在: {path}')

# 检查Python环境
print('\n[3/3] 检查Python环境...')
print(f'Python版本: {sys.version}')
print(f'PyTorch版本: {torch.__version__}')
print(f'CUDA可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA版本: {torch.version.cuda}')
    print(f'GPU数量: {torch.cuda.device_count()}')
    print(f'GPU名称: {torch.cuda.get_device_name(0)}')

print('\n' + '=' * 60)
print('检测完成!')
print('=' * 60)