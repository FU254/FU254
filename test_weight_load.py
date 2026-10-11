import sys
sys.path.insert(0, 'classification')

import torch
from models.models_eva import eva02_tiny_patch16_xattn_fusedLN_SwiGLU_preln_RoPE

print('创建模型...')
model = eva02_tiny_patch16_xattn_fusedLN_SwiGLU_preln_RoPE(pretrained=False)
print('模型创建成功')

weight_path = r'C:\Users\Administrator\Downloads\eva_x_tiny_patch16_merged520k_mim.pt'
print(f'加载权重: {weight_path}')

try:
    checkpoint = torch.load(weight_path, map_location='cpu', weights_only=False)
    print('权重文件加载成功')
    print('Checkpoint类型:', type(checkpoint))
    print('Checkpoint键:', list(checkpoint.keys())[:10])
    
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'module' in checkpoint:
        state_dict = checkpoint['module']
    else:
        state_dict = checkpoint
        
    print('状态字典键数量:', len(state_dict))
    
    msg = model.load_state_dict(state_dict, strict=False)
    print('缺失键数量:', len(msg.missing_keys))
    print('意外键数量:', len(msg.unexpected_keys))
    
    if msg.missing_keys:
        print('前5个缺失键:', msg.missing_keys[:5])
    if msg.unexpected_keys:
        print('前5个意外键:', msg.unexpected_keys[:5])
        
    print('测试完成!')
    
except Exception as e:
    print(f'错误: {e}')
    import traceback
    traceback.print_exc()