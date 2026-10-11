"""
测试EVA-X-small-patch16模型和权重加载
"""

import sys
import torch
import torch.nn as nn

# 添加项目路径
sys.path.insert(0, r'/tmp/pycharm_project_601/EVA-X-main/classification')
from models.models_eva import eva02_small_patch16_xattn_fusedLN_SwiGLU_preln_RoPE


def log(message):
    print(f"[INFO] {message}")


def test_model_creation():
    """测试模型创建"""
    log("创建EVA-X-small-patch16模型...")
    
    model = eva02_small_patch16_xattn_fusedLN_SwiGLU_preln_RoPE(pretrained=False)
    
    log(f"模型创建成功")
    log(f"模型配置:")
    log(f"  - embed_dim: {model.embed_dim}")
    log(f"  - depth: {len(model.blocks)}")
    log(f"  - num_heads: {model.blocks[0].attn.num_heads}")
    
    # 打印模型结构概览
    total_params = sum(p.numel() for p in model.parameters())
    log(f"总参数量: {total_params / 1e6:.2f}M")
    
    # 检查模型结构
    log("\n检查第一个block结构:")
    block = model.blocks[0]
    log(f"  - 注意力机制: {type(block.attn).__name__}")
    log(f"  - MLP机制: {type(block.mlp).__name__}")
    
    if hasattr(block.attn, 'q_proj'):
        log(f"  - 使用分离的QKV投影: q_proj, k_proj, v_proj")
    else:
        log(f"  - 使用融合的QKV投影: qkv")
    
    return model


def test_forward_pass(model):
    """测试前向传播"""
    log("\n测试前向传播...")
    
    # 检查是否有可用的 GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log(f"使用设备: {device}")
    
    # 将模型移动到设备
    model = model.to(device)
    
    # 创建模拟输入并移动到设备
    x = torch.randn(2, 3, 224, 224).to(device)
    log(f"输入形状: {x.shape}")
    
    with torch.no_grad():
        output = model(x)
    
    log(f"输出形状: {output.shape}")
    log("前向传播测试通过!")


def inspect_checkpoint(checkpoint_path):
    """检查检查点内容"""
    if checkpoint_path is None or not checkpoint_path:
        log("\n未提供检查点路径，跳过检查点检查")
        return None
    
    if not os.path.exists(checkpoint_path):
        log(f"\n检查点不存在: {checkpoint_path}")
        return None
    
    log(f"\n加载检查点: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    log("检查点顶层键:")
    for key in checkpoint.keys():
        log(f"  - {key}")
    
    # 找到state_dict
    state_dict = None
    if 'module' in checkpoint:
        state_dict = checkpoint['module']
        log("使用 'module' 中的state_dict")
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
        log("使用 'model' 中的state_dict")
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        log("使用 'state_dict' 中的state_dict")
    else:
        # 直接使用checkpoint
        state_dict = checkpoint
        log("直接使用checkpoint作为state_dict")
    
    # 过滤掉非tensor
    state_dict = {k: v for k, v in state_dict.items() if isinstance(v, torch.Tensor)}
    log(f"\nstate_dict包含 {len(state_dict)} 个键")
    
    log("\n前10个键:")
    for i, key in enumerate(list(state_dict.keys())[:10]):
        log(f"  [{i}] {key}")
    
    return state_dict


def test_weight_loading(model, state_dict):
    """测试权重加载"""
    if state_dict is None:
        return
    
    log("\n测试权重加载...")
    
    # 移除可能的前缀
    cleaned_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            cleaned_dict[k[7:]] = v
        else:
            cleaned_dict[k] = v
    
    # 尝试加载，忽略lm_head等
    msg = model.load_state_dict(cleaned_dict, strict=False)
    
    log(f"权重加载完成")
    if msg.missing_keys:
        log(f"缺失的键 ({len(msg.missing_keys)}):")
        for k in msg.missing_keys[:10]:
            log(f"  - {k}")
        if len(msg.missing_keys) > 10:
            log(f"  ... 还有 {len(msg.missing_keys) - 10} 个")
    
    if msg.unexpected_keys:
        log(f"意外的键 ({len(msg.unexpected_keys)}):")
        for k in msg.unexpected_keys[:10]:
            log(f"  - {k}")
        if len(msg.unexpected_keys) > 10:
            log(f"  ... 还有 {len(msg.unexpected_keys) - 10} 个")


def main():
    import os
    
    log("=" * 60)
    log("EVA-X-small-patch16 模型测试")
    log("=" * 60)
    
    # 1. 创建模型
    model = test_model_creation()
    
    # 2. 测试前向传播
    test_forward_pass(model)
    
    # 3. 检查检查点（如果提供了路径）
    # 请在这里设置您的检查点路径
    checkpoint_path = None  # 例如: r'D:\path\to\your\checkpoint.pt'
    state_dict = inspect_checkpoint(checkpoint_path)
    
    # 4. 测试权重加载
    if state_dict is not None:
        test_weight_loading(model, state_dict)
    
    log("\n" + "=" * 60)
    log("测试完成!")
    log("=" * 60)


if __name__ == "__main__":
    main()

