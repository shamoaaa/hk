"""
精简 FunASR SeacoParaformer 训练 checkpoint 用于推理部署。

移除优化器状态、调度器状态等训练相关数据，只保留模型权重，
大幅减小文件体积，加快加载速度。
"""

import torch
import os
import sys


def slim_checkpoint(input_path: str, output_path: str = None):
    """
    精简 checkpoint，只保留推理所需的模型权重。
    
    Args:
        input_path: 原始 checkpoint 路径
        output_path: 输出路径，默认在同目录生成 _slim 后缀文件
    """
    if not os.path.exists(input_path):
        print(f"[错误] 文件不存在: {input_path}")
        sys.exit(1)

    if output_path is None:
        base, ext = os.path.splitext(input_path)
        if ext:
            output_path = f"{base}_slim{ext}"
        else:
            output_path = f"{input_path}_slim"

    print(f"[1/4] 加载 checkpoint: {input_path}")
    file_size_mb = os.path.getsize(input_path) / (1024 * 1024)
    print(f"       原始文件大小: {file_size_mb:.1f} MB")

    checkpoint = torch.load(input_path, map_location="cpu")

    # 分析 checkpoint 结构
    print(f"\n[2/4] 分析 checkpoint 结构:")
    if isinstance(checkpoint, dict):
        for key in sorted(checkpoint.keys()):
            value = checkpoint[key]
            if isinstance(value, dict):
                print(f"       - {key}: dict ({len(value)} items)")
            elif isinstance(value, torch.Tensor):
                print(f"       - {key}: tensor {value.shape}")
            else:
                print(f"       - {key}: {type(value).__name__}")
    else:
        print(f"       checkpoint 类型: {type(checkpoint).__name__}")

    # 提取模型权重
    print(f"\n[3/4] 提取模型权重，移除训练状态...")
    
    if isinstance(checkpoint, dict):
        # FunASR checkpoint 通常的结构
        # 可能的 key: model_state_dict, state_dict, model, optimizer, scheduler, epoch 等
        model_state = None
        
        # 尝试不同的 key 名称
        for key in ["model_state_dict", "state_dict", "model"]:
            if key in checkpoint:
                model_state = checkpoint[key]
                print(f"       找到模型权重 key: '{key}'")
                break
        
        if model_state is not None:
            # 只保留模型权重
            slim_data = model_state
            
            # 列出被移除的 key
            removed_keys = [k for k in checkpoint.keys() if k not in [key]]
            if removed_keys:
                print(f"       移除的训练状态: {removed_keys}")
        else:
            # 如果没有找到标准 key，检查是否整个 dict 就是 state_dict
            # (所有 value 都是 tensor 的情况)
            all_tensors = all(isinstance(v, torch.Tensor) for v in checkpoint.values())
            if all_tensors:
                print("       checkpoint 本身就是 state_dict 格式")
                slim_data = checkpoint
            else:
                # 尝试移除已知的训练相关 key
                training_keys = [
                    "optimizer", "optimizer_state_dict",
                    "scheduler", "scheduler_state_dict", 
                    "lr_scheduler", "lr_scheduler_state_dict",
                    "epoch", "step", "global_step",
                    "grad_scaler", "scaler",
                    "train_conf", "optim_conf", "scheduler_conf",
                    "sampler_state", "rng_state",
                ]
                slim_data = {}
                removed = []
                for k, v in checkpoint.items():
                    if k in training_keys:
                        removed.append(k)
                    else:
                        slim_data[k] = v
                if removed:
                    print(f"       移除的训练状态 key: {removed}")
                else:
                    print("       未识别到标准训练状态 key，保留所有内容")
                    slim_data = checkpoint
    else:
        # 不是 dict，直接保存
        slim_data = checkpoint

    # 保存精简后的模型
    print(f"\n[4/4] 保存精简模型: {output_path}")
    torch.save(slim_data, output_path)
    
    output_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    reduction = (1 - output_size_mb / file_size_mb) * 100
    
    print(f"\n{'='*50}")
    print(f"  精简完成!")
    print(f"  原始大小: {file_size_mb:.1f} MB")
    print(f"  精简后:   {output_size_mb:.1f} MB")
    print(f"  减少:     {reduction:.1f}%")
    print(f"  输出文件: {output_path}")
    print(f"{'='*50}")
    
    return output_path


if __name__ == "__main__":
    input_file = r"E:\8.音频\03 s10\weight\model.pt.ep35"
    output_file = r"E:\8.音频\03 s10\weight\model.pt"
    
    slim_checkpoint(input_file, output_file)
