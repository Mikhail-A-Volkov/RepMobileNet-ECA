"""
模型逐层维度与参数量分析工具
使用forward hooks实现，无需第三方库
"""
import torch
import torch.nn as nn
import csv
import os
from collections import OrderedDict
from datetime import datetime


class ModelProfiler:
    """
    模型分析器，使用forward hooks记录每层的输入输出形状和参数量
    """
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.hooks = []
        self.layer_info = OrderedDict()
        
    def _register_hooks(self):
        """注册forward hooks到所有叶子模块"""
        def make_hook(name):
            def hook(module, input, output):
                # 只记录叶子模块（没有子模块的模块）
                if len(list(module.children())) == 0:
                    # 获取输入形状
                    if isinstance(input, (tuple, list)) and len(input) > 0:
                        input_shapes = []
                        for inp in input:
                            if isinstance(inp, torch.Tensor):
                                input_shapes.append(list(inp.shape))
                            elif inp is not None:
                                input_shapes.append(str(type(inp)))
                        if not input_shapes:
                            input_shapes = ["N/A"]
                    elif isinstance(input, torch.Tensor):
                        input_shapes = [list(input.shape)]
                    else:
                        input_shapes = ["N/A"]
                    
                    # 获取输出形状
                    if isinstance(output, (tuple, list)) and len(output) > 0:
                        output_shapes = []
                        for out in output:
                            if isinstance(out, torch.Tensor):
                                output_shapes.append(list(out.shape))
                            elif out is not None:
                                output_shapes.append(str(type(out)))
                        if not output_shapes:
                            output_shapes = ["N/A"]
                    elif isinstance(output, torch.Tensor):
                        output_shapes = [list(output.shape)]
                    else:
                        output_shapes = ["N/A"]
                    
                    # 计算参数量
                    total_params = sum(p.numel() for p in module.parameters())
                    trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
                    
                    # 获取模块类型
                    module_type = type(module).__name__
                    
                    # 格式化形状字符串
                    def format_shape(shapes):
                        if not shapes:
                            return "N/A"
                        if len(shapes) == 1:
                            shape = shapes[0]
                            if isinstance(shape, list):
                                return "×".join(map(str, shape))
                            return str(shape)
                        return " | ".join([format_shape([s]) for s in shapes])
                    
                    self.layer_info[name] = {
                        'layer_name': name,
                        'layer_type': module_type,
                        'input_shape': format_shape(input_shapes),
                        'output_shape': format_shape(output_shapes),
                        'num_params': total_params,
                        'trainable_params': trainable_params
                    }
            return hook
        
        # 为所有模块注册hooks
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # 只注册叶子模块
                hook = module.register_forward_hook(make_hook(name))
                self.hooks.append(hook)
    
    def _remove_hooks(self):
        """移除所有hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
    
    def profile(self, sample_input):
        """
        分析模型
        
        Args:
            sample_input: 样本输入tensor，形状为 [B, C, H, W]
        
        Returns:
            layer_info: OrderedDict，包含每层的信息
        """
        # 清空之前的信息
        self.layer_info.clear()
        
        # 注册hooks
        self._register_hooks()
        
        # 保存原始状态
        was_training = self.model.training
        
        try:
            # 设置为eval模式并禁用梯度
            self.model.eval()
            with torch.no_grad():
                # 确保输入在正确的device上
                if isinstance(sample_input, torch.Tensor):
                    sample_input = sample_input.to(self.device)
                else:
                    sample_input = [inp.to(self.device) if isinstance(inp, torch.Tensor) else inp 
                                  for inp in sample_input]
                
                # 前向传播
                _ = self.model(sample_input)
        finally:
            # 恢复原始状态
            if was_training:
                self.model.train()
            else:
                self.model.eval()
            
            # 移除hooks
            self._remove_hooks()
        
        return self.layer_info


def profile_model(model, sample_input, backbone_name, output_dir='output/logs', device='cuda'):
    """
    分析模型并保存结果到CSV和Markdown文件
    
    Args:
        model: 要分析的模型
        sample_input: 样本输入tensor
        backbone_name: backbone名称（用于文件名）
        output_dir: 输出目录
        device: 设备（'cuda' 或 'cpu'）
    
    Returns:
        layer_info: 包含每层信息的OrderedDict
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建profiler
    profiler = ModelProfiler(model, device=device)
    
    # 执行分析
    layer_info = profiler.profile(sample_input)
    
    # 生成时间戳
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 保存CSV文件
    csv_path = os.path.join(output_dir, f'model_profile_{backbone_name}_{timestamp}.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # 写入表头
        writer.writerow(['Layer Name', 'Layer Type', 'Input Shape', 'Output Shape', 
                         'Total Params', 'Trainable Params'])
        
        # 写入数据
        for info in layer_info.values():
            writer.writerow([
                info['layer_name'],
                info['layer_type'],
                info['input_shape'],
                info['output_shape'],
                info['num_params'],
                info['trainable_params']
            ])
    
    print(f'Model profile saved to CSV: {csv_path}')
    
    # 保存Markdown文件
    md_path = os.path.join(output_dir, f'model_profile_{backbone_name}_{timestamp}.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        # 写入标题
        f.write(f'# Model Profile: {backbone_name}\n\n')
        f.write(f'Generated at: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n\n')
        
        # 写入表格
        f.write('| Layer Name | Layer Type | Input Shape | Output Shape | Total Params | Trainable Params |\n')
        f.write('|------------|------------|-------------|--------------|--------------|------------------|\n')
        
        for info in layer_info.values():
            f.write(f"| {info['layer_name']} | {info['layer_type']} | "
                   f"{info['input_shape']} | {info['output_shape']} | "
                   f"{info['num_params']:,} | {info['trainable_params']:,} |\n")
        
        # 写入统计信息
        total_params = sum(info['num_params'] for info in layer_info.values())
        total_trainable = sum(info['trainable_params'] for info in layer_info.values())
        
        f.write('\n## Summary\n\n')
        f.write(f'- **Total Layers**: {len(layer_info)}\n')
        f.write(f'- **Total Parameters**: {total_params:,}\n')
        f.write(f'- **Trainable Parameters**: {total_trainable:,}\n')
        f.write(f'- **Non-trainable Parameters**: {total_params - total_trainable:,}\n')
        f.write(f'- **Model Size (MB)**: {total_params * 4 / (1024 * 1024):.2f} (assuming float32)\n')
    
    print(f'Model profile saved to Markdown: {md_path}')
    
    # 打印摘要信息
    total_params = sum(info['num_params'] for info in layer_info.values())
    total_trainable = sum(info['trainable_params'] for info in layer_info.values())
    print(f'\nModel Profile Summary:')
    print(f'  Total Layers: {len(layer_info)}')
    print(f'  Total Parameters: {total_params:,}')
    print(f'  Trainable Parameters: {total_trainable:,}')
    print(f'  Model Size: {total_params * 4 / (1024 * 1024):.2f} MB\n')
    
    return layer_info

