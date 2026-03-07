"""
模型参数量对比和可视化脚本
比较RepVGG和MobileNetV2作为backbone的参数量
"""
import torch
import torch.nn as nn
import sys
import os
import json
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from sixdrepnet.model import SixDRepNet, SixDRepNet_MobileNetV2


def count_parameters(model):
    """统计模型参数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # 计算模型大小（MB，假设float32）
    model_size_mb = total_params * 4 / (1024 * 1024)
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'model_size_mb': model_size_mb
    }


def get_layer_parameters(model):
    """获取各层参数量"""
    layer_params = {}
    for name, param in model.named_parameters():
        num_params = param.numel()
        layer_params[name] = {
            'params': num_params,
            'shape': list(param.shape)
        }
    return layer_params


def print_model_info(model, model_name, save_json=False):
    """打印模型信息"""
    print(f"\n{'='*70}")
    print(f"{model_name:^70}")
    print(f"{'='*70}")
    
    info = count_parameters(model)
    print(f"\n总参数量: {info['total_params']:,}")
    print(f"可训练参数量: {info['trainable_params']:,}")
    print(f"模型大小: {info['model_size_mb']:.2f} MB")
    
    # 获取各层参数量
    layer_params = get_layer_parameters(model)
    total = info['total_params']
    
    # 打印各层参数量（前15层）
    print(f"\n各层参数量详情（前15层）:")
    print(f"{'层名称':<50} {'参数量':<15} {'占比':<10}")
    print("-" * 75)
    
    sorted_layers = sorted(layer_params.items(), key=lambda x: x[1]['params'], reverse=True)
    for name, layer_info in sorted_layers[:15]:
        num_params = layer_info['params']
        percentage = (num_params / total) * 100
        print(f"{name:<50} {num_params:>15,} {percentage:>9.2f}%")
    
    # 按模块统计参数量
    print(f"\n按模块统计参数量:")
    print(f"{'模块':<30} {'参数量':<15} {'占比':<10}")
    print("-" * 55)
    
    module_params = {}
    for name, layer_info in layer_params.items():
        module_name = name.split('.')[0]
        if module_name not in module_params:
            module_params[module_name] = 0
        module_params[module_name] += layer_info['params']
    
    for module_name, num_params in sorted(module_params.items(), key=lambda x: x[1], reverse=True):
        percentage = (num_params / total) * 100
        print(f"{module_name:<30} {num_params:>15,} {percentage:>9.2f}%")
    
    # 保存JSON文件
    if save_json:
        output_data = {
            'model_name': model_name,
            'total_params': info['total_params'],
            'trainable_params': info['trainable_params'],
            'model_size_mb': info['model_size_mb'],
            'layer_params': layer_params,
            'module_params': module_params
        }
        json_file = f'output/model_info_{model_name.replace(" ", "_")}.json'
        os.makedirs('output', exist_ok=True)
        with open(json_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\n模型信息已保存到: {json_file}")
    
    return info, layer_params


def visualize_comparison(info_repvgg, info_mnv2, save_path='output/model_comparison.png'):
    """可视化模型对比"""
    os.makedirs('output', exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. 参数量对比柱状图
    models = ['RepVGG-B1g2', 'MobileNetV2']
    params = [info_repvgg['total_params'], info_mnv2['total_params']]
    colors = ['#3498db', '#e74c3c']
    
    axes[0].bar(models, params, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    axes[0].set_ylabel('参数量', fontsize=12)
    axes[0].set_title('总参数量对比', fontsize=14, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3, linestyle='--')
    
    # 添加数值标签
    for i, v in enumerate(params):
        axes[0].text(i, v, f'{v/1e6:.2f}M', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 2. 模型大小对比
    sizes = [info_repvgg['model_size_mb'], info_mnv2['model_size_mb']]
    axes[1].bar(models, sizes, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    axes[1].set_ylabel('模型大小 (MB)', fontsize=12)
    axes[1].set_title('模型大小对比', fontsize=14, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3, linestyle='--')
    
    # 添加数值标签
    for i, v in enumerate(sizes):
        axes[1].text(i, v, f'{v:.2f}MB', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 3. 参数量减少百分比
    reduction = (info_repvgg['total_params'] - info_mnv2['total_params']) / info_repvgg['total_params'] * 100
    axes[2].bar(['参数量减少'], [reduction], color='#2ecc71', alpha=0.7, edgecolor='black', linewidth=1.5)
    axes[2].set_ylabel('减少百分比 (%)', fontsize=12)
    axes[2].set_title('参数量减少', fontsize=14, fontweight='bold')
    axes[2].grid(axis='y', alpha=0.3, linestyle='--')
    axes[2].text(0, reduction, f'{reduction:.2f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n对比图已保存到: {save_path}")
    plt.close()


def visualize_layer_comparison(layer_params_repvgg, layer_params_mnv2, save_path='output/layer_comparison.png'):
    """可视化各层参数量对比"""
    os.makedirs('output', exist_ok=True)
    
    # 获取backbone部分的参数量
    repvgg_backbone_params = {}
    mnv2_backbone_params = {}
    
    for name, info in layer_params_repvgg.items():
        if 'linear_reg' not in name and 'gap' not in name:
            module_name = '.'.join(name.split('.')[:2])  # 获取前两级模块名
            if module_name not in repvgg_backbone_params:
                repvgg_backbone_params[module_name] = 0
            repvgg_backbone_params[module_name] += info['params']
    
    for name, info in layer_params_mnv2.items():
        if 'linear_reg' not in name and 'gap' not in name:
            module_name = '.'.join(name.split('.')[:2])
            if module_name not in mnv2_backbone_params:
                mnv2_backbone_params[module_name] = 0
            mnv2_backbone_params[module_name] += info['params']
    
    # 选择参数量最多的前10个模块
    repvgg_top = sorted(repvgg_backbone_params.items(), key=lambda x: x[1], reverse=True)[:10]
    mnv2_top = sorted(mnv2_backbone_params.items(), key=lambda x: x[1], reverse=True)[:10]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # RepVGG各层参数量
    modules_repvgg = [m[0] for m in repvgg_top]
    params_repvgg = [m[1] for m in repvgg_top]
    
    ax1.barh(range(len(modules_repvgg)), params_repvgg, color='#3498db', alpha=0.7, edgecolor='black')
    ax1.set_yticks(range(len(modules_repvgg)))
    ax1.set_yticklabels(modules_repvgg, fontsize=9)
    ax1.set_xlabel('参数量', fontsize=12)
    ax1.set_title('RepVGG-B1g2 主要模块参数量', fontsize=14, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3, linestyle='--')
    ax1.invert_yaxis()
    
    # MobileNetV2各层参数量
    modules_mnv2 = [m[0] for m in mnv2_top]
    params_mnv2 = [m[1] for m in mnv2_top]
    
    ax2.barh(range(len(modules_mnv2)), params_mnv2, color='#e74c3c', alpha=0.7, edgecolor='black')
    ax2.set_yticks(range(len(modules_mnv2)))
    ax2.set_yticklabels(modules_mnv2, fontsize=9)
    ax2.set_xlabel('参数量', fontsize=12)
    ax2.set_title('MobileNetV2 主要模块参数量', fontsize=14, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3, linestyle='--')
    ax2.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"各层对比图已保存到: {save_path}")
    plt.close()


def compare_models():
    """对比两个模型的参数量"""
    print("="*70)
    print("模型参数量对比分析".center(70))
    print("="*70)
    
    # 创建RepVGG模型
    print("\n正在创建RepVGG模型...")
    try:
        model_repvgg = SixDRepNet(
            backbone_name='RepVGG-B1g2',
            backbone_file='',  # 不加载预训练权重，只统计结构
            deploy=False,
            pretrained=False
        )
        info_repvgg, layer_params_repvgg = print_model_info(
            model_repvgg, "SixDRepNet (RepVGG-B1g2)", save_json=True)
    except Exception as e:
        print(f"创建RepVGG模型失败: {e}")
        print("提示：可能需要RepVGG权重文件，但参数量统计不需要权重")
        return
    
    # 创建MobileNetV2模型
    print("\n正在创建MobileNetV2模型...")
    model_mnv2 = SixDRepNet_MobileNetV2(pretrained=False)
    info_mnv2, layer_params_mnv2 = print_model_info(
        model_mnv2, "SixDRepNet (MobileNetV2)", save_json=True)
    
    # 对比结果
    print(f"\n{'='*70}")
    print("对比结果总结".center(70))
    print(f"{'='*70}")
    
    reduction_params = info_repvgg['total_params'] - info_mnv2['total_params']
    reduction_percentage = (reduction_params / info_repvgg['total_params']) * 100
    size_reduction = info_repvgg['model_size_mb'] - info_mnv2['model_size_mb']
    size_reduction_percentage = (size_reduction / info_repvgg['model_size_mb']) * 100
    
    print(f"\n参数量:")
    print(f"  RepVGG-B1g2:     {info_repvgg['total_params']:>15,} ({info_repvgg['total_params']/1e6:.2f}M)")
    print(f"  MobileNetV2:      {info_mnv2['total_params']:>15,} ({info_mnv2['total_params']/1e6:.2f}M)")
    print(f"  减少:             {reduction_params:>15,} ({reduction_percentage:.2f}%)")
    
    print(f"\n模型大小:")
    print(f"  RepVGG-B1g2:     {info_repvgg['model_size_mb']:>15.2f} MB")
    print(f"  MobileNetV2:      {info_mnv2['model_size_mb']:>15.2f} MB")
    print(f"  减少:             {size_reduction:>15.2f} MB ({size_reduction_percentage:.2f}%)")
    
    print(f"\nMobileNetV2参数量是RepVGG的: {info_mnv2['total_params']/info_repvgg['total_params']*100:.2f}%")
    print(f"MobileNetV2模型大小是RepVGG的: {info_mnv2['model_size_mb']/info_repvgg['model_size_mb']*100:.2f}%")
    
    # 可视化
    print(f"\n{'='*70}")
    print("生成可视化图表...".center(70))
    print(f"{'='*70}")
    
    visualize_comparison(info_repvgg, info_mnv2)
    visualize_layer_comparison(layer_params_repvgg, layer_params_mnv2)
    
    print(f"\n{'='*70}")
    print("分析完成！".center(70))
    print(f"{'='*70}")


if __name__ == '__main__':
    compare_models()









