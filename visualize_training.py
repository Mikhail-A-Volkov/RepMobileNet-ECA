"""
训练历史可视化脚本
从training_history.json文件读取训练历史并生成可视化图表
"""
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse


def load_training_history(json_path):
    """加载训练历史"""
    with open(json_path, 'r') as f:
        history = json.load(f)
    return history


def plot_training_curves(history, save_dir='output'):
    """绘制训练曲线"""
    os.makedirs(save_dir, exist_ok=True)
    
    epochs = history['epoch']
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. 训练和验证损失
    ax1 = axes[0, 0]
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2, marker='o', markersize=4)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2, marker='s', markersize=4)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # 2. 验证MAE
    ax2 = axes[0, 1]
    ax2.plot(epochs, history['val_mae'], 'g-', label='Val MAE', linewidth=2, marker='^', markersize=4)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('MAE (degrees)', fontsize=12)
    ax2.set_title('Validation Mean Absolute Error', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    # 3. 各角度误差
    ax3 = axes[1, 0]
    ax3.plot(epochs, history['val_yaw_error'], 'b-', label='Yaw Error', linewidth=2, marker='o', markersize=4)
    ax3.plot(epochs, history['val_pitch_error'], 'r-', label='Pitch Error', linewidth=2, marker='s', markersize=4)
    ax3.plot(epochs, history['val_roll_error'], 'g-', label='Roll Error', linewidth=2, marker='^', markersize=4)
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Error (degrees)', fontsize=12)
    ax3.set_title('Validation Angle Errors', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3, linestyle='--')
    
    # 4. 学习率
    ax4 = axes[1, 1]
    ax4.plot(epochs, history['learning_rate'], 'm-', label='Learning Rate', linewidth=2, marker='d', markersize=4)
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('Learning Rate', fontsize=12)
    ax4.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3, linestyle='--')
    ax4.set_yscale('log')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'training_curves.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"训练曲线已保存到: {save_path}")
    plt.close()


def plot_error_comparison(history, save_dir='output'):
    """绘制角度误差对比图"""
    os.makedirs(save_dir, exist_ok=True)
    
    epochs = history['epoch']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(epochs))
    width = 0.25
    
    ax.bar(x - width, history['val_yaw_error'], width, label='Yaw', alpha=0.8, color='#3498db')
    ax.bar(x, history['val_pitch_error'], width, label='Pitch', alpha=0.8, color='#e74c3c')
    ax.bar(x + width, history['val_roll_error'], width, label='Roll', alpha=0.8, color='#2ecc71')
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Error (degrees)', fontsize=12)
    ax.set_title('Validation Angle Errors by Epoch', fontsize=14, fontweight='bold')
    ax.set_xticks(x[::max(1, len(epochs)//10)])  # 只显示部分epoch标签
    ax.set_xticklabels([epochs[i] for i in range(0, len(epochs), max(1, len(epochs)//10))])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'error_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"误差对比图已保存到: {save_path}")
    plt.close()


def print_summary(history):
    """打印训练总结"""
    print("\n" + "="*70)
    print("训练总结".center(70))
    print("="*70)
    
    best_val_loss_idx = np.argmin(history['val_loss'])
    best_val_mae_idx = np.argmin(history['val_mae'])
    
    print(f"\n最佳验证损失:")
    print(f"  Epoch: {history['epoch'][best_val_loss_idx]}")
    print(f"  Val Loss: {history['val_loss'][best_val_loss_idx]:.6f}")
    print(f"  Val MAE: {history['val_mae'][best_val_loss_idx]:.4f}")
    print(f"  Yaw Error: {history['val_yaw_error'][best_val_loss_idx]:.4f}")
    print(f"  Pitch Error: {history['val_pitch_error'][best_val_loss_idx]:.4f}")
    print(f"  Roll Error: {history['val_roll_error'][best_val_loss_idx]:.4f}")
    
    print(f"\n最佳验证MAE:")
    print(f"  Epoch: {history['epoch'][best_val_mae_idx]}")
    print(f"  Val Loss: {history['val_loss'][best_val_mae_idx]:.6f}")
    print(f"  Val MAE: {history['val_mae'][best_val_mae_idx]:.4f}")
    print(f"  Yaw Error: {history['val_yaw_error'][best_val_mae_idx]:.4f}")
    print(f"  Pitch Error: {history['val_pitch_error'][best_val_mae_idx]:.4f}")
    print(f"  Roll Error: {history['val_roll_error'][best_val_mae_idx]:.4f}")
    
    print(f"\n最终结果 (Epoch {history['epoch'][-1]}):")
    print(f"  Train Loss: {history['train_loss'][-1]:.6f}")
    print(f"  Val Loss: {history['val_loss'][-1]:.6f}")
    print(f"  Val MAE: {history['val_mae'][-1]:.4f}")
    print(f"  Learning Rate: {history['learning_rate'][-1]:.6f}")


def main():
    parser = argparse.ArgumentParser(description='Visualize training history')
    parser.add_argument('--history', dest='history_path',
                       help='Path to training_history.json file',
                       default='output/snapshots/training_history.json', type=str)
    parser.add_argument('--output_dir', dest='output_dir',
                       help='Output directory for plots',
                       default='output', type=str)
    
    args = parser.parse_args()
    
    if not os.path.exists(args.history_path):
        print(f"错误: 找不到训练历史文件: {args.history_path}")
        return
    
    print(f"加载训练历史: {args.history_path}")
    history = load_training_history(args.history_path)
    
    print_summary(history)
    
    print("\n生成可视化图表...")
    plot_training_curves(history, args.output_dir)
    plot_error_comparison(history, args.output_dir)
    
    print("\n可视化完成！")


if __name__ == '__main__':
    main()





