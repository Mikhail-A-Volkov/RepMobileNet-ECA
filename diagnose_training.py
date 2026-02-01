"""
训练诊断脚本
用于分析训练过程中的问题，特别是角度回归问题
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse


def analyze_training_history(history_path):
    """分析训练历史，诊断潜在问题"""
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    epochs = history['epoch']
    train_loss = history['train_loss']
    val_loss = history['val_loss']
    val_yaw = history['val_yaw_error']
    val_pitch = history['val_pitch_error']
    val_roll = history['val_roll_error']
    val_mae = history['val_mae']
    lr = history['learning_rate']
    
    print("="*70)
    print("训练诊断分析".center(70))
    print("="*70)
    
    # 1. 损失分析
    print("\n1. 损失分析:")
    print(f"   最终训练损失: {train_loss[-1]:.6f}")
    print(f"   最终验证损失: {val_loss[-1]:.6f}")
    print(f"   损失下降趋势: {'正常' if train_loss[-1] < train_loss[0] * 0.5 else '异常（下降不明显）'}")
    print(f"   过拟合程度: {val_loss[-1] - train_loss[-1]:.6f} ({'正常' if val_loss[-1] < train_loss[-1] * 1.5 else '可能过拟合'})")
    
    # 2. 角度误差分析
    print("\n2. 角度误差分析:")
    print(f"   最终Yaw误差: {val_yaw[-1]:.4f}度")
    print(f"   最终Pitch误差: {val_pitch[-1]:.4f}度")
    print(f"   最终Roll误差: {val_roll[-1]:.4f}度")
    print(f"   最终MAE: {val_mae[-1]:.4f}度")
    
    # 检查角度误差是否异常
    yaw_ok = val_yaw[-1] < 10
    pitch_ok = val_pitch[-1] < 10
    roll_ok = val_roll[-1] < 10
    
    if not yaw_ok or not pitch_ok:
        print("\n   ⚠️  警告: Yaw或Pitch误差过大！")
        print("   可能的原因:")
        print("     a) 数据预处理问题：角度单位转换错误（弧度/度数）")
        print("     b) 数据分布问题：Yaw/Pitch数据可能集中在某些角度")
        print("     c) 模型容量问题：模型可能无法学习复杂的角度映射")
        print("     d) 损失函数问题：GeodesicLoss可能对某些角度不敏感")
        print("     e) 训练不稳定：学习率可能过大或过小")
    
    # 3. 学习率分析
    print("\n3. 学习率分析:")
    print(f"   初始学习率: {lr[0]:.8f}")
    print(f"   最终学习率: {lr[-1]:.8f}")
    lr_changed = abs(lr[-1] - lr[0]) > 1e-8
    print(f"   学习率变化: {'是' if lr_changed else '否'}")
    if not lr_changed:
        print("   ⚠️  警告: 学习率在整个训练过程中没有变化！")
        print("   建议: 启用学习率调度器（--scheduler True）")
    
    # 4. 收敛分析
    print("\n4. 收敛分析:")
    # 检查最后10个epoch的变化
    if len(epochs) >= 10:
        recent_yaw = val_yaw[-10:]
        recent_pitch = val_pitch[-10:]
        recent_roll = val_roll[-10:]
        recent_loss = val_loss[-10:]
        
        yaw_std = np.std(recent_yaw)
        pitch_std = np.std(recent_pitch)
        roll_std = np.std(recent_roll)
        loss_std = np.std(recent_loss)
        
        print(f"   最后10个epoch的稳定性:")
        print(f"     Yaw标准差: {yaw_std:.4f} ({'稳定' if yaw_std < 2 else '不稳定'})")
        print(f"     Pitch标准差: {pitch_std:.4f} ({'稳定' if pitch_std < 2 else '不稳定'})")
        print(f"     Roll标准差: {roll_std:.4f} ({'稳定' if roll_std < 2 else '不稳定'})")
        print(f"     损失标准差: {loss_std:.6f} ({'稳定' if loss_std < 0.01 else '不稳定'})")
    
    # 5. 改进建议
    print("\n5. 改进建议:")
    suggestions = []
    
    if not lr_changed:
        suggestions.append("启用学习率调度器: --scheduler True --scheduler_type ReduceLROnPlateau")
    
    if val_yaw[-1] > 30 or val_pitch[-1] > 30:
        suggestions.append("检查数据预处理：确认角度单位（弧度/度数）转换正确")
        suggestions.append("检查数据分布：可视化Yaw/Pitch的分布，确保数据平衡")
        suggestions.append("尝试梯度裁剪：--grad_clip 1.0")
        suggestions.append("降低学习率：--lr 0.00005")
        suggestions.append("增加训练轮数：--num_epochs 120")
    
    if val_loss[-1] > train_loss[-1] * 1.5:
        suggestions.append("可能过拟合：增加数据增强或使用dropout")
    
    if len(suggestions) == 0:
        print("   训练状态良好，无需特别改进")
    else:
        for i, suggestion in enumerate(suggestions, 1):
            print(f"   {i}. {suggestion}")
    
    print("\n" + "="*70)


def plot_diagnosis(history_path, output_dir='output'):
    """绘制诊断图表"""
    os.makedirs(output_dir, exist_ok=True)
    
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    epochs = history['epoch']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. 损失曲线
    ax1 = axes[0, 0]
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss Curves')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 角度误差对比
    ax2 = axes[0, 1]
    ax2.plot(epochs, history['val_yaw_error'], 'b-', label='Yaw Error', linewidth=2)
    ax2.plot(epochs, history['val_pitch_error'], 'r-', label='Pitch Error', linewidth=2)
    ax2.plot(epochs, history['val_roll_error'], 'g-', label='Roll Error', linewidth=2)
    ax2.axhline(y=10, color='k', linestyle='--', alpha=0.5, label='Target (10°)')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Error (degrees)')
    ax2.set_title('Angle Errors')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 学习率变化
    ax3 = axes[1, 0]
    ax3.plot(epochs, history['learning_rate'], 'm-', linewidth=2)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Learning Rate')
    ax3.set_title('Learning Rate Schedule')
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3)
    
    # 4. MAE趋势
    ax4 = axes[1, 1]
    ax4.plot(epochs, history['val_mae'], 'g-', linewidth=2, marker='o', markersize=3)
    ax4.axhline(y=5, color='k', linestyle='--', alpha=0.5, label='Target (5°)')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('MAE (degrees)')
    ax4.set_title('Mean Absolute Error')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'training_diagnosis.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n诊断图表已保存到: {save_path}")
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Diagnose training issues')
    parser.add_argument('--history', dest='history_path',
                       help='Path to training_history.json',
                       default='output/snapshots/training_history.json', type=str)
    parser.add_argument('--output_dir', dest='output_dir',
                       help='Output directory for plots',
                       default='output', type=str)
    
    args = parser.parse_args()
    
    if not os.path.exists(args.history_path):
        print(f"错误: 找不到训练历史文件: {args.history_path}")
        exit(1)
    
    analyze_training_history(args.history_path)
    plot_diagnosis(args.history_path, args.output_dir)



