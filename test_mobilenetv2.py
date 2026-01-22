"""
测试MobileNetV2版本的6DRepNet模型
"""
import torch
import sys
import os

# 添加路径以便导入模块
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from sixdrepnet.model import SixDRepNet_MobileNetV2

def test_model():
    print("=" * 50)
    print("测试 SixDRepNet_MobileNetV2 模型")
    print("=" * 50)
    
    # 创建模型（使用预训练权重）
    print("\n1. 创建模型（使用预训练权重）...")
    try:
        model = SixDRepNet_MobileNetV2(pretrained=True)
        print("   ✓ 模型创建成功")
        print(f"   - 特征维度: {model.linear_reg.in_features}")
        print(f"   - 输出维度: {model.linear_reg.out_features}")
    except Exception as e:
        print(f"   ✗ 模型创建失败: {e}")
        return False
    
    # 创建模型（不使用预训练权重）
    print("\n2. 创建模型（不使用预训练权重）...")
    try:
        model_no_pretrain = SixDRepNet_MobileNetV2(pretrained=False)
        print("   ✓ 模型创建成功")
    except Exception as e:
        print(f"   ✗ 模型创建失败: {e}")
        return False
    
    # 测试前向传播
    print("\n3. 测试前向传播...")
    try:
        model.eval()
        batch_size = 2
        input_size = 224
        
        # 创建随机输入
        dummy_input = torch.randn(batch_size, 3, input_size, input_size)
        print(f"   - 输入形状: {dummy_input.shape}")
        
        # 前向传播
        with torch.no_grad():
            output = model(dummy_input)
        
        print(f"   ✓ 前向传播成功")
        print(f"   - 输出形状: {output.shape}")
        print(f"   - 预期形状: [{batch_size}, 3, 3]")
        
        # 验证输出形状
        expected_shape = (batch_size, 3, 3)
        if output.shape == expected_shape:
            print(f"   ✓ 输出形状正确")
        else:
            print(f"   ✗ 输出形状错误: 期望 {expected_shape}, 得到 {output.shape}")
            return False
        
        # 验证是否为旋转矩阵（检查行列式是否接近1）
        det = torch.det(output)
        print(f"   - 旋转矩阵行列式: {det.mean().item():.6f} (应该接近1.0)")
        
        # 验证正交性（R @ R^T 应该接近单位矩阵）
        identity_approx = torch.bmm(output, output.transpose(1, 2))
        identity = torch.eye(3).unsqueeze(0).expand(batch_size, -1, -1)
        ortho_error = torch.mean(torch.abs(identity_approx - identity))
        print(f"   - 正交性误差: {ortho_error.item():.6f} (应该接近0)")
        
    except Exception as e:
        print(f"   ✗ 前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 测试不同输入尺寸
    print("\n4. 测试不同输入尺寸...")
    try:
        test_sizes = [(1, 3, 112, 112), (1, 3, 256, 256)]
        for size in test_sizes:
            dummy_input = torch.randn(*size)
            with torch.no_grad():
                output = model(dummy_input)
            print(f"   ✓ 输入 {size} -> 输出 {output.shape}")
    except Exception as e:
        print(f"   ✗ 不同尺寸测试失败: {e}")
        return False
    
    print("\n" + "=" * 50)
    print("所有测试通过！✓")
    print("=" * 50)
    return True

if __name__ == '__main__':
    success = test_model()
    sys.exit(0 if success else 1)

