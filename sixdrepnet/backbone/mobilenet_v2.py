"""
DEPRECATED:
This local MobileNetV2 implementation is kept only for backward compatibility.
Current project default for SixDRepNet_MobileNetV2 uses torchvision's
`mobilenet_v2` inside `sixdrepnet/model.py`.

Please treat `sixdrepnet/model.py` as the single source of truth for the
active MobileNetV2-based 6DRepNet architecture.
"""

import torch.nn as nn


def _make_divisible(v, divisor, min_value=None):
    """
    确保通道数能被divisor整除
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(nn.Sequential):
    """
    标准卷积+BN+ReLU6模块
    """
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):
    """
    MobileNetV2的倒残差块（Inverted Residual Block）
    """
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # 1x1 pointwise expansion
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            # 3x3 depthwise
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # 1x1 pointwise linear (no activation)
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    """
    MobileNetV2网络结构
    参考论文: MobileNetV2: Inverted Residuals and Linear Bottlenecks
    """
    def __init__(self, num_classes=1000, width_mult=1.0, inverted_residual_setting=None, round_nearest=8):
        """
        Args:
            num_classes: 分类类别数（这里不使用，但保留接口）
            width_mult: 宽度乘数，用于调整模型大小
            inverted_residual_setting: 倒残差块的配置
            round_nearest: 通道数对齐到最近的倍数
        """
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            # MobileNetV2的标准配置
            inverted_residual_setting = [
                # t, c, n, s
                # t: expansion ratio
                # c: output channels
                # n: number of blocks
                # s: stride
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # 只构建特征提取部分，不包含分类器
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        
        # 第一层：标准卷积
        features = [ConvBNReLU(3, input_channel, stride=2)]
        
        # 倒残差块
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        
        # 最后一层：1x1卷积扩展到高维特征空间
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))
        
        self.features = nn.Sequential(*features)

    def forward(self, x):
        x = self.features(x)
        return x


def mobilenet_v2(pretrained=False, **kwargs):
    """
    创建MobileNetV2模型
    
    Args:
        pretrained: 是否加载预训练权重（这里不使用，权重通过外部加载）
        **kwargs: 其他参数传递给MobileNetV2
    """
    model = MobileNetV2(**kwargs)
    return model









