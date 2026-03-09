import math
import copy

import torch
from torch import nn
from torchvision.models import mobilenet_v2

from backbone.repvgg import get_RepVGG_func_by_name
import utils

class SixDRepNet(nn.Module):
    def __init__(self,
                 backbone_name, backbone_file, deploy,
                 pretrained=True):
        super(SixDRepNet, self).__init__()
        repvgg_fn = get_RepVGG_func_by_name(backbone_name)
        backbone = repvgg_fn(deploy)
        if pretrained:
            checkpoint = torch.load(backbone_file)
            if 'state_dict' in checkpoint:
                checkpoint = checkpoint['state_dict']
            ckpt = {k.replace('module.', ''): v for k,
                    v in checkpoint.items()}  # strip the names
            backbone.load_state_dict(ckpt)

        self.layer0, self.layer1, self.layer2, self.layer3, self.layer4 = backbone.stage0, backbone.stage1, backbone.stage2, backbone.stage3, backbone.stage4
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)

        last_channel = 0
        for n, m in self.layer4.named_modules():
            if ('rbr_dense' in n or 'rbr_reparam' in n) and isinstance(m, nn.Conv2d):
                last_channel = m.out_channels

        fea_dim = last_channel

        self.linear_reg = nn.Linear(fea_dim, 6)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.linear_reg(x)
        return utils.compute_rotation_matrix_from_ortho6d(x)


class SixDRepNet2(nn.Module):
    def __init__(self, block, layers, fc_layers=1):
        self.inplanes = 64
        super(SixDRepNet2, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)

        self.linear_reg = nn.Linear(512*block.expansion,6)
      


        # Vestigial layer from previous experiments
        self.fc_finetune = nn.Linear(512 * block.expansion + 3, 3)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        x = self.linear_reg(x)        
        out = utils.compute_rotation_matrix_from_ortho6d(x)

        return out


class SixDRepNet_MobileNetV2(nn.Module):
    """
    使用改造后的MobileNetV2作为backbone的6DRepNet模型
    - CoordConv(3->5)在预处理完成，本模型输入为5通道
    - stage3/stage5可二选一使用scSE或LMFA
    - stage4/stage6后接scSE
    - 支持注释切换为scSE-ECA分支（仅模型结构改动）
    - Head: RepConv x2 -> GAP -> FC -> 6D
    - forward返回6D向量，旋转矩阵转换放到训练/测试后处理
    """
    def __init__(self, pretrained=True, use_stage7_scse=False, use_CoordConv=False, repconv_deploy=False):
        super(SixDRepNet_MobileNetV2, self).__init__()
        mobilenet = mobilenet_v2(pretrained=pretrained)

        # Stem: Conv3x3(5->32,s=2) + BN + ReLU6
        if use_CoordConv:
            self.stem_conv = nn.Conv2d(5, 32, kernel_size=3, stride=2, padding=1, bias=False)
        else:
            self.stem_conv = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
            
        self.stem_bn = nn.BatchNorm2d(32)
        self.stem_relu = nn.ReLU6(inplace=True)

        # 初始化stem（前三通道拷贝预训练，后两通道置零）
        if pretrained:
            with torch.no_grad():
                pre_conv = mobilenet.features[0][0].weight
                self.stem_conv.weight.zero_()
                self.stem_conv.weight[:, :3, :, :] = pre_conv
                self.stem_bn.weight.copy_(mobilenet.features[0][1].weight)
                self.stem_bn.bias.copy_(mobilenet.features[0][1].bias)
                self.stem_bn.running_mean.copy_(mobilenet.features[0][1].running_mean)
                self.stem_bn.running_var.copy_(mobilenet.features[0][1].running_var)

        # stage1~stage7（原样）
        self.stage1 = mobilenet.features[1]    # 16
        self.stage2_1 = mobilenet.features[2]  # 24
        self.stage2_2 = mobilenet.features[3]
        self.stage3_1 = mobilenet.features[4]  # 32
        self.stage3_2 = mobilenet.features[5]
        self.stage3_3 = mobilenet.features[6]
        self.stage4_1 = mobilenet.features[7]  # 64
        self.stage4_2 = mobilenet.features[8]
        self.stage4_3 = mobilenet.features[9]
        self.stage4_4 = mobilenet.features[10]
        self.stage5_1 = mobilenet.features[11]  # 96
        self.stage5_2 = mobilenet.features[12]
        self.stage5_3 = mobilenet.features[13]
        self.stage6_1 = mobilenet.features[14]  # 160
        self.stage6_2 = mobilenet.features[15]
        self.stage6_3 = mobilenet.features[16]
        self.stage7 = mobilenet.features[17]    # 320

        # stage3/stage5: scSE 与 LMFA 二选一
        self.stage3_alt_scse = SCSEBlock(32)
        self.stage3_alt_scse_eca = SCSEECABlock(32)
        self.stage3_lmfa = LMFABlock(32)
        self.stage5_alt_scse = SCSEBlock(96)
        self.stage5_alt_scse_eca = SCSEECABlock(96)
        self.stage5_lmfa = LMFABlock(96)

        # stage4/stage6: 固定使用 scSE
        self.stage4_scse = SCSEBlock(64)
        self.stage4_scse_eca = SCSEECABlock(64)
        self.stage6_scse = SCSEBlock(160)
        self.stage6_scse_eca = SCSEECABlock(160)
        self.use_stage7_scse = use_stage7_scse
        self.stage7_scse = SCSEBlock(320)
        self.stage7_scse_eca = SCSEECABlock(320)

        # Head: RepConv x2 -> GAP -> FC -> 6D
        self.repconv1 = RepConvBlock(320, 320, deploy=repconv_deploy)
        self.repconv2 = RepConvBlock(320, 320, deploy=repconv_deploy)
        # Legacy static head blocks (old "RepConv" style: Conv3x3+BN+ReLU6).
        # Uncomment for parameter/ablation reproduction of the earlier static design.
        self.static_repconv1 = StaticRepConvBlock(320, 320)
        self.static_repconv2 = StaticRepConvBlock(320, 320)
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)
        self.linear_reg = nn.Linear(320, 6)
        # Optional decoupled 2D heads (yaw/pitch/roll); concat -> 6D.
        # This is a model-only change and keeps output shape [B, 6].
        self.yaw_reg_2d = nn.Linear(320, 2)
        self.pitch_reg_2d = nn.Linear(320, 2)
        self.roll_reg_2d = nn.Linear(320, 2)
    
    def forward(self, x):
        # stem
        x = self.stem_conv(x)
        x = self.stem_bn(x)
        x = self.stem_relu(x)

        # stage1/2/3
        x = self.stage1(x)
        x = self.stage2_1(x)
        x = self.stage2_2(x)
        x = self.stage3_1(x)
        x = self.stage3_2(x)
        x = self.stage3_3(x)
        x = self.stage3_lmfa(x)
        # x = self.stage3_alt_scse(x)
        # x = self.stage3_alt_scse_eca(x)

        # stage4/5
        x = self.stage4_1(x)
        x = self.stage4_2(x)
        x = self.stage4_3(x)
        x = self.stage4_4(x)
        x = self.stage4_scse(x)
        # x = self.stage4_scse_eca(x)
        x = self.stage5_1(x)
        x = self.stage5_2(x)
        x = self.stage5_3(x)
        x = self.stage5_lmfa(x)
        # x = self.stage5_alt_scse(x)
        # x = self.stage5_alt_scse_eca(x)

        # stage6/7
        x = self.stage6_1(x)
        x = self.stage6_2(x)
        x = self.stage6_3(x)
        x = self.stage6_scse(x)
        # x = self.stage6_scse_eca(x)
        x = self.stage7(x)
        if self.use_stage7_scse:
            x = self.stage7_scse(x)
            # x = self.stage7_scse_eca(x)

        # head
        # x = self.repconv1(x)
        # x = self.repconv2(x)
        # x = self.static_repconv1(x)
        # x = self.static_repconv2(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        # Optional decoupled 2D head path:
        # yaw_2d = self.yaw_reg_2d(x)
        # pitch_2d = self.pitch_reg_2d(x)
        # roll_2d = self.roll_reg_2d(x)
        # x = torch.cat([yaw_2d, pitch_2d, roll_2d], dim=1)
        x = self.linear_reg(x)
        return x


class StaticRepConvBlock(nn.Module):
    """
    Legacy static "RepConv" block used in earlier experiments:
    a plain 3x3 Conv + BN + ReLU6 (no multi-branch re-parameterization).
    """
    def __init__(self, in_channels, out_channels, stride=1, groups=1):
        super(StaticRepConvBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class RepConvBlock(nn.Module):
    """
    RepConv block with training-time multi-branch structure and deploy-time
    single 3x3 conv re-parameterization.
    """
    def __init__(self, in_channels, out_channels, stride=1, groups=1, deploy=False):
        super(RepConvBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.groups = groups
        self.deploy = deploy
        self.act = nn.ReLU6(inplace=True)

        if deploy:
            self.rbr_reparam = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=groups,
                bias=True,
            )
        else:
            self.rbr_dense = self._conv_bn(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=groups,
            )
            self.rbr_1x1 = self._conv_bn(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
                padding=0,
                groups=groups,
            )
            self.rbr_identity = (
                nn.BatchNorm2d(num_features=in_channels)
                if out_channels == in_channels and stride == 1
                else None
            )

    @staticmethod
    def _conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups):
        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=out_channels),
        )

    def forward(self, x):
        if hasattr(self, "rbr_reparam"):
            return self.act(self.rbr_reparam(x))

        identity_out = self.rbr_identity(x) if self.rbr_identity is not None else 0
        return self.act(self.rbr_dense(x) + self.rbr_1x1(x) + identity_out)

    def _pad_1x1_to_3x3(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0

        if isinstance(branch, nn.Sequential):
            conv = branch[0]
            bn = branch[1]
            kernel = conv.weight
            running_mean = bn.running_mean
            running_var = bn.running_var
            gamma = bn.weight
            beta = bn.bias
            eps = bn.eps
        else:
            # identity branch: BN only
            bn = branch
            if not hasattr(self, "id_tensor"):
                input_dim = self.in_channels // self.groups
                kernel_value = torch.zeros(
                    (self.in_channels, input_dim, 3, 3),
                    dtype=bn.weight.dtype,
                    device=bn.weight.device,
                )
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1.0
                self.id_tensor = kernel_value
            kernel = self.id_tensor
            running_mean = bn.running_mean
            running_var = bn.running_var
            gamma = bn.weight
            beta = bn.bias
            eps = bn.eps

        std = torch.sqrt(running_var + eps)
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return (
            kernel3x3 + self._pad_1x1_to_3x3(kernel1x1) + kernelid,
            bias3x3 + bias1x1 + biasid,
        )

    def switch_to_deploy(self):
        if hasattr(self, "rbr_reparam"):
            return

        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=3,
            stride=self.stride,
            padding=1,
            groups=self.groups,
            bias=True,
        )
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias

        self.__delattr__("rbr_dense")
        self.__delattr__("rbr_1x1")
        if hasattr(self, "rbr_identity"):
            self.__delattr__("rbr_identity")
        if hasattr(self, "id_tensor"):
            self.__delattr__("id_tensor")
        self.deploy = True


def repconv_model_convert(model: torch.nn.Module):
    """
    Convert all RepConvBlock modules in a model to deploy form.
    """
    for module in model.modules():
        if isinstance(module, RepConvBlock):
            module.switch_to_deploy()
    return model


def mobilenet_model_convert(model: torch.nn.Module, save_path=None, do_copy=True):
    """
    Convert MobileNetV2-based 6DRepNet model to deploy form by fusing RepConv
    blocks into single 3x3 convolutions.

    Args:
        model: model instance (typically SixDRepNet_MobileNetV2 with repconv_deploy=False)
        save_path: optional path to save converted state_dict
        do_copy: whether to deepcopy before conversion
    """
    if do_copy:
        model = copy.deepcopy(model)
    repconv_model_convert(model)
    if save_path is not None:
        torch.save(model.state_dict(), save_path)
    return model


class SCSEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SCSEBlock, self).__init__()
        reduced = max(channels // reduction, 1)
        self.cse_pool = nn.AdaptiveAvgPool2d(1)
        self.cse_fc1 = nn.Conv2d(channels, reduced, kernel_size=1, bias=True)
        self.cse_relu = nn.ReLU(inplace=True)
        self.cse_fc2 = nn.Conv2d(reduced, channels, kernel_size=1, bias=True)
        self.cse_hs = nn.Hardsigmoid(inplace=True)
        self.sse_conv = nn.Conv2d(channels, 1, kernel_size=1, bias=True)
        self.sse_hs = nn.Hardsigmoid(inplace=True)

    def forward(self, x):
        c = self.cse_pool(x)
        c = self.cse_fc1(c)
        c = self.cse_relu(c)
        c = self.cse_fc2(c)
        c = self.cse_hs(c)
        s = self.sse_conv(x)
        s = self.sse_hs(s)
        return x * c + x * s


class ECAChannelGate(nn.Module):
    """
    ECA channel gate: GAP -> local cross-channel interaction -> gate.
    """
    def __init__(self, channels, k_size=3):
        super(ECAChannelGate, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1d = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.hsig = nn.Hardsigmoid(inplace=True)

    def forward(self, x):
        y = self.avg_pool(x)  # [B, C, 1, 1]
        y = y.squeeze(-1).transpose(-1, -2)  # [B, 1, C]
        y = self.conv1d(y)
        y = self.hsig(y)
        y = y.transpose(-1, -2).unsqueeze(-1)  # [B, C, 1, 1]
        return y


class SCSEECABlock(nn.Module):
    """
    scSE-ECA variant:
    - cSE branch replaced with ECA channel gate
    - sSE branch unchanged
    """
    def __init__(self, channels, eca_k_size=3):
        super(SCSEECABlock, self).__init__()
        self.eca_gate = ECAChannelGate(channels, k_size=eca_k_size)
        self.sse_conv = nn.Conv2d(channels, 1, kernel_size=1, bias=True)
        self.sse_hs = nn.Hardsigmoid(inplace=True)

    def forward(self, x):
        c = self.eca_gate(x)
        s = self.sse_conv(x)
        s = self.sse_hs(s)
        return x * c + x * s


class LMFABlock(nn.Module):
    def __init__(self, channels):
        super(LMFABlock, self).__init__()
        self.b1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1,
                            groups=channels, bias=False)
        self.b1_bn = nn.BatchNorm2d(channels)
        self.b2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=2,
                            dilation=2, groups=channels, bias=False)
        self.b2_bn = nn.BatchNorm2d(channels)
        self.b3 = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.b3_bn = nn.BatchNorm2d(channels)
        self.fuse = nn.Conv2d(channels * 3, channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.fuse_bn = nn.BatchNorm2d(channels)
        self.fuse_act = nn.ReLU6(inplace=True)
        self.post_fuse_scse = SCSEECABlock(channels) # 使用scSE-ECA分支
        self.hsig = nn.Hardsigmoid(inplace=True)

    def forward(self, x):
        z1 = self.b1_bn(self.b1(x))
        z2 = self.b2_bn(self.b2(x))
        z3 = self.b3_bn(self.b3(x))
        z = torch.cat([z1, z2, z3], dim=1)
        z = self.fuse(z)
        z = self.fuse_bn(z)
        z = self.fuse_act(z)
        z = self.post_fuse_scse(z)
        return x + z
        # z = self.hsig(z)
        # return x * z