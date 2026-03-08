import math

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
    - Head: RepConv x2 -> GAP -> FC -> 6D
    - forward返回6D向量，旋转矩阵转换放到训练/测试后处理
    """
    def __init__(self, pretrained=True, use_stage7_scse=True, use_CoordConv=False):
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
        self.stage3_lmfa = LMFABlock(32)
        self.stage5_alt_scse = SCSEBlock(96)
        self.stage5_lmfa = LMFABlock(96)

        # stage4/stage6: 固定使用 scSE
        self.stage4_scse = SCSEBlock(64)
        self.stage6_scse = SCSEBlock(160)
        self.use_stage7_scse = use_stage7_scse
        self.stage7_scse = SCSEBlock(320)

        # Head: RepConv x2 -> GAP -> FC -> 6D
        self.repconv1 = RepConvBlock(320, 320)
        self.repconv2 = RepConvBlock(320, 320)
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)
        self.linear_reg = nn.Linear(320, 6)
    
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
        # x = self.stage3_lmfa(x)
        x = self.stage3_alt_scse(x)

        # stage4/5
        x = self.stage4_1(x)
        x = self.stage4_2(x)
        x = self.stage4_3(x)
        x = self.stage4_4(x)
        x = self.stage4_scse(x)
        x = self.stage5_1(x)
        x = self.stage5_2(x)
        x = self.stage5_3(x)
        # x = self.stage5_lmfa(x)
        x = self.stage5_alt_scse(x)

        # stage6/7
        x = self.stage6_1(x)
        x = self.stage6_2(x)
        x = self.stage6_3(x)
        x = self.stage6_scse(x)
        x = self.stage7(x)
        if self.use_stage7_scse:
            x = self.stage7_scse(x)

        # head
        # x = self.repconv1(x)
        # x = self.repconv2(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.linear_reg(x)
        return x


class RepConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(RepConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


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


class LMFABlock(nn.Module):
    def __init__(self, channels):
        super(LMFABlock, self).__init__()
        self.b1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1,
                            groups=channels, bias=False)
        self.b2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=2,
                            dilation=2, groups=channels, bias=False)
        self.b3 = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.post_cat_scse = SCSEBlock(channels * 3)
        self.fuse = nn.Conv2d(channels * 3, channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.hsig = nn.Hardsigmoid(inplace=True)

    def forward(self, x):
        z1 = self.b1(x)
        z2 = self.b2(x)
        z3 = self.b3(x)
        z = torch.cat([z1, z2, z3], dim=1)
        z = self.post_cat_scse(z)
        z = self.fuse(z)
        z = self.hsig(z)
        return x + z