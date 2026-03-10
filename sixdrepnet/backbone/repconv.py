import copy

import torch
from torch import nn


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
    """

    if do_copy:
        model = copy.deepcopy(model)
    repconv_model_convert(model)
    if save_path is not None:
        torch.save(model.state_dict(), save_path)
    return model
