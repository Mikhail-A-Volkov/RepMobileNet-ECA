import os
import sys

import torch
from thop import profile


def format_count(value):
    if value >= 1e9:
        return f"{value / 1e9:.3f} G"
    if value >= 1e6:
        return f"{value / 1e6:.3f} M"
    if value >= 1e3:
        return f"{value / 1e3:.3f} K"
    return f"{value:.0f}"


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def measure_model(model, input_tensor):
    model.eval()
    with torch.no_grad():
        macs, _ = profile(model, inputs=(input_tensor,), verbose=False)
    params = count_params(model)
    # Different communities use different conventions:
    # - Convention A: FLOPs == MACs (1 multiply-accumulate counted as 1 operation)
    # - Convention B: FLOPs == 2 * MACs (multiply + add counted separately)
    flops_conv_a = macs
    flops_conv_b = macs * 2
    return {
        "params": params,
        "macs": macs,
        "flops_conv_a": flops_conv_a,
        "flops_conv_b": flops_conv_b,
    }


def print_result(name, result, input_shape):
    print(f"\n[{name}]")
    print(f"Input shape : {input_shape}")
    print(f"Params      : {result['params']:,} ({format_count(result['params'])})")
    print(f"MACs        : {result['macs']:,.0f} ({format_count(result['macs'])})")
    print(
        f"FLOPs(A=MACs): {result['flops_conv_a']:,.0f} "
        f"({format_count(result['flops_conv_a'])})"
    )
    print(
        f"FLOPs(B=2*MACs): {result['flops_conv_b']:,.0f} "
        f"({format_count(result['flops_conv_b'])})"
    )


def main():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    sixdrepnet_dir = os.path.join(root_dir, "sixdrepnet")
    if sixdrepnet_dir not in sys.path:
        sys.path.insert(0, sixdrepnet_dir)

    from model import SixDRepNet, SixDRepNet_MobileNetV2

    repvgg_model = SixDRepNet(
        backbone_name="RepVGG-B1g2",
        backbone_file="",
        deploy=False,
        pretrained=False,
    )
    mobilenet_model = SixDRepNet_MobileNetV2(pretrained=False)

    repvgg_input = torch.randn(1, 3, 224, 224)
    mobilenet_input = torch.randn(1, 5, 224, 224)

    repvgg_result = measure_model(repvgg_model, repvgg_input)
    mobilenet_result = measure_model(mobilenet_model, mobilenet_input)

    print("=" * 72)
    print("Model Complexity Comparison (Batch=1)")
    print("=" * 72)

    print_result("SixDRepNet-RepVGG-B1g2", repvgg_result, tuple(repvgg_input.shape))
    print_result("SixDRepNet-MobileNetV2", mobilenet_result, tuple(mobilenet_input.shape))

    params_reduce = (1 - mobilenet_result["params"] / repvgg_result["params"]) * 100
    macs_reduce = (1 - mobilenet_result["macs"] / repvgg_result["macs"]) * 100
    flops_reduce_a = (1 - mobilenet_result["flops_conv_a"] / repvgg_result["flops_conv_a"]) * 100
    flops_reduce_b = (1 - mobilenet_result["flops_conv_b"] / repvgg_result["flops_conv_b"]) * 100

    print("\n" + "-" * 72)
    print("MobileNetV2 relative to RepVGG-B1g2:")
    print(f"Params reduction: {params_reduce:.2f}%")
    print(f"MACs reduction  : {macs_reduce:.2f}%")
    print(f"FLOPs(A) reduction: {flops_reduce_a:.2f}%")
    print(f"FLOPs(B) reduction: {flops_reduce_b:.2f}%")
    print("-" * 72)


if __name__ == "__main__":
    main()

