import argparse

import torch

from model import SixDRepNet_MobileNetV2, mobilenet_model_convert


parser = argparse.ArgumentParser(description='SixDRepNet MobileNetV2 Conversion')
parser.add_argument('load', metavar='LOAD', help='Path to training checkpoint (.tar/.pth)')
parser.add_argument('save', metavar='SAVE', help='Path to save deploy state_dict (.pth)')
parser.add_argument('--use_stage7_scse', action='store_true',
                    help='Enable stage7 scSE when building model for conversion')
parser.add_argument('--use_coordconv', action='store_true',
                    help='Use CoordConv(5-channel input) model variant')


def load_filtered_state_dict(model, snapshot):
    model_dict = model.state_dict()
    snapshot = {k: v for k, v in snapshot.items() if k in model_dict}
    model_dict.update(snapshot)
    model.load_state_dict(model_dict)


def convert_mobilenet():
    args = parser.parse_args()

    print('Loading MobileNetV2-based model (training form).')
    model = SixDRepNet_MobileNetV2(
        pretrained=False,
        use_stage7_scse=args.use_stage7_scse,
        use_CoordConv=args.use_coordconv,
        repconv_deploy=False,
    )

    print(f'Loading checkpoint: {args.load}')
    saved_state_dict = torch.load(args.load, map_location='cpu')
    if isinstance(saved_state_dict, dict) and 'model_state_dict' in saved_state_dict:
        load_filtered_state_dict(model, saved_state_dict['model_state_dict'])
    elif isinstance(saved_state_dict, dict):
        load_filtered_state_dict(model, saved_state_dict)
    else:
        raise ValueError('Unsupported checkpoint format. Expected dict with weights.')

    print('Converting RepConv blocks to deploy form (single 3x3 conv).')
    mobilenet_model_convert(model, save_path=args.save, do_copy=False)
    print(f'Deploy state_dict saved to: {args.save}')


if __name__ == '__main__':
    convert_mobilenet()
