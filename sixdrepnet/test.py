import time
import math
import re
import sys
import os
import argparse

import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.backends import cudnn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import matplotlib
from matplotlib import pyplot as plt
from PIL import Image
matplotlib.use('TkAgg')

from model import SixDRepNet, SixDRepNet_MobileNetV2
import utils
import datasets

def str2bool(v):
    if isinstance(v, bool):
        return v
    value = str(v).strip().lower()
    if value in ('yes', 'true', 't', '1', 'y'):
        return True
    if value in ('no', 'false', 'f', '0', 'n'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='Head pose estimation using the 6DRepNet.')
    default_gpu = 0 if torch.cuda.is_available() else -1
    parser.add_argument('--gpu',
                        dest='gpu_id', help=f'GPU device id to use (default: {default_gpu} if GPU available, else CPU)',
                        default=default_gpu, type=int)
    parser.add_argument('--data_dir',
                        dest='data_dir', help='Directory path for data.',
                        default='../../../datasets/AFLW2000', type=str)
                        # default='../../../datasets/BIWI', type=str)
    parser.add_argument('--filename_list',
                        dest='filename_list',
                        help='Path to text file containing relative paths for every example.',
                        default='../../../datasets/AFLW2000/files.txt', type=str)  # datasets/BIWI_noTrack.npz
                        # default='../../../datasets/BIWI/files.txt', type=str)
    parser.add_argument('--snapshot',
                        dest='snapshot', help='Name of model snapshot.',
                        default='', type=str)
    parser.add_argument('--batch_size',
                        dest='batch_size', help='Batch size.',
                        default=64, type=int)
    parser.add_argument('--show_viz',
                        dest='show_viz', help='Save images with pose cube.',
                        default=False, type=str2bool)
    parser.add_argument('--dataset',
                        dest='dataset', help='Dataset type.',
                        default='AFLW2000', type=str)
    parser.add_argument('--backbone',
                        dest='backbone', help='Backbone type: RepVGG or MobileNetV2',
                        default='MobileNetV2', type=str)

    args = parser.parse_args()
    return args

def load_filtered_state_dict(model, snapshot):
    # By user apaszke from discuss.pytorch.org
    model_dict = model.state_dict()
    snapshot = {k: v for k, v in snapshot.items() if k in model_dict}
    model_dict.update(snapshot)
    model.load_state_dict(model_dict)


def output_to_rotation_matrix(model_output):
    if isinstance(model_output, torch.Tensor) and model_output.dim() == 2 and model_output.size(1) == 6:
        return utils.compute_rotation_matrix_from_ortho6d(model_output)
    return model_output

if __name__ == '__main__':
    args = parse_args()
    cudnn.enabled = True
    gpu = args.gpu_id
    snapshot_path = args.snapshot
    use_gpu = (gpu >= 0 and torch.cuda.is_available())
    device = torch.device(f'cuda:{gpu}' if use_gpu else 'cpu')
    
    # 根据backbone类型创建模型
    backbone_name = str(args.backbone).strip().lower()
    if backbone_name in ('mobilenetv2', 'mobilenet_v2', 'mobilenet'):
        model = SixDRepNet_MobileNetV2(pretrained=False)
        print('Using MobileNetV2 backbone')
    elif backbone_name in ('repvgg', 'repvgg-b1g2'):
        model = SixDRepNet(backbone_name='RepVGG-B1g2',
                          backbone_file='',
                          deploy=True,
                          pretrained=False)
        print('Using RepVGG-B1g2 backbone')
    else:
        raise ValueError(f'Unsupported backbone: {args.backbone}. Use MobileNetV2/MobileNet_V2 or RepVGG.')

    print('Loading data.')

    if backbone_name in ('mobilenetv2', 'mobilenet_v2', 'mobilenet'):
        transformations = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            utils.AddCoordChannels(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406, 0.0, 0.0], std=[0.229, 0.224, 0.225, 1.0, 1.0])
        ])
    else:
        transformations = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    pose_dataset = datasets.getDataset(
        args.dataset, args.data_dir, args.filename_list, transformations, train_mode = False)
    test_loader = torch.utils.data.DataLoader(
        dataset=pose_dataset,
        batch_size=args.batch_size,
        num_workers=2)


    # Load snapshot
    saved_state_dict = torch.load(snapshot_path, map_location=device)

    if 'model_state_dict' in saved_state_dict:
        model.load_state_dict(saved_state_dict['model_state_dict'])
    else:
        model.load_state_dict(saved_state_dict)    
    model = model.to(device)

    # Test the Model
    model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
    
    total = 0
    yaw_error = pitch_error = roll_error = .0
    v1_err = v2_err = v3_err = .0

    with torch.no_grad():

        for i, (images, r_label, cont_labels, name) in enumerate(test_loader):
            images = torch.Tensor(images).to(device)
            total += cont_labels.size(0)

            # gt matrix
            R_gt = r_label

            # gt euler
            y_gt_deg = cont_labels[:, 0].float()*180/np.pi
            p_gt_deg = cont_labels[:, 1].float()*180/np.pi
            r_gt_deg = cont_labels[:, 2].float()*180/np.pi

            pred_out = model(images)
            R_pred = output_to_rotation_matrix(pred_out)

            euler = utils.compute_euler_angles_from_rotation_matrices(
                R_pred)*180/np.pi
            p_pred_deg = euler[:, 0].cpu()
            y_pred_deg = euler[:, 1].cpu()
            r_pred_deg = euler[:, 2].cpu()

            R_pred = R_pred.cpu()
            v1_err += torch.sum(torch.acos(torch.clamp(
                torch.sum(R_gt[:, 0] * R_pred[:, 0], 1), -1, 1)) * 180/np.pi)
            v2_err += torch.sum(torch.acos(torch.clamp(
                torch.sum(R_gt[:, 1] * R_pred[:, 1], 1), -1, 1)) * 180/np.pi)
            v3_err += torch.sum(torch.acos(torch.clamp(
                torch.sum(R_gt[:, 2] * R_pred[:, 2], 1), -1, 1)) * 180/np.pi)

            pitch_error += torch.sum(torch.min(torch.stack((torch.abs(p_gt_deg - p_pred_deg), torch.abs(p_pred_deg + 360 - p_gt_deg), torch.abs(
                p_pred_deg - 360 - p_gt_deg), torch.abs(p_pred_deg + 180 - p_gt_deg), torch.abs(p_pred_deg - 180 - p_gt_deg))), 0)[0])
            yaw_error += torch.sum(torch.min(torch.stack((torch.abs(y_gt_deg - y_pred_deg), torch.abs(y_pred_deg + 360 - y_gt_deg), torch.abs(
                y_pred_deg - 360 - y_gt_deg), torch.abs(y_pred_deg + 180 - y_gt_deg), torch.abs(y_pred_deg - 180 - y_gt_deg))), 0)[0])
            roll_error += torch.sum(torch.min(torch.stack((torch.abs(r_gt_deg - r_pred_deg), torch.abs(r_pred_deg + 360 - r_gt_deg), torch.abs(
                r_pred_deg - 360 - r_gt_deg), torch.abs(r_pred_deg + 180 - r_gt_deg), torch.abs(r_pred_deg - 180 - r_gt_deg))), 0)[0])

            if args.show_viz:
                name = name[0]
                if args.dataset == 'AFLW2000':
                    cv2_img = cv2.imread(os.path.join(args.data_dir, name + '.jpg'))
                   
                elif args.dataset == 'BIWI':
                    vis = np.uint8(name)
                    h,w,c = vis.shape
                    vis2 = cv2.CreateMat(h, w, cv2.CV_32FC3)
                    vis0 = cv2.fromarray(vis)
                    cv2.CvtColor(vis0, vis2, cv2.CV_GRAY2BGR)
                    cv2_img = cv2.imread(vis2)
                utils.draw_axis(cv2_img, y_pred_deg[0], p_pred_deg[0], r_pred_deg[0], tdx=200, tdy=200, size=100)
                #utils.plot_pose_cube(cv2_img, y_pred_deg[0], p_pred_deg[0], r_pred_deg[0], size=200)
                cv2.imshow("Test", cv2_img)
                cv2.waitKey(5)
                cv2.imwrite(os.path.join('output/img/',name+'.png'),cv2_img)
                
        print('Yaw: %.4f, Pitch: %.4f, Roll: %.4f, MAE: %.4f' % (
            yaw_error / total, pitch_error / total, roll_error / total,
            (yaw_error + pitch_error + roll_error) / (total * 3)))

        # print('Vec1: %.4f, Vec2: %.4f, Vec3: %.4f, VMAE: %.4f' % (
        #     v1_err / total, v2_err / total, v3_err / total,
        #     (v1_err + v2_err + v3_err) / (total * 3)))


