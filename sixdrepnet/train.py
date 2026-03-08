import time
import math
import re
import sys
import os
import argparse
import datetime

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.backends import cudnn
from torch.utils import model_zoo
import torchvision
from torchvision import transforms
import matplotlib
from matplotlib import pyplot as plt
from PIL import Image
# matplotlib.use('TkAgg')
matplotlib.use('Agg')

from model import SixDRepNet, SixDRepNet2, SixDRepNet_MobileNetV2
import datasets
from loss import GeodesicLoss
import utils
import json
from model_profiler import profile_model

def str2bool(v):
    if isinstance(v, bool):
        return v
    return str(v).lower() in ('true', '1', 'yes', 'y', 't')

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='Head pose estimation using the 6DRepNet.')
    # 检查是否有GPU可用，如果有则默认使用GPU 0
    default_gpu = 0 if torch.cuda.is_available() else -1
    parser.add_argument(
        '--gpu', dest='gpu_id', 
        help=f'GPU device id to use (default: {default_gpu} if GPU available, else CPU)',
        default=default_gpu, type=int)
    parser.add_argument(
        '--num_epochs', dest='num_epochs',
        help='Maximum number of training epochs.',
        default=100, type=int)
    parser.add_argument(
        '--batch_size', dest='batch_size', help='Batch size.',
        default=80, type=int)
    parser.add_argument(
        '--lr', dest='lr', help='Base learning rate.',
        default=0.0001, type=float)
    parser.add_argument('--scheduler', default=True, type=str2bool)
    parser.add_argument('--scheduler_type', dest='scheduler_type', 
                       help='Scheduler type: MultiStepLR or ReduceLROnPlateau',
                       default='ReduceLROnPlateau', type=str)
    parser.add_argument('--grad_clip', dest='grad_clip', 
                       help='Gradient clipping value (0 to disable)',
                       default=0.0, type=float)
    parser.add_argument(
        '--dataset', dest='dataset', help='Dataset type.',
        default='Pose_300W_LP', type=str) #Pose_300W_LP
    parser.add_argument(
        '--data_dir', dest='data_dir', help='Directory path for data.',
        default='../../../datasets/300W_LP', type=str)#BIWI_70_30_train.npz
    parser.add_argument(
        '--filename_list', dest='filename_list',
        help='Path to text file containing relative paths for every example.',
        default='../../../datasets/300W_LP/files.txt', type=str) #BIWI_70_30_train.npz #300W_LP/files.txt
    parser.add_argument(
        '--output_string', dest='output_string',
        help='String appended to output snapshots.', default='', type=str)
    parser.add_argument(
        '--snapshot', dest='snapshot', help='Path of model snapshot.',
        default='', type=str)
    parser.add_argument(
        '--backbone', dest='backbone', help='Backbone type: RepVGG or MobileNetV2',
        default='MobileNetV2', type=str)
    parser.add_argument(
        '--val_split', dest='val_split', 
        help='Validation split ratio from training set (e.g., 0.1 for 10%%)',
        default=0.1, type=float)
    parser.add_argument(
        '--val_dataset', dest='val_dataset', 
        help='Validation dataset type. If None, will split from training set. Otherwise use separate dataset (not recommended).',
        default=None, type=str)
    parser.add_argument(
        '--val_data_dir', dest='val_data_dir', help='Directory path for validation data (only used if val_dataset is specified).',
        default=None, type=str)
    parser.add_argument(
        '--val_filename_list', dest='val_filename_list',
        help='Path to text file containing relative paths for validation examples (only used if val_dataset is specified).',
        default=None, type=str)
    parser.add_argument(
        '--val_seed', dest='val_seed', 
        help='Random seed for train/val split',
        default=42, type=int)
    parser.add_argument(
        '--optimizer_mode', dest='optimizer_mode',
        help='Optimizer mode: original (Adam) or improved (AdamW)',
        default='original', type=str)
    parser.add_argument(
        '--weight_decay', dest='weight_decay',
        help='Weight decay for improved optimizer (AdamW)',
        default=1e-4, type=float)
    parser.add_argument(
        '--use_distillation', dest='use_distillation',
        help='Enable knowledge distillation from RepVGG teacher',
        default=False, type=str2bool)
    parser.add_argument(
        '--distill_alpha', dest='distill_alpha',
        help='Loss weight for distillation term. total=(1-alpha)*gt + alpha*distill',
        default=0.3, type=float)
    parser.add_argument(
        '--teacher_snapshot', dest='teacher_snapshot',
        help='Optional checkpoint path for teacher model. If empty, use RepVGG pretrained backbone weights.',
        default='../../../output/best_epoch_for_6DRepNet.tar', type=str)

    args = parser.parse_args()
    return args

def load_filtered_state_dict(model, snapshot):
    # By user apaszke from discuss.pytorch.org
    model_dict = model.state_dict()
    snapshot = {k: v for k, v in snapshot.items() if k in model_dict}
    model_dict.update(snapshot)
    model.load_state_dict(model_dict)


def output_to_rotation_matrix(model_output):
    """
    兼容两种输出:
    - [B, 3, 3]: 已是旋转矩阵
    - [B, 6]: 6D表示，转换为旋转矩阵
    """
    if isinstance(model_output, torch.Tensor) and model_output.dim() == 2 and model_output.size(1) == 6:
        return utils.compute_rotation_matrix_from_ortho6d(model_output)
    return model_output


def validate(model, val_loader, criterion, gpu):
    """在验证集上评估模型"""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    
    yaw_error = pitch_error = roll_error = 0.0
    
    with torch.no_grad():
        for images, r_label, cont_labels, _ in val_loader:
            if gpu >= 0:
                images = images.cuda(gpu)
                R_gt = r_label.cuda(gpu)
            else:
                images = images
                R_gt = r_label
            
            # 预测（模型可输出6D或旋转矩阵）
            pred_out = model(images)
            R_pred = output_to_rotation_matrix(pred_out)
            
            # 计算损失
            loss = criterion(R_gt, R_pred)
            total_loss += loss.item() * images.size(0)
            total_samples += images.size(0)
            
            # 计算欧拉角误差
            # 注意：compute_euler_angles_from_rotation_matrices返回的是[x, y, z]
            # 根据get_R函数：x=pitch, y=yaw, z=roll
            euler_pred = utils.compute_euler_angles_from_rotation_matrices(R_pred) * 180 / np.pi
            
            # 处理cont_labels：如果为空列表或列表，从旋转矩阵计算欧拉角
            # DataLoader收集批次时，如果原始数据是空列表，会保持为列表
            if isinstance(cont_labels, list):
                # 从旋转矩阵计算ground truth欧拉角
                # compute_euler_angles返回[pitch, yaw, roll] = [x, y, z]
                euler_gt = utils.compute_euler_angles_from_rotation_matrices(R_gt.cpu()) * 180 / np.pi
            elif isinstance(cont_labels, torch.Tensor) and cont_labels.size(0) > 0:
                # cont_labels是tensor，格式通常是[yaw, pitch, roll]（根据数据集代码）
                # 需要转换为[pitch, yaw, roll]格式以匹配euler_pred
                if len(cont_labels.shape) == 2 and cont_labels.size(1) == 3:
                    # cont_labels格式：[yaw, pitch, roll]（弧度）
                    euler_gt_rad = cont_labels.float()
                    # 转换为[pitch, yaw, roll]格式并转换为度数
                    euler_gt = torch.stack([
                        euler_gt_rad[:, 1],  # pitch
                        euler_gt_rad[:, 0],  # yaw
                        euler_gt_rad[:, 2]   # roll
                    ], dim=1) * 180 / np.pi
                else:
                    # 如果格式不对，从旋转矩阵计算
                    euler_gt = utils.compute_euler_angles_from_rotation_matrices(R_gt.cpu()) * 180 / np.pi
            else:
                # 其他情况，从旋转矩阵计算
                euler_gt = utils.compute_euler_angles_from_rotation_matrices(R_gt.cpu()) * 180 / np.pi
            
            # 欧拉角顺序：[pitch, yaw, roll] = [x, y, z]（与compute_euler_angles一致）
            p_gt_deg = euler_gt[:, 0]  # pitch (x)
            y_gt_deg = euler_gt[:, 1]  # yaw (y)
            r_gt_deg = euler_gt[:, 2]  # roll (z)
            
            p_pred_deg = euler_pred[:, 0].cpu()  # pitch
            y_pred_deg = euler_pred[:, 1].cpu()  # yaw
            r_pred_deg = euler_pred[:, 2].cpu()  # roll
            
            # 计算角度误差（考虑周期性）
            pitch_error += torch.sum(torch.min(torch.stack((
                torch.abs(p_gt_deg - p_pred_deg), 
                torch.abs(p_pred_deg + 360 - p_gt_deg), 
                torch.abs(p_pred_deg - 360 - p_gt_deg), 
                torch.abs(p_pred_deg + 180 - p_gt_deg), 
                torch.abs(p_pred_deg - 180 - p_gt_deg)
            )), 0)[0]).item()
            
            yaw_error += torch.sum(torch.min(torch.stack((
                torch.abs(y_gt_deg - y_pred_deg), 
                torch.abs(y_pred_deg + 360 - y_gt_deg), 
                torch.abs(y_pred_deg - 360 - y_gt_deg), 
                torch.abs(y_pred_deg + 180 - y_gt_deg), 
                torch.abs(y_pred_deg - 180 - y_gt_deg)
            )), 0)[0]).item()
            
            roll_error += torch.sum(torch.min(torch.stack((
                torch.abs(r_gt_deg - r_pred_deg), 
                torch.abs(r_pred_deg + 360 - r_gt_deg), 
                torch.abs(r_pred_deg - 360 - r_gt_deg), 
                torch.abs(r_pred_deg + 180 - r_gt_deg), 
                torch.abs(r_pred_deg - 180 - r_gt_deg)
            )), 0)[0]).item()
    
    avg_loss = total_loss / total_samples
    avg_yaw_error = yaw_error / total_samples
    avg_pitch_error = pitch_error / total_samples
    avg_roll_error = roll_error / total_samples
    avg_mae = (avg_yaw_error + avg_pitch_error + avg_roll_error) / 3.0
    
    model.train()
    return {
        'loss': avg_loss,
        'yaw_error': avg_yaw_error,
        'pitch_error': avg_pitch_error,
        'roll_error': avg_roll_error,
        'mae': avg_mae
    }


if __name__ == '__main__':

    args = parse_args()
    cudnn.enabled = True
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    gpu = args.gpu_id
    
    # 显式检查GPU可用性
    if gpu >= 0:
        if not torch.cuda.is_available():
            print(f'Warning: GPU {gpu} requested but CUDA is not available. Using CPU instead.')
            gpu = -1
        elif gpu >= torch.cuda.device_count():
            print(f'Warning: GPU {gpu} requested but only {torch.cuda.device_count()} GPU(s) available. Using GPU 0 instead.')
            gpu = 0
        else:
            print(f'Using GPU {gpu}')
    else:
        print('Using CPU')
    
    b_scheduler = args.scheduler

    if not os.path.exists('../../../output/snapshots'):
        os.makedirs('../../../output/snapshots')

    # summary_name = '{}_{}_bs{}'.format(
    #     'SixDRepNet', int(time.time()), args.batch_size)
    current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    summary_name = '{}_{}_bs{}'.format(
        'SixDRepNet', current_time, args.batch_size)

    if not os.path.exists('../../../output/snapshots/{}'.format(summary_name)):
        os.makedirs('../../../output/snapshots/{}'.format(summary_name))

    use_CoordConv = True

    # 创建模型
    if args.backbone == 'MobileNetV2':
        model = SixDRepNet_MobileNetV2(pretrained=True, use_CoordConv=use_CoordConv)
        backbone_name = 'MobileNetV2'
    else:
        model = SixDRepNet(backbone_name='RepVGG-B1g2',
                          backbone_file='../../../weights/RepVGG/RepVGG-B1g2-train.pth',
                          deploy=False,
                          pretrained=True)
        backbone_name = 'RepVGG-B1g2'
 
    if not args.snapshot == '':
        saved_state_dict = torch.load(args.snapshot)
        if 'model_state_dict' in saved_state_dict:
            model.load_state_dict(saved_state_dict['model_state_dict'])
        else:
            model.load_state_dict(saved_state_dict)

    print('Loading data.')

    if args.backbone == 'MobileNetV2' and use_CoordConv:
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406, 0.0, 0.0],
            std=[0.229, 0.224, 0.225, 1.0, 1.0])
        transformations = transforms.Compose([
            transforms.RandomResizedCrop(size=224, scale=(0.8, 1)),
            transforms.ToTensor(),
            utils.AddCoordChannels(),
            normalize
        ])
        val_transformations = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            utils.AddCoordChannels(),
            normalize
        ])
    else:
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
        transformations = transforms.Compose([
            transforms.RandomResizedCrop(size=224, scale=(0.8, 1)),
            transforms.ToTensor(),
            normalize
        ])
        val_transformations = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])

    # 加载完整训练集
    full_dataset = datasets.getDataset(
        args.dataset, args.data_dir, args.filename_list, transformations)

    # 准备验证集：从训练集中划分或使用独立数据集
    if args.val_dataset is not None:
        # 使用独立的验证数据集（不推荐，但保留此选项）
        print(f'Using separate validation dataset: {args.val_dataset}')
        val_dataset = datasets.getDataset(
            args.val_dataset, args.val_data_dir, args.val_filename_list, 
            val_transformations, train_mode=False)
        train_dataset = full_dataset
    else:
        # 从训练集中划分验证集（推荐方式）
        dataset_size = len(full_dataset)
        val_size = int(dataset_size * args.val_split)
        train_size = dataset_size - val_size
        
        # 使用固定随机种子保证可复现
        generator = torch.Generator().manual_seed(args.val_seed)
        train_indices, val_indices = random_split(
            list(range(dataset_size)), [train_size, val_size], generator=generator
        )
        
        print(f'Train/Val split: {train_size}/{val_size} (ratio: {args.val_split*100:.1f}%)')
        print(f'Using random seed {args.val_seed} for train/val split')
        
        # 创建训练数据集（使用训练transform，包含数据增强）
        train_dataset = torch.utils.data.Subset(full_dataset, train_indices.indices)
        
        # 创建验证数据集（使用验证transform，不包含数据增强）
        # 需要创建新的数据集实例，使用验证集的transform
        val_dataset_full = datasets.getDataset(
            args.dataset, args.data_dir, args.filename_list, 
            val_transformations)
        val_dataset = torch.utils.data.Subset(val_dataset_full, val_indices.indices)
    
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4)
    
    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2)

    if gpu >= 0:
        model.cuda(gpu)
    else:
        model.cpu()
    
    # 模型分析：在训练开始前分析模型结构（仅执行一次）
    print('='*70)
    print('Model Profiling'.center(70))
    print('='*70)
    try:
        # 获取一个样本batch用于分析
        sample_batch = next(iter(train_loader))
        sample_images = sample_batch[0]  # 获取images
        if isinstance(sample_images, torch.Tensor):
            sample_images = sample_images[:1]  # 只取第一个样本，减少计算
            if gpu >= 0:
                sample_images = sample_images.cuda(gpu)
        else:
            sample_images = sample_images[0:1] if len(sample_images) > 0 else sample_images
            if gpu >= 0 and isinstance(sample_images, torch.Tensor):
                sample_images = sample_images.cuda(gpu)
        
        # 执行模型分析
        profile_model(
            model=model,
            sample_input=sample_images,
            backbone_name=backbone_name,
            output_dir='../../../output/snapshots/{}'.format(summary_name),
            device=f'cuda:{gpu}' if gpu >= 0 else 'cpu'
        )
    except Exception as e:
        print(f'Warning: Model profiling failed: {e}')
        print('Training will continue without profiling...')
        import traceback
        traceback.print_exc()
    print('='*70)
    print('')
    
    if gpu >= 0:
        crit = GeodesicLoss().cuda(gpu) #torch.nn.MSELoss().cuda(gpu)
    else:
        crit = GeodesicLoss()  # CPU模式
    kd_criterion = GeodesicLoss().cuda(gpu) if gpu >= 0 else GeodesicLoss()
    
    # 知识蒸馏配置（默认关闭）
    use_distillation = args.use_distillation
    teacher_model = None
    if use_distillation:
        if args.backbone != 'MobileNetV2':
            print('Warning: Distillation is only designed for MobileNetV2 student. Disabling distillation.')
            use_distillation = False
        elif not (0.0 <= args.distill_alpha <= 1.0):
            print(f'Warning: distill_alpha={args.distill_alpha} is out of [0,1]. Clamping to valid range.')
            args.distill_alpha = max(0.0, min(1.0, args.distill_alpha))
        else:
            teacher_model = SixDRepNet(
                backbone_name='RepVGG-B1g2',
                backbone_file='../../../weights/RepVGG/RepVGG-B1g2-train.pth',
                deploy=False,
                pretrained=True
            )
            if args.teacher_snapshot != '':
                print(f'Loading teacher snapshot: {args.teacher_snapshot}')
                teacher_ckpt = torch.load(args.teacher_snapshot, map_location='cpu')
                if 'model_state_dict' in teacher_ckpt:
                    teacher_model.load_state_dict(teacher_ckpt['model_state_dict'])
                else:
                    teacher_model.load_state_dict(teacher_ckpt)
            if gpu >= 0:
                teacher_model = teacher_model.cuda(gpu)
            else:
                teacher_model = teacher_model.cpu()
            teacher_model.eval()
            for p in teacher_model.parameters():
                p.requires_grad = False
            print(f'Knowledge distillation enabled. Alpha={args.distill_alpha:.3f}')
    
    # 优化器开关（默认保持原始配置）
    optimizer_mode = args.optimizer_mode.lower()
    if optimizer_mode == 'improved':
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))
        print(f'Using improved optimizer: AdamW (lr={args.lr}, weight_decay={args.weight_decay})')
    else:
        optimizer = torch.optim.Adam(model.parameters(), args.lr)
        print(f'Using original optimizer: Adam (lr={args.lr})')


    # 学习率调度器配置
    scheduler = None
    if b_scheduler:
        if args.scheduler_type == 'ReduceLROnPlateau':
            # 基于验证损失的动态学习率调整
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)
            print('Using ReduceLROnPlateau scheduler (based on validation loss)')
        else:
            # MultiStepLR - 在多个epoch降低学习率
            milestones = [int(num_epochs * 0.3), int(num_epochs * 0.6), int(num_epochs * 0.8)]
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=milestones, gamma=0.5)
            print(f'Using MultiStepLR scheduler with milestones: {milestones}')
    else:
        print('No learning rate scheduler enabled. Learning rate will remain constant.')
    
    # 判断是否使用Plateau调度器（需要在scheduler创建后判断）
    use_plateau_scheduler = (b_scheduler and scheduler is not None and 
                            isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau))
    
    # 初始化训练历史记录
    training_history = {
        'train_loss': [],
        'train_gt_loss': [],
        'train_distill_loss': [],
        'val_loss': [],
        'val_mae': [],
        'val_yaw_error': [],
        'val_pitch_error': [],
        'val_roll_error': [],
        'learning_rate': [],
        'epoch': []
    }
    
    # 最佳模型跟踪
    best_val_loss = float('inf')
    best_val_mae = float('inf')
    best_epoch = -1

    print('Starting training.')
    for epoch in range(num_epochs):
        model.train()
        loss_sum = .0
        gt_loss_sum = .0
        distill_loss_sum = .0
        iter = 0
        
        for i, (images, gt_mat, _, _) in enumerate(train_loader):
            iter += 1
            images = torch.Tensor(images)
            if gpu >= 0:
                images = images.cuda(gpu)
                gt_mat = gt_mat.cuda(gpu)

            # Forward pass（模型可输出6D或旋转矩阵）
            pred_out = model(images)
            pred_mat = output_to_rotation_matrix(pred_out)

            # 监督损失
            gt_loss = crit(gt_mat, pred_mat)
            
            # 蒸馏损失（teacher输出旋转矩阵）
            distill_loss = torch.tensor(0.0, device=pred_mat.device)
            if use_distillation and teacher_model is not None:
                with torch.no_grad():
                    # 若student输入为CoordConv 5通道，teacher仅使用RGB 3通道
                    teacher_images = images[:, :3, :, :] if images.size(1) > 3 else images
                    teacher_out = teacher_model(teacher_images)
                    teacher_mat = output_to_rotation_matrix(teacher_out)
                distill_loss = kd_criterion(teacher_mat, pred_mat)
                loss = (1.0 - args.distill_alpha) * gt_loss + args.distill_alpha * distill_loss
            else:
                loss = gt_loss

            optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪（如果启用）
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            
            optimizer.step()

            loss_sum += loss.item()
            gt_loss_sum += gt_loss.item()
            if use_distillation and teacher_model is not None:
                distill_loss_sum += distill_loss.item()

            if (i+1) % 100 == 0:
                if use_distillation and teacher_model is not None:
                    print('Epoch [%d/%d], Iter [%d/%d] Loss: %.6f (GT: %.6f, KD: %.6f)' % (
                        epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.item(),
                        gt_loss.item(), distill_loss.item()))
                else:
                    print('Epoch [%d/%d], Iter [%d/%d] Loss: %.6f' % (
                        epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.item()))
        
        # 计算平均训练损失
        avg_train_loss = loss_sum / iter
        avg_gt_loss = gt_loss_sum / iter
        avg_distill_loss = distill_loss_sum / iter if (use_distillation and teacher_model is not None) else 0.0
        
        # 验证集评估
        print('Validating...')
        val_metrics = validate(model, val_loader, crit, gpu)
        
        # 学习率调度（根据scheduler类型）
        if b_scheduler and scheduler is not None:
            if use_plateau_scheduler:
                # ReduceLROnPlateau需要传入验证损失
                scheduler.step(val_metrics['loss'])
                current_lr = optimizer.param_groups[0]['lr']
            else:
                # MultiStepLR在epoch结束时更新
                scheduler.step()
                if hasattr(scheduler, 'get_last_lr'):
                    current_lr = scheduler.get_last_lr()[0]
                else:
                    current_lr = optimizer.param_groups[0]['lr']
        else:
            current_lr = optimizer.param_groups[0]['lr']
        
        # 记录训练历史
        training_history['epoch'].append(epoch + 1)
        training_history['train_loss'].append(avg_train_loss)
        training_history['train_gt_loss'].append(avg_gt_loss)
        training_history['train_distill_loss'].append(avg_distill_loss)
        training_history['val_loss'].append(val_metrics['loss'])
        training_history['val_mae'].append(val_metrics['mae'])
        training_history['val_yaw_error'].append(val_metrics['yaw_error'])
        training_history['val_pitch_error'].append(val_metrics['pitch_error'])
        training_history['val_roll_error'].append(val_metrics['roll_error'])
        training_history['learning_rate'].append(current_lr)
        
        # 打印epoch总结
        print('Epoch [%d/%d] Summary:' % (epoch+1, num_epochs))
        print('  Train Loss: %.6f' % avg_train_loss)
        if use_distillation and teacher_model is not None:
            print('  Train GT Loss: %.6f, Train KD Loss: %.6f' % (avg_gt_loss, avg_distill_loss))
        print('  Val Loss: %.6f' % val_metrics['loss'])
        print('  Val MAE: %.4f (Yaw: %.4f, Pitch: %.4f, Roll: %.4f)' % (
            val_metrics['mae'], val_metrics['yaw_error'], 
            val_metrics['pitch_error'], val_metrics['roll_error']))
        print('  Learning Rate: %.8f' % current_lr)
        
        # 如果角度误差异常大，给出警告
        if val_metrics['yaw_error'] > 30 or val_metrics['pitch_error'] > 30:
            print('  WARNING: Large angle errors detected! This may indicate:')
            print('    - Data preprocessing issues (angle unit conversion)')
            print('    - Model capacity limitations')
            print('    - Training instability')
            print('    - Consider checking data distribution and model architecture')
        
        # 保存训练历史（每个epoch都保存）
        history_path = '../../../output/snapshots/{}/training_history.json'.format(summary_name)
        with open(history_path, 'w') as f:
            json.dump(training_history, f, indent=2)
        
        # 保存最佳模型（基于验证损失）
        is_best = False
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_val_mae = val_metrics['mae']
            best_epoch = epoch + 1
            is_best = True
        
        # 保存检查点（包含完整训练状态）
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
            'train_loss': avg_train_loss,
            'val_loss': val_metrics['loss'],
            'val_mae': val_metrics['mae'],
            'val_yaw_error': val_metrics['yaw_error'],
            'val_pitch_error': val_metrics['pitch_error'],
            'val_roll_error': val_metrics['roll_error'],
            'best_val_loss': best_val_loss,
            'best_val_mae': best_val_mae,
            'best_epoch': best_epoch,
            'learning_rate': current_lr,
            'batch_size': batch_size,
            'num_epochs': num_epochs,
            'backbone': backbone_name,
            'optimizer_mode': optimizer_mode,
            'weight_decay': args.weight_decay if optimizer_mode == 'improved' else 0.0,
            'use_distillation': use_distillation,
            'distill_alpha': args.distill_alpha if use_distillation else 0.0,
            'teacher_snapshot': args.teacher_snapshot if use_distillation else '',
            'model_config': {
                'backbone': backbone_name,
                'deploy': False,
            },
            'training_history': training_history
        }
        
        checkpoint_path = '../../../output/snapshots/{}/{}checkpoint_epoch_{}.tar'.format(
            summary_name, args.output_string + '_' if args.output_string else '', epoch + 1)
        # 不进行每一步epoch的保存
        # torch.save(checkpoint, checkpoint_path)
        
        # 保存最佳模型
        if is_best:
            best_model_path = '../../../output/snapshots/{}/{}best_model.tar'.format(
                summary_name, args.output_string + '_' if args.output_string else '')
            torch.save(checkpoint, best_model_path)
            print('  *** New best model saved! (Val Loss: %.6f, Val MAE: %.4f) ***' % (
                best_val_loss, best_val_mae))
        
        print('')
    
    print('Training completed!')
    print('Best model: Epoch %d, Val Loss: %.6f, Val MAE: %.4f' % (
        best_epoch, best_val_loss, best_val_mae))
