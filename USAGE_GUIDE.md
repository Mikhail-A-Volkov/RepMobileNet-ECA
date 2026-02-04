# 使用指南

## 1. 训练模型

### 1.1 使用RepVGG作为backbone

```bash
python sixdrepnet/train.py \
    --gpu 0 \
    --num_epochs 80 \
    --batch_size 80 \
    --lr 0.0001 \
    --dataset Pose_300W_LP \
    --data_dir ../../../datasets/300W_LP \
    --filename_list ../../../datasets/300W_LP/files.txt \
    --val_dataset AFLW2000 \
    --val_data_dir ../../../datasets/AFLW2000 \
    --val_filename_list ../../../datasets/AFLW2000/files.txt \
    --backbone RepVGG \
    --scheduler True
```

### 1.2 使用MobileNetV2作为backbone

```bash
python sixdrepnet/train.py \
    --gpu 0 \
    --num_epochs 80 \
    --batch_size 80 \
    --lr 0.0001 \
    --dataset Pose_300W_LP \
    --data_dir ../../../datasets/300W_LP \
    --filename_list ../../../datasets/300W_LP/files.txt \
    --val_dataset AFLW2000 \
    --val_data_dir ../../../datasets/AFLW2000 \
    --val_filename_list ../../../datasets/AFLW2000/files.txt \
    --backbone MobileNetV2 \
    --scheduler True
```

### 1.3 训练输出说明

训练过程中会生成以下文件：

- `output/snapshots/{summary_name}/training_history.json`: 训练历史记录（每个epoch）
- `output/snapshots/{summary_name}/checkpoint_epoch_{N}.tar`: 每个epoch的完整检查点
- `output/snapshots/{summary_name}/best_model.tar`: 最佳模型（基于验证损失）

每个检查点包含：
- `epoch`: 训练轮次
- `model_state_dict`: 模型权重
- `optimizer_state_dict`: 优化器状态
- `scheduler_state_dict`: 学习率调度器状态
- `train_loss`: 训练损失
- `val_loss`: 验证损失
- `val_mae`: 验证MAE
- `val_yaw_error`, `val_pitch_error`, `val_roll_error`: 各角度误差
- `best_val_loss`, `best_val_mae`, `best_epoch`: 最佳结果记录
- `learning_rate`: 当前学习率
- `training_history`: 完整训练历史

## 2. 测试模型

### 2.1 在AFLW2000上测试RepVGG模型

```bash
python sixdrepnet/test.py \
    --gpu 0 \
    --batch_size 64 \
    --dataset AFLW2000 \
    --data_dir datasets/AFLW2000 \
    --filename_list datasets/AFLW2000/files.txt \
    --snapshot output/snapshots/your_model/best_model.tar \
    --backbone RepVGG \
    --show_viz False
```

### 2.2 在AFLW2000上测试MobileNetV2模型

```bash
python sixdrepnet/test.py \
    --gpu 0 \
    --batch_size 64 \
    --dataset AFLW2000 \
    --data_dir datasets/AFLW2000 \
    --filename_list datasets/AFLW2000/files.txt \
    --snapshot output/snapshots/your_model/best_model.tar \
    --backbone MobileNetV2 \
    --show_viz False
```

### 2.3 在BIWI上测试

```bash
python sixdrepnet/test.py \
    --gpu 0 \
    --batch_size 64 \
    --dataset BIWI \
    --data_dir datasets/BIWI \
    --filename_list datasets/BIWI/BIWI_test.npz \
    --snapshot output/snapshots/your_model/best_model.tar \
    --backbone MobileNetV2 \
    --show_viz False
```

## 3. 模型参数量对比和可视化

### 3.1 对比RepVGG和MobileNetV2的参数量

```bash
python compare_models.py
```

这会生成：
- `output/model_info_SixDRepNet_(RepVGG-B1g2).json`: RepVGG模型详细信息
- `output/model_info_SixDRepNet_(MobileNetV2).json`: MobileNetV2模型详细信息
- `output/model_comparison.png`: 参数量对比图
- `output/layer_comparison.png`: 各层参数量对比图

### 3.2 可视化训练历史

```bash
python visualize_training.py \
    --history output/snapshots/your_model/training_history.json \
    --output_dir output
```

这会生成：
- `output/training_curves.png`: 训练曲线（损失、MAE、角度误差、学习率）
- `output/error_comparison.png`: 角度误差对比图

## 4. 权重文件说明

### 4.1 当前保存格式

训练脚本现在会保存以下信息：

**每个epoch的检查点** (`checkpoint_epoch_{N}.tar`):
- 模型权重
- 优化器状态
- 学习率调度器状态
- 当前epoch的训练和验证指标
- 最佳结果记录
- 完整训练历史

**最佳模型** (`best_model.tar`):
- 包含验证损失最低时的所有信息

**训练历史** (`training_history.json`):
- JSON格式的训练历史，便于后续分析和可视化

### 4.2 用于学术论文的信息

保存的检查点包含以下可用于论文分析的信息：

1. **训练指标**:
   - 每个epoch的训练损失
   - 每个epoch的验证损失
   - 学习率变化

2. **评估指标**:
   - 验证MAE（平均绝对误差）
   - Yaw、Pitch、Roll角度误差
   - 最佳epoch和对应的所有指标

3. **模型配置**:
   - Backbone类型
   - 训练超参数（batch size, learning rate等）

4. **训练历史**:
   - 完整的训练曲线数据，可用于绘制图表

## 5. 注意事项

1. **验证集**: 训练脚本现在会在每个epoch后在验证集上评估，这可能会增加训练时间。

2. **最佳模型**: 最佳模型基于验证损失选择，如果验证损失没有改善，不会覆盖之前的最佳模型。

3. **训练历史**: `training_history.json`文件会在每个epoch后更新，如果训练中断，可以从最新的检查点恢复。

4. **内存使用**: 保存完整训练历史可能会占用一些内存，但通常可以忽略不计。

5. **测试脚本**: 测试脚本现在支持通过`--backbone`参数选择backbone类型，确保与训练时使用的backbone一致。





