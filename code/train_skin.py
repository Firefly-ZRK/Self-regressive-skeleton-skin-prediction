import jittor as jt
import numpy as np
import os
import argparse
import time
import random

from jittor import nn
from jittor import optim

from dataset.dataset import get_dataloader, transform, HandAwareSampler
from dataset.format import id_to_name
from dataset.sampler import SamplerMix
from models.skin import create_model

from dataset.exporter import Exporter

# Set Jittor flags
jt.flags.use_cuda = 1

# 骨骼等效性损失计算函数
def bone_centric_loss(outputs, targets, criterion, freeze_mask=None):
    """
    计算骨骼为中心的损失，确保所有骨骼得到平等的训练
    
    Args:
        outputs: 模型输出的蒙皮权重，形状为 [B, N, J]
        targets: 真实的蒙皮权重，形状为 [B, N, J]
        criterion: 损失函数
        freeze_mask: 指示哪些骨骼需要冻结的掩码，形状为 [J]，值为1表示冻结
        
    Returns:
        loss: 总损失
        loss_per_bone: 每个骨骼的损失，形状为 [J]
    """
    batch_size, num_vertices, num_joints = outputs.shape
    
    # 向量化计算所有骨骼的损失 [B, N, J]
    bone_losses = criterion(outputs, targets)  # [B, N, J]
    
    # 沿着batch和vertex维度取平均，得到每个骨骼的损失 [J]
    loss_per_bone = jt.mean(bone_losses, dims=(0, 1))  # [J]
    
    # 处理冻结mask
    if freeze_mask is not None:
        # 将冻结骨骼的损失设为0，保持与原始逻辑一致
        loss_per_bone = loss_per_bone * (1 - freeze_mask)
        
        # 计算活跃（非冻结）骨骼数量
        active_bones = jt.sum(1 - freeze_mask)
        if active_bones > 0:
            loss = jt.sum(loss_per_bone) / active_bones
        else:
            loss = jt.sum(loss_per_bone) / num_joints
    else:
        loss = jt.mean(loss_per_bone)
    
    return loss, loss_per_bone

# 特征保留损失，用于确保模型不丢失重要特征
def feature_retention_loss(point_features, bone_features, weight=0.01):
    """
    计算特征保留损失，确保特征的多样性和丰富性
    
    Args:
        point_features: 点特征 [B, N, F]
        bone_features: 骨骼特征 [B, J, F]
        weight: 损失权重
        
    Returns:
        loss: 特征保留损失
    """
    # 计算点特征的协方差矩阵
    point_mean = jt.mean(point_features, dim=1, keepdims=True)  # [B, 1, F]
    point_centered = point_features - point_mean  # [B, N, F]
    point_cov = jt.matmul(point_centered.transpose(1, 2), point_centered)  # [B, F, F]
    
    # 计算骨骼特征的协方差矩阵
    bone_mean = jt.mean(bone_features, dim=1, keepdims=True)  # [B, 1, F]
    bone_centered = bone_features - bone_mean  # [B, J, F]
    bone_cov = jt.matmul(bone_centered.transpose(1, 2), bone_centered)  # [B, F, F]
    
    # 计算协方差矩阵的对角元素之和（即特征方差之和）
    point_var = jt.diagonal(point_cov, dim1=1, dim2=2).sum(1)  # [B]
    bone_var = jt.diagonal(bone_cov, dim1=1, dim2=2).sum(1)  # [B]
    
    # 鼓励特征方差较大，以保留更多信息
    loss = -weight * (jt.mean(point_var) + jt.mean(bone_var))
    
    return loss

def train(args):
    """
    Main training function
    
    Args:
        args: Command line arguments
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Set up logging
    log_file = os.path.join(args.output_dir, 'training_log.txt')
    
    def log_message(message):
        """Helper function to log messages to file and print to console"""
        with open(log_file, 'a') as f:
            f.write(f"{message}\n")
        print(message)
    
    # Log training parameters
    log_message(f"Starting training with parameters: {args}")
    
    # Create model
    model = create_model(
        model_name=args.model_name,
        model_type=args.model_type,
        feat_dim=args.feat_dim
    )
    
    log_message(f"Created model: {args.model_type} with feature dimension {args.feat_dim}")
    if args.model_type == 'cross_attention':
        log_message("Using cross-attention model with enhanced feature retention")
    
    # Load pre-trained model if specified
    if args.pretrained_model:
        log_message(f"Loading pretrained model from {args.pretrained_model}")
        model.load(args.pretrained_model)
    
    # Load checkpoint if specified
    if args.resume_from_checkpoint:
        log_message(f"Loading checkpoint from {args.resume_from_checkpoint}")
        model.load(args.resume_from_checkpoint)
        log_message(f"Resuming training from epoch {args.start_epoch + 1}")
    
    # Create optimizer
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")
    
    # Create loss function
    criterion_mse = nn.MSELoss()  # 不使用reduction参数
    criterion_l1 = nn.L1Loss()    # 不使用reduction参数
    
    # 创建自定义损失函数来计算不进行reduction的损失
    def mse_loss_no_reduction(pred, target):
        return (pred - target) ** 2

    def l1_loss_no_reduction(pred, target):
        return jt.abs(pred - target)
    
    # Create dataloaders - 使用手部感知采样器
    train_loader = get_dataloader(
        data_root=args.data_root,
        data_list=args.train_data_list,
        train=True,
        batch_size=args.batch_size,
        shuffle=True,
        sampler=HandAwareSampler(num_samples=1024, vertex_samples=512, hand_boost_ratio=3.0),
        transform=transform,
        use_pose_augmentation=True,
        pose_augmentation_ratio=0.3,
    )
    
    if args.val_data_list:
        val_loader = get_dataloader(
            data_root=args.data_root,
            data_list=args.val_data_list,
            train=False,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=HandAwareSampler(num_samples=1024, vertex_samples=512, hand_boost_ratio=3.0),
            transform=transform,
            use_pose_augmentation=False,  # 验证时不使用数据增强
        )
    else:
        val_loader = None
    
    # 获取骨骼数量
    num_joints = 52  # 固定为52个关节
    
    # Training loop
    best_loss = 99999999
    for epoch in range(args.start_epoch, args.epochs):
        # 添加温和的学习率调度策略 - 解决学习率衰减严重问题
        if epoch in [int(args.epochs * 0.5), int(args.epochs * 0.7), int(args.epochs * 0.9)]:
            new_lr = optimizer.lr * 0.65  # 使用0.5而不是0.1，避免衰减过于严重
            log_message(f"Reducing learning rate to {new_lr}")
            optimizer.lr = new_lr
        
        # Training phase
        model.train()
        train_loss_mse = 0.0
        train_loss_l1 = 0.0
        train_loss_per_bone = jt.zeros(num_joints)
        
        # 自适应调整冻结比例 - 手部优先策略
        if args.use_bone_equivalence:
            freeze_ratio = max(0.05, 0.2 - epoch * 0.15 / args.epochs)  # 随着训练进行，逐渐减少冻结比例
            # 手部关节冻结概率更低
            hand_freeze_ratio = freeze_ratio * 0.2  # 手部关节冻结概率为普通关节的30%
            log_message(f"Epoch {epoch+1}: Using freeze ratio {freeze_ratio:.3f}, hand freeze ratio {hand_freeze_ratio:.3f}")
        
        start_time = time.time()
        for batch_idx, data in enumerate(train_loader):
            # Get data and labels
            vertices, joints, skin = data['vertices'], data['joints'], data['skin']

            vertices: jt.Var
            joints: jt.Var
            skin: jt.Var
            
            # 骨骼等效性训练策略：随机冻结一部分骨骼，手部关节优先训练
            if args.use_bone_equivalence:
                # 分别处理主要关节和手部关节的冻结
                freeze_mask = jt.zeros(num_joints)
                freeze_mask_np = freeze_mask.numpy()
                
                # 主要关节（前22个）使用标准冻结比例
                main_joint_count = int(22 * freeze_ratio)
                if main_joint_count > 0:
                    main_freeze_indices = random.sample(range(22), main_joint_count)
                    freeze_mask_np[main_freeze_indices] = 1
                
                # 手部关节（22-51）使用更低的冻结比例
                hand_joint_count = int(30 * hand_freeze_ratio)  # 30个手指关节
                if hand_joint_count > 0:
                    hand_freeze_indices = random.sample(range(22, 52), hand_joint_count)
                    freeze_mask_np[hand_freeze_indices] = 1
                
                freeze_mask = jt.Var(freeze_mask_np)
            else:
                freeze_mask = None
            
            # 前向传播
            if args.model_type == 'cross_attention':
                # 交叉注意力模型 - 获取中间特征用于特征保留损失
                outputs = model(vertices, joints)
                
                # 如果需要特征保留，提取中间特征
                if hasattr(model, 'point_encoder') and hasattr(model, 'bone_encoder') and args.use_feature_retention:
                    with jt.no_grad():
                        point_features = model.point_encoder(vertices)
                        bone_features = model.bone_encoder(joints)
            else:
                # 标准或增强模型 - 不关心中间特征
                outputs = model(vertices, joints)
            
            # 骨骼等效性损失计算
            if args.use_bone_equivalence:
                loss_mse, mse_per_bone = bone_centric_loss(outputs, skin, mse_loss_no_reduction, freeze_mask)
                loss_l1, l1_per_bone = bone_centric_loss(outputs, skin, l1_loss_no_reduction, freeze_mask)
                
                # 累积每个骨骼的损失，用于监控
                train_loss_per_bone += l1_per_bone.detach()
            else:
                # 传统损失计算
                loss_mse = criterion_mse(outputs, skin)
                loss_l1 = criterion_l1(outputs, skin)
            
            # 总损失
            loss = loss_mse + loss_l1
            
            # 添加特征保留损失（仅适用于交叉注意力模型）
            if args.model_type == 'cross_attention' and args.use_feature_retention:
                feat_loss = feature_retention_loss(point_features, bone_features)
                loss += feat_loss
                
                # 如果损失值很大，打印警告
                if jt.abs(feat_loss).item() > 1.0:
                    log_message(f"Warning: Large feature retention loss: {feat_loss.item():.4f}")
            
            # Backward pass and optimize
            optimizer.zero_grad()
            optimizer.backward(loss)
            optimizer.step()
            
            # Calculate statistics
            train_loss_mse += loss_mse.item()
            train_loss_l1 += loss_l1.item()
            
            # Print progress
            if (batch_idx + 1) % args.print_freq == 0 or (batch_idx + 1) == len(train_loader):
                log_message(f"Epoch [{epoch+1}/{args.epochs}] Batch [{batch_idx+1}/{len(train_loader)}] "
                           f"Loss mse: {loss_mse.item():.4f} Loss l1: {loss_l1.item():.4f}")
        
        # Calculate epoch statistics
        train_loss_mse /= len(train_loader)
        train_loss_l1 /= len(train_loader)
        train_loss_per_bone /= len(train_loader)
        epoch_time = time.time() - start_time
        
        log_message(f"Epoch [{epoch+1}/{args.epochs}] "
                   f"Train Loss mse: {train_loss_mse:.4f} "
                   f"Train Loss l1: {train_loss_l1:.4f} "
                   f"Time: {epoch_time:.2f}s "
                   f"LR: {optimizer.lr:.6f}")
        
        # 打印每个骨骼的平均损失
        if args.use_bone_equivalence and (epoch + 1) % 5 == 0:
            log_message("Per-bone loss:")
            for j in range(num_joints):
                if j in id_to_name:
                    name = id_to_name[j]
                else:
                    name = f"joint_{j}"
                log_message(f"  {name}: {train_loss_per_bone[j].item():.4f}")

        # Validation phase
        if val_loader is not None and (epoch + 1) % args.val_freq == 0:
            model.eval()
            val_loss_mse = 0.0
            val_loss_l1 = 0.0
            val_loss_per_bone = jt.zeros(num_joints)
            
            show_id = np.random.randint(0, len(val_loader))
            for batch_idx, data in enumerate(val_loader):
                # Get data and labels
                vertices, joints, skin = data['vertices'], data['joints'], data['skin']
                
                # Forward pass
                outputs = model(vertices, joints)
                
                # 骨骼等效性损失计算 (验证时不冻结骨骼)
                if args.use_bone_equivalence:
                    loss_mse, mse_per_bone = bone_centric_loss(outputs, skin, mse_loss_no_reduction)
                    loss_l1, l1_per_bone = bone_centric_loss(outputs, skin, l1_loss_no_reduction)
                    
                    # 累积每个骨骼的验证损失
                    val_loss_per_bone += l1_per_bone.detach()
                else:
                    # 传统损失计算
                    loss_mse = criterion_mse(outputs, skin)
                    loss_l1 = criterion_l1(outputs, skin)
                
                # export render results(disabled for performance)
                # if batch_idx == show_id:
                #     exporter = Exporter()
                #     for i in id_to_name:
                #         name = id_to_name[i]
                #         # export every joint's corresponding skinning
                #         exporter._render_skin(path=f"tmp/skin/epoch_{epoch}/{name}_ref.png",vertices=vertices.numpy()[0], skin=skin.numpy()[0, :, i], joint=joints[0, i])
                #         exporter._render_skin(path=f"tmp/skin/epoch_{epoch}/{name}_pred.png",vertices=vertices.numpy()[0], skin=outputs.numpy()[0, :, i], joint=joints[0, i])

                val_loss_mse += loss_mse.item()
                val_loss_l1 += loss_l1.item()
            
            # Calculate validation statistics
            val_loss_mse /= len(val_loader)
            val_loss_l1 /= len(val_loader)
            val_loss_per_bone /= len(val_loader)
            
            log_message(f"Validation Loss: mse: {val_loss_mse:.4f} l1: {val_loss_l1:.4f}")
            
            # 打印每个骨骼的验证损失
            if args.use_bone_equivalence and (epoch + 1) % 5 == 0:
                log_message("Per-bone validation loss:")
                for j in range(num_joints):
                    if j in id_to_name:
                        name = id_to_name[j]
                    else:
                        name = f"joint_{j}"
                    log_message(f"  {name}: {val_loss_per_bone[j].item():.4f}")
            
            # Save best model
            if val_loss_l1 < best_loss:
                best_loss = val_loss_l1
                model_path = os.path.join(args.output_dir, 'best_model.pkl')
                model.save(model_path)
                log_message(f"Saved best model with l1 loss {best_loss:.4f} to {model_path}")
        
        # Save checkpoint
        if (epoch + 1) % args.save_freq == 0:
            checkpoint_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pkl')
            model.save(checkpoint_path)
            log_message(f"Saved checkpoint to {checkpoint_path}")
    
    # Save final model
    final_model_path = os.path.join(args.output_dir, 'final_model.pkl')
    model.save(final_model_path)
    log_message(f"Training completed. Saved final model to {final_model_path}")
    
    return model, best_loss

def main():
    """Parse arguments and start training"""
    parser = argparse.ArgumentParser(description='Train a point cloud model')
    
    # Dataset parameters
    parser.add_argument('--train_data_list', type=str, required=True,
                        help='Path to the training data list file')
    parser.add_argument('--val_data_list', type=str, default='',
                        help='Path to the validation data list file')
    parser.add_argument('--data_root', type=str, default='data',
                        help='Root directory for the data files')
    
    # Model parameters
    parser.add_argument('--model_name', type=str, default='pct',
                        choices=['pct', 'pct2', 'custom_pct', 'skeleton'],
                        help='Model architecture to use')
    parser.add_argument('--model_type', type=str, default='cross_attention',
                        choices=['cross_attention'],
                        help='Model type for skin model')
    parser.add_argument('--feat_dim', type=int, default=256,
                        help='Feature dimension for model')
    parser.add_argument('--pretrained_model', type=str, default='',
                        help='Path to pretrained model')
    parser.add_argument('--resume_from_checkpoint', type=str, default='',
                        help='Path to checkpoint to resume from')
    parser.add_argument('--start_epoch', type=int, default=0,
                        help='Starting epoch number (used with resume_from_checkpoint)')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Number of training epochs')
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['sgd', 'adam'],
                        help='Optimizer to use')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay (L2 penalty)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Momentum for SGD optimizer')
    parser.add_argument('--use_bone_equivalence', action='store_true',
                        help='Use bone equivalence training strategy')
    parser.add_argument('--use_feature_retention', action='store_true',
                        help='Use feature retention loss (for cross_attention model)')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='output/skin',
                        help='Directory to save output files')
    parser.add_argument('--print_freq', type=int, default=10,
                        help='Print frequency')
    parser.add_argument('--save_freq', type=int, default=10,
                        help='Save frequency')
    parser.add_argument('--val_freq', type=int, default=1,
                        help='Validation frequency')
    parser.add_argument('--disable_preload', action='store_true',
                        help='Disable data preloading to save memory')
    
    args = parser.parse_args()
    
    # Start training
    train(args)

def seed_all(seed):
    jt.set_global_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

if __name__ == '__main__':
    seed_all(123)
    main()