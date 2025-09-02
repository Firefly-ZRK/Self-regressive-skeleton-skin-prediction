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

# Bone-centric loss calculation function
def bone_centric_loss(outputs, targets, criterion, freeze_mask=None):
    """
    Calculate bone-centric loss to ensure all bones get equal training
    
    Args:
        outputs: Model output skinning weights, shape [B, N, J]
        targets: Ground truth skinning weights, shape [B, N, J]
        criterion: Loss function
        freeze_mask: Mask indicating which bones to freeze, shape [J], value 1 means freeze
        
    Returns:
        loss: Total loss
        loss_per_bone: Loss per bone, shape [J]
    """
    batch_size, num_vertices, num_joints = outputs.shape
    
    # Vectorized computation of loss for all bones [B, N, J]
    bone_losses = criterion(outputs, targets)  # [B, N, J]
    
    # Average along batch and vertex dimensions to get loss per bone [J]
    loss_per_bone = jt.mean(bone_losses, dims=(0, 1))  # [J]
    
    # Handle freeze mask
    if freeze_mask is not None:
        # Set loss of frozen bones to 0, consistent with original logic
        loss_per_bone = loss_per_bone * (1 - freeze_mask)
        
        # Calculate number of active (non-frozen) bones
        active_bones = jt.sum(1 - freeze_mask)
        if active_bones > 0:
            loss = jt.sum(loss_per_bone) / active_bones
        else:
            loss = jt.sum(loss_per_bone) / num_joints
    else:
        loss = jt.mean(loss_per_bone)
    
    return loss, loss_per_bone

# Feature retention loss to ensure model doesn't lose important features
def feature_retention_loss(point_features, bone_features, weight=0.01):
    """
    Calculate feature retention loss to ensure feature diversity and richness
    
    Args:
        point_features: Point features [B, N, F]
        bone_features: Bone features [B, J, F]
        weight: Loss weight
        
    Returns:
        loss: Feature retention loss
    """
    # Calculate covariance matrix of point features
    point_mean = jt.mean(point_features, dim=1, keepdims=True)  # [B, 1, F]
    point_centered = point_features - point_mean  # [B, N, F]
    point_cov = jt.matmul(point_centered.transpose(1, 2), point_centered)  # [B, F, F]
    
    # Calculate covariance matrix of bone features
    bone_mean = jt.mean(bone_features, dim=1, keepdims=True)  # [B, 1, F]
    bone_centered = bone_features - bone_mean  # [B, J, F]
    bone_cov = jt.matmul(bone_centered.transpose(1, 2), bone_centered)  # [B, F, F]
    
    # Calculate sum of diagonal elements of covariance matrices (i.e., sum of feature variances)
    point_var = jt.diagonal(point_cov, dim1=1, dim2=2).sum(1)  # [B]
    bone_var = jt.diagonal(bone_cov, dim1=1, dim2=2).sum(1)  # [B]
    
    # Encourage larger feature variance to retain more information
    loss = -weight * (jt.mean(point_var) + jt.mean(bone_var))
    
    return loss

def train(args):
    """
    Main training function for skin model
    
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
    criterion_mse = nn.MSELoss()  # Don't use reduction parameter
    criterion_l1 = nn.L1Loss()    # Don't use reduction parameter
    
    # Create custom loss functions for non-reduced loss computation
    def mse_loss_no_reduction(pred, target):
        return (pred - target) ** 2

    def l1_loss_no_reduction(pred, target):
        return jt.abs(pred - target)
    
    # Create dataloaders - Use hand-aware sampler
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
            use_pose_augmentation=False,  # No data augmentation during validation
        )
    else:
        val_loader = None
    
    # Get number of joints
    num_joints = 52  # Fixed to 52 joints
    
    # Training loop
    best_loss = 99999999
    for epoch in range(args.start_epoch, args.epochs):
        # Add gentle learning rate scheduling strategy - solve severe learning rate decay problem
        if epoch in [int(args.epochs * 0.5), int(args.epochs * 0.7), int(args.epochs * 0.9)]:
            new_lr = optimizer.lr * 0.65  # Use 0.5 instead of 0.1 to avoid excessive decay
            log_message(f"Reducing learning rate to {new_lr}")
            optimizer.lr = new_lr
        
        # Training phase
        model.train()
        train_loss_mse = 0.0
        train_loss_l1 = 0.0
        train_loss_per_bone = jt.zeros(num_joints)
        
        # Adaptive adjustment of freeze ratio - hand-priority strategy
        if args.use_bone_equivalence:
            freeze_ratio = max(0.05, 0.2 - epoch * 0.15 / args.epochs)  # Gradually reduce freeze ratio as training progresses
            # Lower freeze probability for hand joints
            hand_freeze_ratio = freeze_ratio * 0.2  # Hand joint freeze probability is 30% of normal joints
            log_message(f"Epoch {epoch+1}: Using freeze ratio {freeze_ratio:.3f}, hand freeze ratio {hand_freeze_ratio:.3f}")
        
        start_time = time.time()
        for batch_idx, data in enumerate(train_loader):
            # Get data and labels
            vertices, joints, skin = data['vertices'], data['joints'], data['skin']

            vertices: jt.Var
            joints: jt.Var
            skin: jt.Var
            
            # Bone equivalence training strategy: randomly freeze some bones, prioritize hand joint training
            if args.use_bone_equivalence:
                # Separately handle freezing of main joints and hand joints
                freeze_mask = jt.zeros(num_joints)
                freeze_mask_np = freeze_mask.numpy()
                
                # Main joints (first 22) use standard freeze ratio
                main_joint_count = int(22 * freeze_ratio)
                if main_joint_count > 0:
                    main_freeze_indices = random.sample(range(22), main_joint_count)
                    freeze_mask_np[main_freeze_indices] = 1
                
                # Hand joints (22-51) use lower freeze ratio
                hand_joint_count = int(30 * hand_freeze_ratio)  # 30 finger joints
                if hand_joint_count > 0:
                    hand_freeze_indices = random.sample(range(22, 52), hand_joint_count)
                    freeze_mask_np[hand_freeze_indices] = 1
                
                freeze_mask = jt.Var(freeze_mask_np)
            else:
                freeze_mask = None
            
            # Forward pass
            if args.model_type == 'cross_attention':
                # Cross-attention model - get intermediate features for feature retention loss
                outputs = model(vertices, joints)
                
                # Extract intermediate features for feature retention if needed
                if hasattr(model, 'point_encoder') and hasattr(model, 'bone_encoder') and args.use_feature_retention:
                    with jt.no_grad():
                        point_features = model.point_encoder(vertices)
                        bone_features = model.bone_encoder(joints)
            else:
                # Standard or enhanced model - don't care about intermediate features
                outputs = model(vertices, joints)
            
            # Bone equivalence loss calculation
            if args.use_bone_equivalence:
                loss_mse, mse_per_bone = bone_centric_loss(outputs, skin, mse_loss_no_reduction, freeze_mask)
                loss_l1, l1_per_bone = bone_centric_loss(outputs, skin, l1_loss_no_reduction, freeze_mask)
                
                # Accumulate loss per bone for monitoring
                train_loss_per_bone += l1_per_bone.detach()
            else:
                # Traditional loss calculation
                loss_mse = criterion_mse(outputs, skin)
                loss_l1 = criterion_l1(outputs, skin)
            
            # Total loss
            loss = loss_mse + loss_l1
            
            # Add feature retention loss (only for cross-attention model)
            if args.model_type == 'cross_attention' and args.use_feature_retention:
                feat_loss = feature_retention_loss(point_features, bone_features)
                loss += feat_loss
                
                # Print warning if loss value is very large
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
        
        # Print average loss per bone
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
                
                # Bone equivalence loss calculation (no bone freezing during validation)
                if args.use_bone_equivalence:
                    loss_mse, mse_per_bone = bone_centric_loss(outputs, skin, mse_loss_no_reduction)
                    loss_l1, l1_per_bone = bone_centric_loss(outputs, skin, l1_loss_no_reduction)
                    
                    # Accumulate validation loss per bone
                    val_loss_per_bone += l1_per_bone.detach()
                else:
                    # Traditional loss calculation
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
            
            # Print validation loss per bone
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
    parser = argparse.ArgumentParser(description='Train a skin weight prediction model')
    
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
    """Set random seeds for reproducibility"""
    jt.set_global_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

if __name__ == '__main__':
    seed_all(123)
    main()