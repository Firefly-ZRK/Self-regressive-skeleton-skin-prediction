import jittor as jt
import numpy as np
import os
import argparse
import time
import random

from jittor import nn
from jittor import optim

from dataset.dataset import get_dataloader, transform, HandAwareSampler
from dataset.sampler import SamplerMix
from dataset.exporter import Exporter
from models.skeleton import create_model, PARENT_IDS, JOINT_NAMES

from models.metrics import J2J

# Set Jittor flags
jt.flags.use_cuda = 1

def train(args):
    """
    Main training function for skeleton model
    
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
        output_channels=args.num_joints * 3  # Each joint has 3 coordinates
    )
    
    log_message(f"Created model: {args.model_type}")
    if args.model_type == 'autoregressive':
        log_message(f"Using autoregressive model with LSTM for {args.num_joints} joints")
    
    # Use hand-aware sampler for better handling of 52 joints
    sampler = HandAwareSampler(num_samples=1024, vertex_samples=512, hand_boost_ratio=3.0)
    
    # Load pre-trained model if specified
    if args.pretrained_model:
        log_message(f"Loading pretrained model from {args.pretrained_model}")
        model.load(args.pretrained_model)
    
    # Create optimizer
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")
    
    # Create loss function
    criterion = nn.MSELoss()
    
    # Create dataloaders
    train_loader = get_dataloader(
        data_root=args.data_root,
        data_list=args.train_data_list,
        train=True,
        batch_size=args.batch_size,
        shuffle=True,
        sampler=sampler,
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
            sampler=sampler,
            transform=transform,
            use_pose_augmentation=False,  # No data augmentation during validation
        )
    else:
        val_loader = None
    
    # Training loop
    best_loss = 99999999
    for epoch in range(args.epochs):
        # Modify learning rate schedule strategy - gentler decay
        if epoch in [int(args.epochs * 0.5), int(args.epochs * 0.7), int(args.epochs * 0.9)]:
            new_lr = optimizer.lr * 0.65  # Changed from 0.1 to 0.5, reduce decay degree
            log_message(f"Reducing learning rate to {new_lr}")
            optimizer.lr = new_lr

        # Training phase
        model.train()
        train_loss = 0.0
        
        # Calculate current epoch's teacher forcing probability
        if args.model_type == 'autoregressive':
            # Use higher teacher forcing probability at the beginning, gradually decrease as training progresses
            # For LSTM, we can use slightly lower initial teacher_forcing_ratio as LSTM has stronger memory capability
            teacher_forcing_ratio = max(0.0, 0.95 - (epoch / (args.epochs * 0.8)))  # Extend decay time, keep minimum 0.2
            log_message(f"Epoch [{epoch+1}/{args.epochs}] Teacher forcing ratio: {teacher_forcing_ratio:.4f}")
        else:
            teacher_forcing_ratio = 0.0
        
        start_time = time.time()
        for batch_idx, data in enumerate(train_loader):
            # Get data and labels
            vertices, joints = data['vertices'], data['joints']
            
            # Handle input data shape
            if args.model_type == 'autoregressive':
                # Autoregressive model needs vertices shape [B, N, 3], joints shape [B, J, 3]
                vertices_input = vertices  # [B, N, 3]
                joints_input = joints.reshape(vertices.shape[0], -1, 3)  # [B, J, 3]
                
                # Provide real joints for teacher forcing during forward pass
                outputs = model(vertices_input, joints_input, teacher_forcing_ratio)
                
                # Calculate basic loss
                base_loss = criterion(outputs, joints_input)
                loss = base_loss
                
                # Enhanced per-joint loss terms - solve hand weight insufficiency problem
                if args.use_joint_losses:
                    joint_losses = []
                    for j in range(joints_input.shape[1]):
                        joint_loss = criterion(outputs[:, j, :], joints_input[:, j, :])
                        joint_losses.append(joint_loss)
                    
                    # Expand key joints including finger joints - adjust weights
                    main_key_joints = [0, 3, 6, 7, 10, 11, 14, 18]  # Main joints
                    hand_key_joints = [9, 13, 22, 25, 28, 31, 34, 37, 40, 43, 46, 49]  # Wrist and finger root joints
                    
                    # Give standard weight to main joints
                    for j in main_key_joints:
                        loss += 0.2 * joint_losses[j]
                    
                    # Give slightly lower weight to hand key joints
                    for j in hand_key_joints:
                        if j < len(joint_losses):
                            loss += 0.15 * joint_losses[j]  # Slightly lower weight than main joints
            else:
                # Old model needs vertices shape [B, 3, N]
                vertices_input = vertices.permute(0, 2, 1)  # [B, 3, N]
                
                # Standard forward pass
                outputs = model(vertices_input)
                joints_flat = joints.reshape(outputs.shape[0], -1)  # [B, J*3]
                loss = criterion(outputs, joints_flat)
            
            # Backward pass and optimize - Add gradient clipping optimization
            optimizer.zero_grad()
            optimizer.backward(loss)
            
            # Gentler gradient clipping - avoid gradient vanishing
            optimizer.clip_grad_norm(1.0)  # Reduced from 2.0 to 1.0, allow larger gradient updates
            optimizer.step()
            
            # Calculate statistics
            train_loss += loss.item()
            
            # Print progress
            if (batch_idx + 1) % args.print_freq == 0 or (batch_idx + 1) == len(train_loader):
                log_message(f"Epoch [{epoch+1}/{args.epochs}] Batch [{batch_idx+1}/{len(train_loader)}] "
                           f"Loss: {loss.item():.4f}")
        
        # Calculate epoch statistics
        train_loss /= len(train_loader)
        epoch_time = time.time() - start_time
        
        log_message(f"Epoch [{epoch+1}/{args.epochs}] "
                   f"Train Loss: {train_loss:.4f} "
                   f"Time: {epoch_time:.2f}s "
                   f"LR: {optimizer.lr:.6f}")

        # Validation phase
        if val_loader is not None and (epoch + 1) % args.val_freq == 0:
            model.eval()
            val_loss = 0.0
            J2J_loss = 0.0
            
            # Joint prediction accuracy tracking for autoregressive model
            if args.model_type == 'autoregressive':
                joint_errors = np.zeros(args.num_joints)
                # Separately track errors for main joints and hand joints
                main_joint_errors = np.zeros(22)
                hand_joint_errors = np.zeros(30)
            
            show_id = np.random.randint(0, len(val_loader))
            for batch_idx, data in enumerate(val_loader):
                # Get data and labels
                vertices, joints = data['vertices'], data['joints']
                
                # Handle validation data
                if args.model_type == 'autoregressive':
                    # Autoregressive model, no teacher forcing
                    vertices_input = vertices  # [B, N, 3]
                    joints_input = joints.reshape(vertices.shape[0], -1, 3)  # [B, J, 3]
                    
                    # No teacher forcing during inference
                    outputs = model(vertices_input)
                    loss = criterion(outputs, joints_input)
                    
                    # Track error for each joint
                    for j in range(args.num_joints):
                        joint_error = jt.mean(jt.sqrt(jt.sum((outputs[:, j, :] - joints_input[:, j, :]) ** 2, dim=1)))
                        joint_errors[j] += joint_error.item() / len(val_loader)
                        
                        # Separately track main joints and hand joints
                        if j < 22:
                            main_joint_errors[j] += joint_error.item() / len(val_loader)
                        else:
                            hand_joint_errors[j - 22] += joint_error.item() / len(val_loader)
                    
                    # Ensure output is [B, J, 3] shape
                    outputs_reshaped = outputs  # [B, J, 3]
                    joints_reshaped = joints_input  # [B, J, 3]
                else:
                    # Old model processing logic
                    joints_flat = joints.reshape(joints.shape[0], -1)
                    
                    # Reshape input if needed
                    if vertices.ndim == 3:  # [B, N, 3]
                        vertices_input = vertices.permute(0, 2, 1)  # [B, 3, N]
                    
                    # Forward pass
                    outputs = model(vertices_input)
                    loss = criterion(outputs, joints_flat)
                    
                    # Reshape output to [B, J, 3] to calculate J2J loss
                    outputs_reshaped = outputs.reshape(-1, args.num_joints, 3)
                    joints_reshaped = joints.reshape(-1, args.num_joints, 3)
                
                # Export render results (disabled for performance)
                # if batch_idx == show_id:
                #     exporter = Exporter()
                #     # export every joint's corresponding skinning
                #     from dataset.format import parents
                #     exporter._render_skeleton(path=f"tmp/skeleton/epoch_{epoch}/skeleton_ref.png", joints=joints_reshaped[0].numpy(), parents=parents)
                #     exporter._render_skeleton(path=f"tmp/skeleton/epoch_{epoch}/skeleton_pred.png", joints=outputs_reshaped[0].numpy(), parents=parents)
                #     exporter._render_pc(path=f"tmp/skeleton/epoch_{epoch}/vertices.png", vertices=vertices[0].numpy())

                val_loss += loss.item()
                for i in range(outputs_reshaped.shape[0]):
                    J2J_loss += J2J(outputs_reshaped[i], joints_reshaped[i]).item() / outputs_reshaped.shape[0]
            
            # Calculate validation statistics
            val_loss /= len(val_loader)
            J2J_loss /= len(val_loader)
            
            log_message(f"Validation Loss: {val_loss:.4f} J2J Loss: {J2J_loss:.4f}")
            
            # Detailed joint error analysis - separately show main joints and hand joints
            if args.model_type == 'autoregressive' and (epoch + 1) % 5 == 0:
                log_message("=== Main Joint Errors (first 22) ===")
                for j in range(22):
                    joint_name = JOINT_NAMES[j] if j < len(JOINT_NAMES) else f"joint_{j}"
                    log_message(f"  {joint_name}: {joint_errors[j]:.4f}")
                
                log_message("=== Hand Joint Errors (last 30) ===")
                avg_hand_error = np.mean(hand_joint_errors)
                log_message(f"  Average hand error: {avg_hand_error:.4f}")
                
                # Separately show left and right hand errors
                left_hand_error = np.mean(hand_joint_errors[:15])  # Left hand 15 joints
                right_hand_error = np.mean(hand_joint_errors[15:])  # Right hand 15 joints
                log_message(f"  Left hand average error: {left_hand_error:.4f}")
                log_message(f"  Right hand average error: {right_hand_error:.4f}")
                
                # Show finger joint with largest error
                worst_hand_joint_idx = np.argmax(hand_joint_errors) + 22
                if worst_hand_joint_idx < len(JOINT_NAMES):
                    log_message(f"  Worst hand joint: {JOINT_NAMES[worst_hand_joint_idx]} ({hand_joint_errors[worst_hand_joint_idx - 22]:.4f})")
            
            # Save best model
            if J2J_loss < best_loss:
                best_loss = J2J_loss
                model_path = os.path.join(args.output_dir, 'best_model.pkl')
                model.save(model_path)
                log_message(f"Saved best model with loss {best_loss:.4f} to {model_path}")
        
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
    parser = argparse.ArgumentParser(description='Train a skeleton joint prediction model')
    
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
    parser.add_argument('--model_type', type=str, default='autoregressive',
                        choices=['autoregressive'],
                        help='Model type for skeleton model')
    parser.add_argument('--num_joints', type=int, default=52,
                        help='Number of joints to predict')
    parser.add_argument('--pretrained_model', type=str, default='',
                        help='Path to pretrained model')
    parser.add_argument('--use_joint_losses', action='store_true',
                        help='Use additional per-joint losses for autoregressive model')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
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
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='output/skeleton',
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