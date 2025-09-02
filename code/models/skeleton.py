import jittor as jt
from jittor import nn
from jittor import init
from jittor.contrib import concat
import numpy as np
import sys

# Import the PCT model components
from PCT.networks.cls.pct import Point_Transformer, Point_Transformer2, Point_Transformer_Last, SA_Layer, Local_op, sample_and_group
from PCT.networks.cls.pct import SkeletonTransformer

# Skeleton joint names and parent relationships definition (52 joints)
JOINT_NAMES = [
    'hips',           # 0
    'spine',          # 1
    'chest',          # 2
    'upper_chest',    # 3
    'neck',           # 4
    'head',           # 5
    'l_shoulder',     # 6
    'l_upper_arm',    # 7
    'l_lower_arm',    # 8
    'l_hand',         # 9
    'r_shoulder',     # 10
    'r_upper_arm',    # 11
    'r_lower_arm',    # 12
    'r_hand',         # 13
    'l_upper_leg',    # 14
    'l_lower_leg',    # 15
    'l_foot',         # 16
    'l_toe_base',     # 17
    'r_upper_leg',    # 18
    'r_lower_leg',    # 19
    'r_foot',         # 20
    'r_toe_base',     # 21
    # Left hand finger joints
    'l_hand_thumb_1',   # 22
    'l_hand_thumb_2',   # 23
    'l_hand_thumb_3',   # 24
    'l_hand_index_1',   # 25
    'l_hand_index_2',   # 26
    'l_hand_index_3',   # 27
    'l_hand_middle_1',  # 28
    'l_hand_middle_2',  # 29
    'l_hand_middle_3',  # 30
    'l_hand_ring_1',    # 31
    'l_hand_ring_2',    # 32
    'l_hand_ring_3',    # 33
    'l_hand_pinky_1',   # 34
    'l_hand_pinky_2',   # 35
    'l_hand_pinky_3',   # 36
    # Right hand finger joints
    'r_hand_thumb_1',   # 37
    'r_hand_thumb_2',   # 38
    'r_hand_thumb_3',   # 39
    'r_hand_index_1',   # 40
    'r_hand_index_2',   # 41
    'r_hand_index_3',   # 42
    'r_hand_middle_1',  # 43
    'r_hand_middle_2',  # 44
    'r_hand_middle_3',  # 45
    'r_hand_ring_1',    # 46
    'r_hand_ring_2',    # 47
    'r_hand_ring_3',    # 48
    'r_hand_pinky_1',   # 49
    'r_hand_pinky_2',   # 50
    'r_hand_pinky_3',   # 51
]

# Define parent node index for each joint (52 joints)
PARENT_IDS = [
    -1,     # 0: 'hips' (root node, no parent)
    0,      # 1: 'spine' (parent is hips)
    1,      # 2: 'chest' (parent is spine)
    2,      # 3: 'upper_chest' (parent is chest)
    3,      # 4: 'neck' (parent is upper_chest)
    4,      # 5: 'head' (parent is neck)
    3,      # 6: 'l_shoulder' (parent is upper_chest)
    6,      # 7: 'l_upper_arm' (parent is l_shoulder)
    7,      # 8: 'l_lower_arm' (parent is l_upper_arm)
    8,      # 9: 'l_hand' (parent is l_lower_arm)
    3,      # 10: 'r_shoulder' (parent is upper_chest)
    10,     # 11: 'r_upper_arm' (parent is r_shoulder)
    11,     # 12: 'r_lower_arm' (parent is r_upper_arm)
    12,     # 13: 'r_hand' (parent is r_lower_arm)
    0,      # 14: 'l_upper_leg' (parent is hips)
    14,     # 15: 'l_lower_leg' (parent is l_upper_leg)
    15,     # 16: 'l_foot' (parent is l_lower_leg)
    16,     # 17: 'l_toe_base' (parent is l_foot)
    0,      # 18: 'r_upper_leg' (parent is hips)
    18,     # 19: 'r_lower_leg' (parent is r_upper_leg)
    19,     # 20: 'r_foot' (parent is r_lower_leg)
    20,     # 21: 'r_toe_base' (parent is r_foot)
    # Left hand finger joint parents
    9,      # 22: 'l_hand_thumb_1' (parent is l_hand)
    22,     # 23: 'l_hand_thumb_2' (parent is l_hand_thumb_1)
    23,     # 24: 'l_hand_thumb_3' (parent is l_hand_thumb_2)
    9,      # 25: 'l_hand_index_1' (parent is l_hand)
    25,     # 26: 'l_hand_index_2' (parent is l_hand_index_1)
    26,     # 27: 'l_hand_index_3' (parent is l_hand_index_2)
    9,      # 28: 'l_hand_middle_1' (parent is l_hand)
    28,     # 29: 'l_hand_middle_2' (parent is l_hand_middle_1)
    29,     # 30: 'l_hand_middle_3' (parent is l_hand_middle_2)
    9,      # 31: 'l_hand_ring_1' (parent is l_hand)
    31,     # 32: 'l_hand_ring_2' (parent is l_hand_ring_1)
    32,     # 33: 'l_hand_ring_3' (parent is l_hand_ring_2)
    9,      # 34: 'l_hand_pinky_1' (parent is l_hand)
    34,     # 35: 'l_hand_pinky_2' (parent is l_hand_pinky_1)
    35,     # 36: 'l_hand_pinky_3' (parent is l_hand_pinky_2)
    # Right hand finger joint parents
    13,     # 37: 'r_hand_thumb_1' (parent is r_hand)
    37,     # 38: 'r_hand_thumb_2' (parent is r_hand_thumb_1)
    38,     # 39: 'r_hand_thumb_3' (parent is r_hand_thumb_2)
    13,     # 40: 'r_hand_index_1' (parent is r_hand)
    40,     # 41: 'r_hand_index_2' (parent is r_hand_index_1)
    41,     # 42: 'r_hand_index_3' (parent is r_hand_index_2)
    13,     # 43: 'r_hand_middle_1' (parent is r_hand)
    43,     # 44: 'r_hand_middle_2' (parent is r_hand_middle_1)
    44,     # 45: 'r_hand_middle_3' (parent is r_hand_middle_2)
    13,     # 46: 'r_hand_ring_1' (parent is r_hand)
    46,     # 47: 'r_hand_ring_2' (parent is r_hand_ring_1)
    47,     # 48: 'r_hand_ring_3' (parent is r_hand_ring_2)
    13,     # 49: 'r_hand_pinky_1' (parent is r_hand)
    49,     # 50: 'r_hand_pinky_2' (parent is r_hand_pinky_1)
    50,     # 51: 'r_hand_pinky_3' (parent is r_hand_pinky_2)
]

# Define hand joint indices
LEFT_HAND_JOINTS = list(range(22, 37))   # Left hand joints 22-36
RIGHT_HAND_JOINTS = list(range(37, 52))  # Right hand joints 37-51
HAND_JOINTS = LEFT_HAND_JOINTS + RIGHT_HAND_JOINTS

class JointPredictor(nn.Module):
    """Joint predictor for predicting individual joint positions"""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 512),  # Increase intermediate layer dimension to improve feature retention
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),  # Add Dropout to enhance generalization
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 3)  # Output 3D coordinates
        )
    
    def execute(self, x: jt.Var) -> jt.Var:
        return self.mlp(x)

class HandRefinementModule(nn.Module):
    """Hand refinement module specialized for predicting finger details"""
    def __init__(self, feat_dim: int = 256, hidden_dim: int = 256):
        super().__init__()
        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim
        
        # Hand feature extractor
        self.hand_encoder = nn.Sequential(
            nn.Linear(3, 64),  # Hand wrist position encoding
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.ReLU()
        )
        
        # Hand refinement LSTM
        self.hand_lstm = nn.LSTMCell(128 + feat_dim, hidden_dim)
        
        # Finger joint predictor
        self.finger_predictor = JointPredictor(hidden_dim)
        
        # Initializers
        self.hidden_init = nn.Sequential(
            nn.Linear(feat_dim + 128, hidden_dim),
            nn.Tanh()
        )
        
        self.cell_init = nn.Sequential(
            nn.Linear(feat_dim + 128, hidden_dim),
            nn.Tanh()
        )
    
    def execute(self, hand_position: jt.Var, global_feat: jt.Var, num_fingers: int = 15):
        """
        Predict finger joint positions
        
        Args:
            hand_position: Hand wrist position [B, 3]
            global_feat: Global features [B, feat_dim]
            num_fingers: Number of finger joints (15 joints per hand)
        
        Returns:
            finger_joints: Finger joint positions [B, num_fingers, 3]
        """
        batch_size = hand_position.shape[0]
        
        # Encode hand wrist position
        hand_feat = self.hand_encoder(hand_position)  # [B, 128]
        
        # Concatenate features
        combined_feat = concat([hand_feat, global_feat], dim=1)  # [B, 128 + feat_dim]
        
        # Initialize LSTM state
        hidden = self.hidden_init(combined_feat)  # [B, hidden_dim]
        cell = self.cell_init(combined_feat)  # [B, hidden_dim]
        
        # Autoregressive generation of finger joints
        finger_joints = []
        for i in range(num_fingers):
            # LSTM input
            lstm_input = combined_feat  # Can add position encoding if needed
            hidden, cell = self.hand_lstm(lstm_input, (hidden, cell))
            
            # Predict joint offset (relative to wrist)
            joint_offset = self.finger_predictor(hidden)
            joint_position = hand_position + joint_offset
            finger_joints.append(joint_position)
        
        return jt.stack(finger_joints, dim=1)  # [B, num_fingers, 3]

class AutoregressiveSkeletonModel(nn.Module):
    """Autoregressive skeleton generation model supporting 52 joints with hand refinement"""
    def __init__(self, feat_dim: int = 256, hidden_dim: int = 512, num_joints: int = 52):
        super().__init__()
        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim
        self.num_joints = num_joints
        
        # Shape encoder - Use Point Transformer to extract global geometric features from point cloud
        self.shape_encoder = Point_Transformer(output_channels=feat_dim)
        
        # Feature retention mechanism - Add feature enhancement layer
        self.feature_enhancer = nn.Sequential(
            nn.Linear(feat_dim, feat_dim * 2),
            nn.LayerNorm(feat_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feat_dim * 2, feat_dim),
            nn.LayerNorm(feat_dim)
        )
        
        # Condition encoder - Encode current joint and parent coordinates to features
        self.condition_encoder = nn.Sequential(
            nn.Linear(6, 128),  # Input is parent coordinates(3) and current joint relative coordinates(3)
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, hidden_dim)
        )
        
        # Main LSTM - For predicting main skeleton (first 22 joints)
        self.main_lstm = nn.LSTMCell(hidden_dim + feat_dim, hidden_dim)
        
        # Main joint predictor
        self.main_joint_predictor = JointPredictor(hidden_dim)
        
        # Hand refinement modules
        self.left_hand_refiner = HandRefinementModule(feat_dim, hidden_dim)
        self.right_hand_refiner = HandRefinementModule(feat_dim, hidden_dim)
        
        # Hidden state initializers
        self.hidden_init = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.Tanh()
        )
        
        self.cell_init = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.Tanh()
        )
        
        # Parent relationships
        self.parent_ids = PARENT_IDS
    
    def execute(self, vertices: jt.Var, ground_truth_joints: jt.Var = None, teacher_forcing_ratio: float = 1.0):
        """
        Args:
            vertices: Point cloud vertices, shape [B, N, 3] or [B, 3, N]
            ground_truth_joints: Real joint coordinates for teacher forcing, shape [B, J, 3]
            teacher_forcing_ratio: Teacher forcing probability, range [0, 1]
        Returns:
            predicted_joints: Predicted joint coordinates, shape [B, 52, 3]
        """
        # Ensure vertices shape is [B, 3, N]
        if vertices.ndim == 3 and vertices.shape[1] == 3:
            vertices_transposed = vertices  # [B, 3, N]
        else:
            vertices_transposed = vertices.permute(0, 2, 1)  # [B, 3, N]
        
        batch_size = vertices.shape[0]
        
        # Extract global shape features
        global_feat = self.shape_encoder(vertices_transposed)  # [B, feat_dim]
        
        # Enhance feature representation
        global_feat = self.feature_enhancer(global_feat) + global_feat  # Residual connection
        
        # Stage 1: Predict main 22 joints (torso, limbs)
        main_joints = self._predict_main_joints(
            global_feat, ground_truth_joints, teacher_forcing_ratio
        )
        
        # Stage 2: Hand refinement - Predict finger joints
        left_hand_joints = self.left_hand_refiner(
            main_joints[:, 9, :],  # Left wrist position
            global_feat,
            num_fingers=15  # 15 finger joints for left hand
        )
        
        right_hand_joints = self.right_hand_refiner(
            main_joints[:, 13, :],  # Right wrist position
            global_feat,
            num_fingers=15  # 15 finger joints for right hand
        )
        
        # Concatenate all joints
        all_joints = jt.concat([
            main_joints,          # [B, 22, 3] - Main joints
            left_hand_joints,     # [B, 15, 3] - Left hand joints
            right_hand_joints     # [B, 15, 3] - Right hand joints
        ], dim=1)  # [B, 52, 3]
        
        return all_joints
    
    def _predict_main_joints(self, global_feat: jt.Var, ground_truth_joints: jt.Var = None, teacher_forcing_ratio: float = 1.0):
        """
        Predict the main 22 joints (torso, limbs)
        """
        batch_size = global_feat.shape[0]
        
        # Initialize LSTM hidden state and cell state
        hidden = self.hidden_init(global_feat)  # [B, hidden_dim]
        cell = self.cell_init(global_feat)  # [B, hidden_dim]
        
        # Initialize root node parent coordinates as zero vector
        root_parent_coord = jt.zeros((batch_size, 3), dtype=global_feat.dtype)
        
        # Initialize output joint array
        predicted_joints = []
        
        # Autoregressive generation of first 22 joints
        for j in range(22):
            if j == 0:
                # Root node has no parent, use zero vector as parent coordinate
                parent_coord = root_parent_coord
                relative_coord = jt.zeros((batch_size, 3), dtype=global_feat.dtype)
            else:
                parent_id = self.parent_ids[j]
                
                # Get parent coordinates
                if ground_truth_joints is not None and jt.rand(1).item() < teacher_forcing_ratio:
                    # Teacher forcing: Use real parent coordinates
                    parent_coord = ground_truth_joints[:, parent_id, :]
                else:
                    # Use previously predicted parent coordinates
                    parent_coord = predicted_joints[parent_id]
                
                # If ground truth joints provided, calculate relative coordinates
                if ground_truth_joints is not None and jt.rand(1).item() < teacher_forcing_ratio:
                    relative_coord = ground_truth_joints[:, j, :] - parent_coord
                else:
                    # Otherwise set to zero, let model predict relative coordinates
                    relative_coord = jt.zeros((batch_size, 3), dtype=global_feat.dtype)
            
            # Encode condition information (parent coordinates and relative coordinates)
            condition = self.condition_encoder(concat([parent_coord, relative_coord], dim=1))
            
            # Concatenate condition and global features, feed to LSTM cell
            lstm_input = concat([condition, global_feat], dim=1)
            hidden, cell = self.main_lstm(lstm_input, (hidden, cell))
            
            # Predict joint coordinates
            if j == 0:
                # Root node predicts absolute coordinates directly
                joint_coord = self.main_joint_predictor(hidden)
            else:
                # For non-root nodes, predict offset relative to parent
                joint_offset = self.main_joint_predictor(hidden)
                joint_coord = parent_coord + joint_offset
            
            predicted_joints.append(joint_coord)
        
        # Stack predicted joints into tensor [B, 22, 3]
        return jt.stack(predicted_joints, dim=1)

# Factory function to create models - Only keep autoregressive
def create_model(model_name='pct', model_type='autoregressive', output_channels=156, **kwargs):
    """
    Create model - Only supports autoregressive type
    
    Args:
        model_name: Model name, only supports 'pct'
        model_type: Model type, only supports 'autoregressive'
        output_channels: Output channels, should be 156 (52*3)
        
    Returns:
        model: Created model
    """
    if model_name == "pct":
        if model_type == 'autoregressive':
            num_joints = output_channels // 3
            if num_joints != 52:
                print(f"Warning: Expected 52 joints, got {num_joints}. Forcing to 52.")
                num_joints = 52
            return AutoregressiveSkeletonModel(feat_dim=256, hidden_dim=512, num_joints=num_joints)
        else:
            raise ValueError(f"Unsupported model_type: {model_type}. Only 'autoregressive' is supported.")
    else:
        raise ValueError(f"Unsupported model_name: {model_name}. Only 'pct' is supported.")
