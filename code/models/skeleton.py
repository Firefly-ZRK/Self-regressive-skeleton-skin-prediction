import jittor as jt
from jittor import nn
from jittor import init
from jittor.contrib import concat
import numpy as np
import sys

# Import the PCT model components
from PCT.networks.cls.pct import Point_Transformer, Point_Transformer2, Point_Transformer_Last, SA_Layer, Local_op, sample_and_group
from PCT.networks.cls.pct import SkeletonTransformer

# 骨骼关节及其父节点关系定义 (52个关节)
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
    # 左手手指关节
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
    # 右手手指关节
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

# 定义每个关节的父节点索引 (52个关节)
PARENT_IDS = [
    -1,     # 0: 'hips' (根节点，没有父节点)
    0,      # 1: 'spine' (父节点是hips)
    1,      # 2: 'chest' (父节点是spine)
    2,      # 3: 'upper_chest' (父节点是chest)
    3,      # 4: 'neck' (父节点是upper_chest)
    4,      # 5: 'head' (父节点是neck)
    3,      # 6: 'l_shoulder' (父节点是upper_chest)
    6,      # 7: 'l_upper_arm' (父节点是l_shoulder)
    7,      # 8: 'l_lower_arm' (父节点是l_upper_arm)
    8,      # 9: 'l_hand' (父节点是l_lower_arm)
    3,      # 10: 'r_shoulder' (父节点是upper_chest)
    10,     # 11: 'r_upper_arm' (父节点是r_shoulder)
    11,     # 12: 'r_lower_arm' (父节点是r_upper_arm)
    12,     # 13: 'r_hand' (父节点是r_lower_arm)
    0,      # 14: 'l_upper_leg' (父节点是hips)
    14,     # 15: 'l_lower_leg' (父节点是l_upper_leg)
    15,     # 16: 'l_foot' (父节点是l_lower_leg)
    16,     # 17: 'l_toe_base' (父节点是l_foot)
    0,      # 18: 'r_upper_leg' (父节点是hips)
    18,     # 19: 'r_lower_leg' (父节点是r_upper_leg)
    19,     # 20: 'r_foot' (父节点是r_lower_leg)
    20,     # 21: 'r_toe_base' (父节点是r_foot)
    # 左手手指关节的父节点
    9,      # 22: 'l_hand_thumb_1' (父节点是l_hand)
    22,     # 23: 'l_hand_thumb_2' (父节点是l_hand_thumb_1)
    23,     # 24: 'l_hand_thumb_3' (父节点是l_hand_thumb_2)
    9,      # 25: 'l_hand_index_1' (父节点是l_hand)
    25,     # 26: 'l_hand_index_2' (父节点是l_hand_index_1)
    26,     # 27: 'l_hand_index_3' (父节点是l_hand_index_2)
    9,      # 28: 'l_hand_middle_1' (父节点是l_hand)
    28,     # 29: 'l_hand_middle_2' (父节点是l_hand_middle_1)
    29,     # 30: 'l_hand_middle_3' (父节点是l_hand_middle_2)
    9,      # 31: 'l_hand_ring_1' (父节点是l_hand)
    31,     # 32: 'l_hand_ring_2' (父节点是l_hand_ring_1)
    32,     # 33: 'l_hand_ring_3' (父节点是l_hand_ring_2)
    9,      # 34: 'l_hand_pinky_1' (父节点是l_hand)
    34,     # 35: 'l_hand_pinky_2' (父节点是l_hand_pinky_1)
    35,     # 36: 'l_hand_pinky_3' (父节点是l_hand_pinky_2)
    # 右手手指关节的父节点
    13,     # 37: 'r_hand_thumb_1' (父节点是r_hand)
    37,     # 38: 'r_hand_thumb_2' (父节点是r_hand_thumb_1)
    38,     # 39: 'r_hand_thumb_3' (父节点是r_hand_thumb_2)
    13,     # 40: 'r_hand_index_1' (父节点是r_hand)
    40,     # 41: 'r_hand_index_2' (父节点是r_hand_index_1)
    41,     # 42: 'r_hand_index_3' (父节点是r_hand_index_2)
    13,     # 43: 'r_hand_middle_1' (父节点是r_hand)
    43,     # 44: 'r_hand_middle_2' (父节点是r_hand_middle_1)
    44,     # 45: 'r_hand_middle_3' (父节点是r_hand_middle_2)
    13,     # 46: 'r_hand_ring_1' (父节点是r_hand)
    46,     # 47: 'r_hand_ring_2' (父节点是r_hand_ring_1)
    47,     # 48: 'r_hand_ring_3' (父节点是r_hand_ring_2)
    13,     # 49: 'r_hand_pinky_1' (父节点是r_hand)
    49,     # 50: 'r_hand_pinky_2' (父节点是r_hand_pinky_1)
    50,     # 51: 'r_hand_pinky_3' (父节点是r_hand_pinky_2)
]

# 定义手部关节索引
LEFT_HAND_JOINTS = list(range(22, 37))   # 左手关节 22-36
RIGHT_HAND_JOINTS = list(range(37, 52))  # 右手关节 37-51
HAND_JOINTS = LEFT_HAND_JOINTS + RIGHT_HAND_JOINTS

class JointPredictor(nn.Module):
    """关节点预测器，用于预测单个关节的位置"""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 512),  # 增加中间层维度，提高特征保留能力
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),  # 添加Dropout以增强泛化能力
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 3)  # 输出3D坐标
        )
    
    def execute(self, x: jt.Var) -> jt.Var:
        return self.mlp(x)

class HandRefinementModule(nn.Module):
    """手部细化模块，专门用于预测手指细节"""
    def __init__(self, feat_dim: int = 256, hidden_dim: int = 256):
        super().__init__()
        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim
        
        # 手部特征提取器
        self.hand_encoder = nn.Sequential(
            nn.Linear(3, 64),  # 手腕位置编码
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.ReLU()
        )
        
        # 手指细化LSTM
        self.hand_lstm = nn.LSTMCell(128 + feat_dim, hidden_dim)
        
        # 手指关节预测器
        self.finger_predictor = JointPredictor(hidden_dim)
        
        # 初始化器
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
        预测手指关节位置
        
        Args:
            hand_position: 手腕位置 [B, 3]
            global_feat: 全局特征 [B, feat_dim]
            num_fingers: 手指关节数量（每只手15个关节）
        
        Returns:
            finger_joints: 手指关节位置 [B, num_fingers, 3]
        """
        batch_size = hand_position.shape[0]
        
        # 编码手腕位置
        hand_feat = self.hand_encoder(hand_position)  # [B, 128]
        
        # 拼接特征
        combined_feat = concat([hand_feat, global_feat], dim=1)  # [B, 128 + feat_dim]
        
        # 初始化LSTM状态
        hidden = self.hidden_init(combined_feat)  # [B, hidden_dim]
        cell = self.cell_init(combined_feat)  # [B, hidden_dim]
        
        # 自回归生成手指关节
        finger_joints = []
        for i in range(num_fingers):
            # LSTM输入
            lstm_input = combined_feat  # 可以根据需要添加位置编码
            hidden, cell = self.hand_lstm(lstm_input, (hidden, cell))
            
            # 预测关节偏移（相对于手腕）
            joint_offset = self.finger_predictor(hidden)
            joint_position = hand_position + joint_offset
            finger_joints.append(joint_position)
        
        return jt.stack(finger_joints, dim=1)  # [B, num_fingers, 3]

class AutoregressiveSkeletonModel(nn.Module):
    """自回归骨架生成模型，支持52个关节，包含手部细化"""
    def __init__(self, feat_dim: int = 256, hidden_dim: int = 512, num_joints: int = 52):
        super().__init__()
        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim
        self.num_joints = num_joints
        
        # 形状编码器 - 使用Point Transformer提取点云的全局几何特征
        self.shape_encoder = Point_Transformer(output_channels=feat_dim)
        
        # 引入特征保留机制 - 增加一个特征增强层
        self.feature_enhancer = nn.Sequential(
            nn.Linear(feat_dim, feat_dim * 2),
            nn.LayerNorm(feat_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feat_dim * 2, feat_dim),
            nn.LayerNorm(feat_dim)
        )
        
        # 条件编码器 - 将当前关节和父节点坐标编码为特征
        self.condition_encoder = nn.Sequential(
            nn.Linear(6, 128),  # 输入是父节点坐标(3)和当前关节的相对坐标(3)
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, hidden_dim)
        )
        
        # 主干LSTM - 用于预测主要骨骼（前22个关节）
        self.main_lstm = nn.LSTMCell(hidden_dim + feat_dim, hidden_dim)
        
        # 主干关节预测器
        self.main_joint_predictor = JointPredictor(hidden_dim)
        
        # 手部细化模块
        self.left_hand_refiner = HandRefinementModule(feat_dim, hidden_dim)
        self.right_hand_refiner = HandRefinementModule(feat_dim, hidden_dim)
        
        # 隐藏状态初始化器
        self.hidden_init = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.Tanh()
        )
        
        self.cell_init = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.Tanh()
        )
        
        # 父节点关系
        self.parent_ids = PARENT_IDS
    
    def execute(self, vertices: jt.Var, ground_truth_joints: jt.Var = None, teacher_forcing_ratio: float = 1.0):
        """
        Args:
            vertices: 点云顶点，形状为[B, N, 3]或[B, 3, N]
            ground_truth_joints: 用于teacher forcing的真实关节坐标，形状为[B, J, 3]
            teacher_forcing_ratio: teacher forcing的概率，范围[0, 1]
        Returns:
            predicted_joints: 预测的关节坐标，形状为[B, 52, 3]
        """
        # 确保vertices的形状为[B, 3, N]
        if vertices.ndim == 3 and vertices.shape[1] == 3:
            vertices_transposed = vertices  # [B, 3, N]
        else:
            vertices_transposed = vertices.permute(0, 2, 1)  # [B, 3, N]
        
        batch_size = vertices.shape[0]
        
        # 提取全局形状特征
        global_feat = self.shape_encoder(vertices_transposed)  # [B, feat_dim]
        
        # 增强特征表达能力
        global_feat = self.feature_enhancer(global_feat) + global_feat  # 残差连接
        
        # 第一阶段：预测主要22个关节（躯干、四肢）
        main_joints = self._predict_main_joints(
            global_feat, ground_truth_joints, teacher_forcing_ratio
        )
        
        # 第二阶段：手部细化 - 预测手指关节
        left_hand_joints = self.left_hand_refiner(
            main_joints[:, 9, :],  # 左手腕位置
            global_feat,
            num_fingers=15  # 左手15个手指关节
        )
        
        right_hand_joints = self.right_hand_refiner(
            main_joints[:, 13, :],  # 右手腕位置
            global_feat,
            num_fingers=15  # 右手15个手指关节
        )
        
        # 拼接所有关节
        all_joints = jt.concat([
            main_joints,          # [B, 22, 3] - 主要关节
            left_hand_joints,     # [B, 15, 3] - 左手关节
            right_hand_joints     # [B, 15, 3] - 右手关节
        ], dim=1)  # [B, 52, 3]
        
        return all_joints
    
    def _predict_main_joints(self, global_feat: jt.Var, ground_truth_joints: jt.Var = None, teacher_forcing_ratio: float = 1.0):
        """
        预测主要的22个关节（躯干、四肢）
        """
        batch_size = global_feat.shape[0]
        
        # 初始化LSTM的隐藏状态和单元状态
        hidden = self.hidden_init(global_feat)  # [B, hidden_dim]
        cell = self.cell_init(global_feat)  # [B, hidden_dim]
        
        # 初始化根节点的父节点坐标为零向量
        root_parent_coord = jt.zeros((batch_size, 3), dtype=global_feat.dtype)
        
        # 初始化输出关节数组
        predicted_joints = []
        
        # 自回归生成前22个关节点
        for j in range(22):
            if j == 0:
                # 根节点没有父节点，使用零向量作为父节点坐标
                parent_coord = root_parent_coord
                relative_coord = jt.zeros((batch_size, 3), dtype=global_feat.dtype)
            else:
                parent_id = self.parent_ids[j]
                
                # 获取父节点坐标
                if ground_truth_joints is not None and jt.rand(1).item() < teacher_forcing_ratio:
                    # Teacher forcing: 使用真实的父节点坐标
                    parent_coord = ground_truth_joints[:, parent_id, :]
                else:
                    # 使用之前预测的父节点坐标
                    parent_coord = predicted_joints[parent_id]
                
                # 如果提供了真实关节，计算相对坐标
                if ground_truth_joints is not None and jt.rand(1).item() < teacher_forcing_ratio:
                    relative_coord = ground_truth_joints[:, j, :] - parent_coord
                else:
                    # 否则设置为零，让模型预测相对坐标
                    relative_coord = jt.zeros((batch_size, 3), dtype=global_feat.dtype)
            
            # 编码条件信息（父节点坐标和相对坐标）
            condition = self.condition_encoder(concat([parent_coord, relative_coord], dim=1))
            
            # 拼接条件和全局特征，送入LSTM单元
            lstm_input = concat([condition, global_feat], dim=1)
            hidden, cell = self.main_lstm(lstm_input, (hidden, cell))
            
            # 预测关节坐标
            if j == 0:
                # 根节点直接预测绝对坐标
                joint_coord = self.main_joint_predictor(hidden)
            else:
                # 对于非根节点，预测相对于父节点的偏移
                joint_offset = self.main_joint_predictor(hidden)
                joint_coord = parent_coord + joint_offset
            
            predicted_joints.append(joint_coord)
        
        # 将预测的关节点堆叠成张量 [B, 22, 3]
        return jt.stack(predicted_joints, dim=1)

# Factory function to create models - 只保留autoregressive
def create_model(model_name='pct', model_type='autoregressive', output_channels=156, **kwargs):
    """
    创建模型 - 只支持autoregressive类型
    
    Args:
        model_name: 模型名称，只支持'pct'
        model_type: 模型类型，只支持'autoregressive'
        output_channels: 输出通道数，应为156 (52*3)
        
    Returns:
        model: 创建的模型
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
