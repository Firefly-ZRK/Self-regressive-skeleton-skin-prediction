import jittor as jt
from jittor import nn
from jittor import init
from jittor.contrib import concat
import numpy as np
import sys
from math import sqrt

# Import the PCT model components
from PCT.networks.cls.pct import Point_Transformer, Point_Transformer2, Point_Transformer_Last, SA_Layer, Local_op, sample_and_group
from PCT.networks.cls.pct import SkinningTransformer

# 定义手部关节索引 (52个关节版本)
LEFT_HAND_JOINTS = list(range(22, 37))   # 左手关节 22-36
RIGHT_HAND_JOINTS = list(range(37, 52))  # 右手关节 37-51
HAND_JOINTS = LEFT_HAND_JOINTS + RIGHT_HAND_JOINTS

# 主要关节索引（非手指关节）
MAIN_JOINTS = list(range(22))  # 前22个关节是主要关节

class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
        )
    
    def execute(self, x):
        B = x.shape[0]
        return self.encoder(x.reshape(-1, self.input_dim)).reshape(B, -1, self.output_dim)

class SimpleSkinModel(nn.Module):

    def __init__(self, feat_dim: int, num_joints: int):
        super().__init__()
        self.num_joints = num_joints
        self.feat_dim = feat_dim

        self.pct = Point_Transformer(output_channels=feat_dim)
        self.joint_mlp = MLP(3 + feat_dim, feat_dim)
        self.vertex_mlp = MLP(3 + feat_dim, feat_dim)
        self.relu = nn.ReLU()
    
    def execute(self, vertices: jt.Var, joints: jt.Var):
        # (B, latents)
        shape_latent = self.relu(self.pct(vertices.permute(0, 2, 1)))

        # (B, N, latents)
        vertices_latent = (
            self.vertex_mlp(concat([vertices, shape_latent.unsqueeze(1).repeat(1, vertices.shape[1], 1)], dim=-1))
        )

        # (B, num_joints, latents)
        joints_latent = (
            self.joint_mlp(concat([joints, shape_latent.unsqueeze(1).repeat(1, self.num_joints, 1)], dim=-1))
        )

        # (B, N, num_joints)
        res = nn.softmax(vertices_latent @ joints_latent.permute(0, 2, 1) / sqrt(self.feat_dim), dim=-1)
        assert not jt.isnan(res).any()

        return res

class EnhancedSkinModel(nn.Module):
    def __init__(self, feat_dim: int = 256, num_joints: int = 22):
        super().__init__()
        self.skinning_transformer = SkinningTransformer(feat_dim=feat_dim, num_joints=num_joints)
    
    def execute(self, vertices: jt.Var, joints: jt.Var):
        return self.skinning_transformer(vertices, joints)

# 位置编码
class PositionalEncoding(nn.Module):
    """关节/骨骼的位置编码，用于增强空间位置信息"""
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        # 确保d_model是6的倍数
        self.d_model = d_model - (d_model % 6) if d_model % 6 != 0 else d_model
        
    def execute(self, x: jt.Var):
        """
        Args:
            x: 坐标张量 [B, N, 3]
        Returns:
            位置编码后的特征 [B, N, d_model]
        """
        B, N = x.shape[0], x.shape[1]
        # 创建位置编码矩阵
        pe = jt.zeros((B, N, self.d_model))
        
        # 确保d_model是6的倍数
        feature_dim = self.d_model // 6
        
        # 对每个维度使用不同频率的正弦/余弦函数
        div_term = jt.exp(jt.arange(0, self.d_model, 2) * -(jt.log(jt.Var([10000.0])) / self.d_model))
        
        # 对x, y, z三个坐标分别编码
        for dim in range(3):
            pos = x[:, :, dim].unsqueeze(2)  # [B, N, 1]
            pe[:, :, dim::6] = jt.sin(pos * div_term[:feature_dim])
            pe[:, :, dim+3::6] = jt.cos(pos * div_term[:feature_dim])
        
        return self.dropout(pe)

# 增强的MLP模块，增加层数和添加残差连接以保留特征
class EnhancedMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        self.layer3 = nn.Linear(hidden_dim, output_dim)
        self.norm3 = nn.LayerNorm(output_dim)
        
        # 如果输入和输出维度不同，添加一个投影层以便进行残差连接
        self.shortcut = None
        if input_dim != output_dim:
            self.shortcut = nn.Linear(input_dim, output_dim)
    
    def execute(self, x):
        # 第一层
        identity = x
        x = self.layer1(x)
        x = self.norm1(x)
        x = nn.relu(x)
        
        # 第二层
        x = self.layer2(x)
        x = self.norm2(x)
        x = nn.relu(x)
        
        # 第三层
        x = self.layer3(x)
        x = self.norm3(x)
        
        # 残差连接
        if self.shortcut is not None:
            identity = self.shortcut(identity)
        
        x = x + identity
        return nn.relu(x)

# 点编码器
class PointEncoder(nn.Module):
    """点云编码器，提取每个顶点的局部和全局几何特征"""
    def __init__(self, feat_dim: int = 256):
        super().__init__()
        self.feat_dim = feat_dim
        # 使用PCT网络提取点云特征
        self.pct = Point_Transformer(output_channels=feat_dim)
        # 顶点局部特征提取
        self.vertex_mlp = nn.Sequential(
            nn.Linear(3, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.ReLU()
        )
        # 特征融合 - 使用增强的MLP和残差连接
        self.fusion = EnhancedMLP(feat_dim + 128, feat_dim, hidden_dim=512)
        
        # 特征增强层 - 确保不丢失重要特征
        self.feature_enhancer = nn.Sequential(
            nn.Linear(feat_dim, feat_dim * 2),
            nn.LayerNorm(feat_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feat_dim * 2, feat_dim),
            nn.LayerNorm(feat_dim)
        )
        
    def execute(self, vertices: jt.Var):
        """
        Args:
            vertices: 点云顶点 [B, N, 3]
        Returns:
            point_features: 点特征 [B, N, feat_dim]
        """
        batch_size, num_points = vertices.shape[0], vertices.shape[1]
        
        # 提取全局特征
        global_feat = self.pct(vertices.permute(0, 2, 1))  # [B, feat_dim]
        
        # 提取每个顶点的局部特征
        local_feat = self.vertex_mlp(vertices)  # [B, N, 128]
        
        # 融合全局和局部特征
        global_feat_expanded = global_feat.unsqueeze(1).repeat(1, num_points, 1)  # [B, N, feat_dim]
        fused_features = self.fusion(concat([local_feat, global_feat_expanded], dim=-1))  # [B, N, feat_dim]
        
        # 应用特征增强以确保特征不会丢失
        point_features = fused_features + self.feature_enhancer(fused_features)  # 残差连接
        
        return point_features

# 骨骼编码器
class BoneEncoder(nn.Module):
    """骨骼编码器，将关节点坐标编码为骨骼特征"""
    def __init__(self, feat_dim: int = 256, num_joints: int = 52):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_joints = num_joints
        
        # 位置编码 - 为52个关节调整，使用312(6*52=312)
        self.pos_encoding = PositionalEncoding(d_model=312, dropout=0.1)
        
        # 骨骼特征提取 - 使用增强的MLP
        self.joint_encoder = EnhancedMLP(3 + 312, feat_dim, hidden_dim=512)  # 输入是关节坐标(3)和位置编码(312)
        
        # 特征增强层
        self.feature_enhancer = nn.Sequential(
            nn.Linear(feat_dim, feat_dim * 2),
            nn.LayerNorm(feat_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feat_dim * 2, feat_dim),
            nn.LayerNorm(feat_dim)
        )
        
    def execute(self, joints: jt.Var):
        """
        Args:
            joints: 关节坐标 [B, J, 3]
        Returns:
            bone_features: 骨骼特征 [B, J, feat_dim]
        """
        # 添加位置编码
        pos_encoded = self.pos_encoding(joints)  # [B, J, 312]
        
        # 拼接原始坐标和位置编码
        joint_input = concat([joints, pos_encoded], dim=-1)  # [B, J, 3+312]
        
        # 编码为骨骼特征
        bone_features = self.joint_encoder(joint_input)  # [B, J, feat_dim]
        
        # 应用特征增强
        bone_features = bone_features + self.feature_enhancer(bone_features)  # 残差连接
        
        return bone_features

# 交叉注意力模块
class CrossAttention(nn.Module):
    """交叉注意力模块，计算点与骨骼之间的关系"""
    def __init__(self, feat_dim: int = 256, num_heads: int = 8):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_heads = num_heads
        self.head_dim = feat_dim // num_heads
        assert self.head_dim * num_heads == feat_dim, "feat_dim必须能被num_heads整除"
        
        # 投影矩阵
        self.q_proj = nn.Linear(feat_dim, feat_dim)
        self.k_proj = nn.Linear(feat_dim, feat_dim)
        self.v_proj = nn.Linear(feat_dim, feat_dim)
        
        # 输出投影
        self.out_proj = nn.Linear(feat_dim, feat_dim)
        
        # 引入输入特征的残差连接
        self.layer_norm = nn.LayerNorm(feat_dim)
        
        # 缩放因子
        self.scale = self.head_dim ** -0.5
        
    def execute(self, point_features: jt.Var, bone_features: jt.Var):
        """
        Args:
            point_features: 点特征 [B, N, feat_dim]
            bone_features: 骨骼特征 [B, J, feat_dim]
        Returns:
            attention_weights: 注意力权重 [B, N, J]
            attention_output: 注意力输出 [B, N, feat_dim]
        """
        B, N, _ = point_features.shape
        _, J, _ = bone_features.shape
        
        # 保存输入特征用于残差连接
        identity = point_features
        
        # 计算查询、键和值
        q = self.q_proj(point_features)  # [B, N, feat_dim]
        k = self.k_proj(bone_features)   # [B, J, feat_dim]
        v = self.v_proj(bone_features)   # [B, J, feat_dim]
        
        # 重塑为多头形式
        q = q.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, num_heads, N, head_dim]
        k = k.reshape(B, J, self.num_heads, self.head_dim).permute(0, 2, 3, 1)  # [B, num_heads, head_dim, J]
        v = v.reshape(B, J, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, num_heads, J, head_dim]
        
        # 计算注意力分数
        attn = (q @ k) * self.scale  # [B, num_heads, N, J]
        
        # Softmax归一化
        attn_weights = nn.softmax(attn, dim=-1)  # [B, num_heads, N, J]
        
        # 计算注意力输出
        attn_output = (attn_weights @ v)  # [B, num_heads, N, head_dim]
        attn_output = attn_output.permute(0, 2, 1, 3).reshape(B, N, self.feat_dim)  # [B, N, feat_dim]
        
        # 最终投影
        output = self.out_proj(attn_output)  # [B, N, feat_dim]
        
        # 应用残差连接和层归一化
        output = self.layer_norm(output + identity)
        
        # 求平均得到最终的注意力权重
        attention_weights = attn_weights.mean(dim=1)  # [B, N, J]
        
        return attention_weights, output

class HandAwareAttention(nn.Module):
    """手部感知注意力模块，对手部区域使用更深的注意力"""
    def __init__(self, feat_dim: int = 256, num_heads: int = 8):
        super().__init__()
        self.feat_dim = feat_dim
        
        # 主要关节的注意力
        self.main_attention = CrossAttention(feat_dim, num_heads)
        
        # 手部关节的深度注意力
        self.hand_attention1 = CrossAttention(feat_dim, num_heads)
        self.hand_attention2 = CrossAttention(feat_dim, num_heads)
        
        # 特征融合
        self.fusion = EnhancedMLP(feat_dim * 2, feat_dim, hidden_dim=512)
        
    def execute(self, point_features: jt.Var, bone_features: jt.Var):
        """
        Args:
            point_features: 点特征 [B, N, feat_dim]
            bone_features: 骨骼特征 [B, J, feat_dim] (J=52)
        Returns:
            combined_weights: 组合注意力权重 [B, N, J]
            combined_output: 组合输出特征 [B, N, feat_dim]
        """
        # 分离主要关节和手部关节
        main_bone_features = bone_features[:, :22, :]  # [B, 22, feat_dim]
        hand_bone_features = bone_features[:, 22:, :]  # [B, 30, feat_dim]
        
        # 计算主要关节的注意力
        main_weights, main_output = self.main_attention(point_features, main_bone_features)
        
        # 计算手部关节的深度注意力
        hand_output1 = point_features
        hand_weights1, hand_output1 = self.hand_attention1(hand_output1, hand_bone_features)
        hand_weights2, hand_output2 = self.hand_attention2(hand_output1, hand_bone_features)
        
        # 融合主要关节和手部关节的输出
        fused_output = self.fusion(concat([main_output, hand_output2], dim=-1))
        
        # 拼接权重
        combined_weights = concat([main_weights, hand_weights2], dim=-1)  # [B, N, 52]
        
        return combined_weights, fused_output

class CrossAttentionSkinModel(nn.Module):
    """基于交叉注意力的蒙皮预测模型，支持52个关节"""
    def __init__(self, feat_dim: int = 256, num_joints: int = 52, num_heads: int = 8):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_joints = num_joints
        
        # 点云编码器
        self.point_encoder = PointEncoder(feat_dim=feat_dim)
        
        # 骨骼编码器
        self.bone_encoder = BoneEncoder(feat_dim=feat_dim, num_joints=num_joints)
        
        # 多层交叉注意力
        self.cross_attention1 = CrossAttention(feat_dim=feat_dim, num_heads=num_heads)
        self.cross_attention2 = CrossAttention(feat_dim=feat_dim, num_heads=num_heads)
        self.cross_attention3 = CrossAttention(feat_dim=feat_dim, num_heads=num_heads)
        
        # 手部感知注意力
        self.hand_attention = HandAwareAttention(feat_dim=feat_dim, num_heads=num_heads)
        
        # 最终的蒙皮权重预测 - 使用增强的MLP
        self.skin_predictor = EnhancedMLP(feat_dim * 4, num_joints, hidden_dim=1024)
        
        # 特征整合
        self.feature_integration = nn.Sequential(
            nn.Linear(feat_dim * 3, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feat_dim, feat_dim),
            nn.LayerNorm(feat_dim)
        )
        
    def execute(self, vertices: jt.Var, joints: jt.Var):
        """
        Args:
            vertices: 点云顶点 [B, N, 3]
            joints: 关节坐标 [B, J, 3] (J=52)
        Returns:
            skin_weights: 蒙皮权重 [B, N, J]
        """
        # 编码点云和骨骼
        point_features = self.point_encoder(vertices)  # [B, N, feat_dim]
        bone_features = self.bone_encoder(joints)      # [B, J, feat_dim]
        
        # 多层交叉注意力，逐步细化特征
        _, attn_output1 = self.cross_attention1(point_features, bone_features)
        _, attn_output2 = self.cross_attention2(attn_output1, bone_features)
        _, attn_output3 = self.cross_attention3(attn_output2, bone_features)
        
        # 手部感知注意力
        hand_weights, hand_output = self.hand_attention(attn_output3, bone_features)
        
        # 特征整合
        integrated_features = self.feature_integration(
            concat([attn_output1, attn_output2, attn_output3], dim=-1)
        )
        
        # 拼接所有特征，预测最终的蒙皮权重
        final_features = concat([
            point_features,      # 原始点特征
            integrated_features, # 整合的注意力特征
            attn_output3,       # 最后一层注意力输出
            hand_output         # 手部感知输出
        ], dim=-1)  # [B, N, feat_dim*4]
        
        skin_logits = self.skin_predictor(final_features)  # [B, N, 52]
        
        # Softmax归一化，确保每个顶点的权重和为1
        skin_weights = nn.softmax(skin_logits, dim=-1)
        
        return skin_weights

# Factory function to create models - 只保留cross_attention
def create_model(model_name='pct', model_type='cross_attention', feat_dim=256, **kwargs):
    """
    创建模型 - 只支持cross_attention类型
    
    Args:
        model_name: 模型名称，只支持'pct'
        model_type: 模型类型，只支持'cross_attention'
        feat_dim: 特征维度
        
    Returns:
        model: 创建的模型
    """
    if model_name == "pct":
        if model_type == 'cross_attention':
            return CrossAttentionSkinModel(feat_dim=feat_dim, num_joints=52, num_heads=8)
        else:
            raise ValueError(f"Unsupported model_type: {model_type}. Only 'cross_attention' is supported.")
    else:
        raise ValueError(f"Unsupported model_name: {model_name}. Only 'pct' is supported.")