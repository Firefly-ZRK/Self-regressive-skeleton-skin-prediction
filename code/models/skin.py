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

# Define hand joint indices (52 joints version)
LEFT_HAND_JOINTS = list(range(22, 37))   # Left hand joints 22-36
RIGHT_HAND_JOINTS = list(range(37, 52))  # Right hand joints 37-51
HAND_JOINTS = LEFT_HAND_JOINTS + RIGHT_HAND_JOINTS

# Main joint indices (non-finger joints)
MAIN_JOINTS = list(range(22))  # First 22 joints are main joints

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

# Positional encoding
class PositionalEncoding(nn.Module):
    """Positional encoding for joints/bones to enhance spatial position information"""
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        # Ensure d_model is divisible by 6
        self.d_model = d_model - (d_model % 6) if d_model % 6 != 0 else d_model
        
    def execute(self, x: jt.Var):
        """
        Args:
            x: Coordinate tensor [B, N, 3]
        Returns:
            Position-encoded features [B, N, d_model]
        """
        B, N = x.shape[0], x.shape[1]
        # Create positional encoding matrix
        pe = jt.zeros((B, N, self.d_model))
        
        # Ensure d_model is divisible by 6
        feature_dim = self.d_model // 6
        
        # Use sine/cosine functions with different frequencies for each dimension
        div_term = jt.exp(jt.arange(0, self.d_model, 2) * -(jt.log(jt.Var([10000.0])) / self.d_model))
        
        # Encode x, y, z coordinates separately
        for dim in range(3):
            pos = x[:, :, dim].unsqueeze(2)  # [B, N, 1]
            pe[:, :, dim::6] = jt.sin(pos * div_term[:feature_dim])
            pe[:, :, dim+3::6] = jt.cos(pos * div_term[:feature_dim])
        
        return self.dropout(pe)

# Enhanced MLP module with increased layers and residual connections for feature retention
class EnhancedMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        self.layer3 = nn.Linear(hidden_dim, output_dim)
        self.norm3 = nn.LayerNorm(output_dim)
        
        # Add projection layer for residual connection if input and output dimensions differ
        self.shortcut = None
        if input_dim != output_dim:
            self.shortcut = nn.Linear(input_dim, output_dim)
    
    def execute(self, x):
        # First layer
        identity = x
        x = self.layer1(x)
        x = self.norm1(x)
        x = nn.relu(x)
        
        # Second layer
        x = self.layer2(x)
        x = self.norm2(x)
        x = nn.relu(x)
        
        # Third layer
        x = self.layer3(x)
        x = self.norm3(x)
        
        # Residual connection
        if self.shortcut is not None:
            identity = self.shortcut(identity)
        
        x = x + identity
        return nn.relu(x)

# Point encoder
class PointEncoder(nn.Module):
    """Point cloud encoder that extracts local and global geometric features for each vertex"""
    def __init__(self, feat_dim: int = 256):
        super().__init__()
        self.feat_dim = feat_dim
        # Use PCT network to extract point cloud features
        self.pct = Point_Transformer(output_channels=feat_dim)
        # Vertex local feature extraction
        self.vertex_mlp = nn.Sequential(
            nn.Linear(3, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.ReLU()
        )
        # Feature fusion - Use enhanced MLP with residual connections
        self.fusion = EnhancedMLP(feat_dim + 128, feat_dim, hidden_dim=512)
        
        # Feature enhancement layer - Ensure important features are not lost
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
            vertices: Point cloud vertices [B, N, 3]
        Returns:
            point_features: Point features [B, N, feat_dim]
        """
        batch_size, num_points = vertices.shape[0], vertices.shape[1]
        
        # Extract global features
        global_feat = self.pct(vertices.permute(0, 2, 1))  # [B, feat_dim]
        
        # Extract local features for each vertex
        local_feat = self.vertex_mlp(vertices)  # [B, N, 128]
        
        # Fuse global and local features
        global_feat_expanded = global_feat.unsqueeze(1).repeat(1, num_points, 1)  # [B, N, feat_dim]
        fused_features = self.fusion(concat([local_feat, global_feat_expanded], dim=-1))  # [B, N, feat_dim]
        
        # Apply feature enhancement to ensure features are not lost
        point_features = fused_features + self.feature_enhancer(fused_features)  # Residual connection
        
        return point_features

# Bone encoder
class BoneEncoder(nn.Module):
    """Bone encoder that encodes joint coordinates into bone features"""
    def __init__(self, feat_dim: int = 256, num_joints: int = 52):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_joints = num_joints
        
        # Positional encoding - Adjusted for 52 joints, using 312 (6*52=312)
        self.pos_encoding = PositionalEncoding(d_model=312, dropout=0.1)
        
        # Bone feature extraction - Use enhanced MLP
        self.joint_encoder = EnhancedMLP(3 + 312, feat_dim, hidden_dim=512)  # Input is joint coordinates(3) + positional encoding(312)
        
        # Feature enhancement layer
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
            joints: Joint coordinates [B, J, 3]
        Returns:
            bone_features: Bone features [B, J, feat_dim]
        """
        # Add positional encoding
        pos_encoded = self.pos_encoding(joints)  # [B, J, 312]
        
        # Concatenate original coordinates with positional encoding
        joint_input = concat([joints, pos_encoded], dim=-1)  # [B, J, 3+312]
        
        # Encode to bone features
        bone_features = self.joint_encoder(joint_input)  # [B, J, feat_dim]
        
        # Apply feature enhancement
        bone_features = bone_features + self.feature_enhancer(bone_features)  # Residual connection
        
        return bone_features

# Cross-attention module
class CrossAttention(nn.Module):
    """Cross-attention module for computing relationships between points and bones"""
    def __init__(self, feat_dim: int = 256, num_heads: int = 8):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_heads = num_heads
        self.head_dim = feat_dim // num_heads
        assert self.head_dim * num_heads == feat_dim, "feat_dim must be divisible by num_heads"
        
        # Projection matrices
        self.q_proj = nn.Linear(feat_dim, feat_dim)
        self.k_proj = nn.Linear(feat_dim, feat_dim)
        self.v_proj = nn.Linear(feat_dim, feat_dim)
        
        # Output projection
        self.out_proj = nn.Linear(feat_dim, feat_dim)
        
        # Add residual connection for input features
        self.layer_norm = nn.LayerNorm(feat_dim)
        
        # Scaling factor
        self.scale = self.head_dim ** -0.5
        
    def execute(self, point_features: jt.Var, bone_features: jt.Var):
        """
        Args:
            point_features: Point features [B, N, feat_dim]
            bone_features: Bone features [B, J, feat_dim]
        Returns:
            attention_weights: Attention weights [B, N, J]
            attention_output: Attention output [B, N, feat_dim]
        """
        B, N, _ = point_features.shape
        _, J, _ = bone_features.shape
        
        # Save input features for residual connection
        identity = point_features
        
        # Compute query, key, and value
        q = self.q_proj(point_features)  # [B, N, feat_dim]
        k = self.k_proj(bone_features)   # [B, J, feat_dim]
        v = self.v_proj(bone_features)   # [B, J, feat_dim]
        
        # Reshape to multi-head form
        q = q.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, num_heads, N, head_dim]
        k = k.reshape(B, J, self.num_heads, self.head_dim).permute(0, 2, 3, 1)  # [B, num_heads, head_dim, J]
        v = v.reshape(B, J, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, num_heads, J, head_dim]
        
        # Compute attention scores
        attn = (q @ k) * self.scale  # [B, num_heads, N, J]
        
        # Softmax normalization
        attn_weights = nn.softmax(attn, dim=-1)  # [B, num_heads, N, J]
        
        # Compute attention output
        attn_output = (attn_weights @ v)  # [B, num_heads, N, head_dim]
        attn_output = attn_output.permute(0, 2, 1, 3).reshape(B, N, self.feat_dim)  # [B, N, feat_dim]
        
        # Final projection
        output = self.out_proj(attn_output)  # [B, N, feat_dim]
        
        # Apply residual connection and layer normalization
        output = self.layer_norm(output + identity)
        
        # Average to get final attention weights
        attention_weights = attn_weights.mean(dim=1)  # [B, N, J]
        
        return attention_weights, output

class HandAwareAttention(nn.Module):
    """Hand-aware attention module using deeper attention for hand regions"""
    def __init__(self, feat_dim: int = 256, num_heads: int = 8):
        super().__init__()
        self.feat_dim = feat_dim
        
        # Main joint attention
        self.main_attention = CrossAttention(feat_dim, num_heads)
        
        # Deep attention for hand joints
        self.hand_attention1 = CrossAttention(feat_dim, num_heads)
        self.hand_attention2 = CrossAttention(feat_dim, num_heads)
        
        # Feature fusion
        self.fusion = EnhancedMLP(feat_dim * 2, feat_dim, hidden_dim=512)
        
    def execute(self, point_features: jt.Var, bone_features: jt.Var):
        """
        Args:
            point_features: Point features [B, N, feat_dim]
            bone_features: Bone features [B, J, feat_dim] (J=52)
        Returns:
            combined_weights: Combined attention weights [B, N, J]
            combined_output: Combined output features [B, N, feat_dim]
        """
        # Separate main joints and hand joints
        main_bone_features = bone_features[:, :22, :]  # [B, 22, feat_dim]
        hand_bone_features = bone_features[:, 22:, :]  # [B, 30, feat_dim]
        
        # Compute main joint attention
        main_weights, main_output = self.main_attention(point_features, main_bone_features)
        
        # Compute deep attention for hand joints
        hand_output1 = point_features
        hand_weights1, hand_output1 = self.hand_attention1(hand_output1, hand_bone_features)
        hand_weights2, hand_output2 = self.hand_attention2(hand_output1, hand_bone_features)
        
        # Fuse main joint and hand joint outputs
        fused_output = self.fusion(concat([main_output, hand_output2], dim=-1))
        
        # Concatenate weights
        combined_weights = concat([main_weights, hand_weights2], dim=-1)  # [B, N, 52]
        
        return combined_weights, fused_output

class CrossAttentionSkinModel(nn.Module):
    """Cross-attention based skinning prediction model supporting 52 joints"""
    def __init__(self, feat_dim: int = 256, num_joints: int = 52, num_heads: int = 8):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_joints = num_joints
        
        # Point cloud encoder
        self.point_encoder = PointEncoder(feat_dim=feat_dim)
        
        # Bone encoder
        self.bone_encoder = BoneEncoder(feat_dim=feat_dim, num_joints=num_joints)
        
        # Multi-layer cross-attention
        self.cross_attention1 = CrossAttention(feat_dim=feat_dim, num_heads=num_heads)
        self.cross_attention2 = CrossAttention(feat_dim=feat_dim, num_heads=num_heads)
        self.cross_attention3 = CrossAttention(feat_dim=feat_dim, num_heads=num_heads)
        
        # Hand-aware attention
        self.hand_attention = HandAwareAttention(feat_dim=feat_dim, num_heads=num_heads)
        
        # Final skinning weight prediction - Use enhanced MLP
        self.skin_predictor = EnhancedMLP(feat_dim * 4, num_joints, hidden_dim=1024)
        
        # Feature integration
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
            vertices: Point cloud vertices [B, N, 3]
            joints: Joint coordinates [B, J, 3] (J=52)
        Returns:
            skin_weights: Skinning weights [B, N, J]
        """
        # Encode point cloud and bones
        point_features = self.point_encoder(vertices)  # [B, N, feat_dim]
        bone_features = self.bone_encoder(joints)      # [B, J, feat_dim]
        
        # Multi-layer cross-attention, progressively refining features
        _, attn_output1 = self.cross_attention1(point_features, bone_features)
        _, attn_output2 = self.cross_attention2(attn_output1, bone_features)
        _, attn_output3 = self.cross_attention3(attn_output2, bone_features)
        
        # Hand-aware attention
        hand_weights, hand_output = self.hand_attention(attn_output3, bone_features)
        
        # Feature integration
        integrated_features = self.feature_integration(
            concat([attn_output1, attn_output2, attn_output3], dim=-1)
        )
        
        # Concatenate all features to predict final skinning weights
        final_features = concat([
            point_features,      # Original point features
            integrated_features, # Integrated attention features
            attn_output3,       # Last layer attention output
            hand_output         # Hand-aware output
        ], dim=-1)  # [B, N, feat_dim*4]
        
        skin_logits = self.skin_predictor(final_features)  # [B, N, 52]
        
        # Softmax normalization to ensure weight sum equals 1 for each vertex
        skin_weights = nn.softmax(skin_logits, dim=-1)
        
        return skin_weights

# Factory function to create models - Only keep cross_attention
def create_model(model_name='pct', model_type='cross_attention', feat_dim=256, **kwargs):
    """
    Create model - Only supports cross_attention type
    
    Args:
        model_name: Model name, only supports 'pct'
        model_type: Model type, only supports 'cross_attention'
        feat_dim: Feature dimension
        
    Returns:
        model: Created model
    """
    if model_name == "pct":
        if model_type == 'cross_attention':
            return CrossAttentionSkinModel(feat_dim=feat_dim, num_joints=52, num_heads=8)
        else:
            raise ValueError(f"Unsupported model_type: {model_type}. Only 'cross_attention' is supported.")
    else:
        raise ValueError(f"Unsupported model_name: {model_name}. Only 'pct' is supported.")