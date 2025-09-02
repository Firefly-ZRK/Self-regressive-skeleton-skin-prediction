import jittor as jt
from jittor import nn  
from jittor import init
from jittor.contrib import concat
import numpy as np
from PCT.misc.ops import FurthestPointSampler
from PCT.misc.ops import knn_point, index_points


def sample_and_group(npoint, nsample, xyz, points):
    B, N, C = xyz.shape
    S = npoint 
    # xyz = xyz.contiguous()
    sampler = FurthestPointSampler(npoint)
    _, fps_idx = sampler(xyz) # [B, npoint]
    # print ('fps size=', fps_idx.size())
    # fps_idx = sampler(xyz).long() # [B, npoint]
    new_xyz = index_points(xyz, fps_idx) 
    new_points = index_points(points, fps_idx)
    # new_xyz = xyz[:]
    # new_points = points[:]

    idx = knn_point(nsample, xyz, new_xyz)
    #idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    grouped_points = index_points(points, idx)
    grouped_points_norm = grouped_points - new_points.view(B, S, 1, -1)
    new_points = concat([grouped_points_norm, new_points.view(B, S, 1, -1).repeat(1, 1, nsample, 1)], dim=-1)
    return new_xyz, new_points



class Point_Transformer2(nn.Module):
    def __init__(self, output_channels=40):
        super(Point_Transformer2, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.gather_local_0 = Local_op(in_channels=128, out_channels=128)
        self.gather_local_1 = Local_op(in_channels=256, out_channels=256)
        self.pt_last = Point_Transformer_Last()

        self.relu = nn.ReLU()
        self.conv_fuse = nn.Sequential(nn.Conv1d(1280, 1024, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(1024),
                                   nn.LeakyReLU(scale=0.2))

        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=0.5)
        self.linear3 = nn.Linear(256, output_channels)

    def execute(self, x):
        xyz = x.permute(0, 2, 1)
        batch_size, _, _ = x.size()
        x = self.relu(self.bn1(self.conv1(x))) # B, D, N
        x = self.relu(self.bn2(self.conv2(x))) # B, D, N
        x = x.permute(0, 2, 1)
        new_xyz, new_feature = sample_and_group(npoint=512, nsample=32, xyz=xyz, points=x)         
        feature_0 = self.gather_local_0(new_feature)
        feature = feature_0.permute(0, 2, 1)
        new_xyz, new_feature = sample_and_group(npoint=256, nsample=32, xyz=new_xyz, points=feature) 
        feature_1 = self.gather_local_1(new_feature)
        # add position embedding on each layer
        x = self.pt_last(feature_1, new_xyz)
        x = concat([x, feature_1], dim=1)
        x = self.conv_fuse(x)
        x = jt.max(x, 2)
        x = x.view(batch_size, -1)

        x = self.relu(self.bn6(self.linear1(x)))
        x = self.dp1(x)
        x = self.relu(self.bn7(self.linear2(x)))
        x = self.dp2(x)
        x = self.linear3(x)

        return x



class Point_Transformer(nn.Module):
    def __init__(self, output_channels=40):
        super(Point_Transformer, self).__init__()
        
        # 增加输入通道的处理能力，更好地捕捉点云的几何特征
        self.conv1 = nn.Conv1d(3, 128, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(128, 128, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)

        # 使用级联的SA层提取不同尺度的特征
        self.sa1 = SA_Layer(128)
        self.sa2 = SA_Layer(128)
        self.sa3 = SA_Layer(128)
        self.sa4 = SA_Layer(128)

        # 多尺度特征融合
        self.conv_fuse = nn.Sequential(nn.Conv1d(512, 1024, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(1024),
                                   nn.LeakyReLU(scale=0.2))

        # 特征降维和分类头
        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=0.5)
        self.linear3 = nn.Linear(256, output_channels)

        self.relu = nn.ReLU()
        
    def execute(self, x):
        # x is expected to be [B, 3, N]
        batch_size, C, N = x.size()
        
        # Store original input for xyz coordinates
        x_input = x
        
        # Apply convolutions
        x = self.relu(self.bn1(self.conv1(x)))  # B, 128, N
        x = self.relu(self.bn2(self.conv2(x)))  # B, 128, N

        # Apply self-attention layers with xyz coordinates
        x1 = self.sa1(x, x_input)
        x2 = self.sa2(x1, x_input)
        x3 = self.sa3(x2, x_input)
        x4 = self.sa4(x3, x_input)
        
        # Concatenate features from all SA layers
        x = concat((x1, x2, x3, x4), dim=1)

        x = self.conv_fuse(x)
        # x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x = jt.max(x, 2)
        x = x.view(batch_size, -1)
        x = self.relu(self.bn6(self.linear1(x)))
        x = self.dp1(x)
        x = self.relu(self.bn7(self.linear2(x)))
        x = self.dp2(x)
        x = self.linear3(x)
        return x

class Point_Transformer_Last(nn.Module):
    def __init__(self, channels=256):
        super(Point_Transformer_Last, self).__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.conv_pos = nn.Conv1d(3, channels, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm1d(channels)

        self.sa1 = SA_Layer(channels)
        self.sa2 = SA_Layer(channels)
        self.sa3 = SA_Layer(channels)
        self.sa4 = SA_Layer(channels)

        self.relu = nn.ReLU()
        
    def pos_xyz(self, xyz):
        return self.conv_pos(xyz)
        
    def execute(self, x, xyz):
        # 
        # b, 3, npoint, nsample  
        # conv2d 3 -> 128 channels 1, 1
        # b * npoint, c, nsample 
        # permute reshape
        batch_size, _, N = x.size()
        # add position embedding
        xyz = xyz.permute(0, 2, 1)
        xyz = self.pos_xyz(xyz)
        # end
        x = self.relu(self.bn1(self.conv1(x))) # B, D, N

        x1 = self.sa1(x, xyz)
        x2 = self.sa2(x1, xyz)
        x3 = self.sa3(x2, xyz)
        x4 = self.sa4(x3, xyz)
        
        x = concat((x1, x2, x3, x4), dim=1)

        return x

class Local_op(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Local_op, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def execute(self, x):
        b, n, s, d = x.size()  # torch.Size([32, 512, 32, 6]) 
        x = x.permute(0, 1, 3, 2)
        x = x.reshape(-1, d, s)
        batch_size, _, N = x.size()
        x = self.relu(self.bn1(self.conv1(x))) # B, D, N
        x = self.relu(self.bn2(self.conv2(x))) # B, D, N
        x = jt.max(x, 2)
        x = x.view(batch_size, -1)
        x = x.reshape(b, n, -1).permute(0, 2, 1)
        return x



class SA_Layer(nn.Module):
    def __init__(self, channels):
        super(SA_Layer, self).__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
      # self.q_conv.conv.weight = self.k_conv.conv.weight 
        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        # Add a projection for xyz coordinates
        self.xyz_proj = nn.Conv1d(3, channels, 1, bias=False)

    def execute(self, x, xyz):
        # Project xyz to the same channel dimension as x
        xyz_feat = self.xyz_proj(xyz)
        
        # Now we can safely add them
        x = x + xyz_feat
        
        x_q = self.q_conv(x).permute(0, 2, 1) # b, n, c 
        x_k = self.k_conv(x)# b, c, n        
        x_v = self.v_conv(x)
        energy = nn.bmm(x_q, x_k) # b, n, n 
        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdims=True))
        x_r = nn.bmm(x_v, attention) # b, c, n 
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r
        return x


# 添加针对骨骼预测的改进模型
class SkeletonTransformer(nn.Module):
    def __init__(self, output_channels=66, num_joints=22):
        super(SkeletonTransformer, self).__init__()
        
        # 基础特征提取
        self.conv1 = nn.Conv1d(3, 128, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(128, 128, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)
        
        # 多层次的空间感知SA层
        self.sa1 = SA_Layer(128)
        self.sa2 = SA_Layer(128)
        self.sa3 = SA_Layer(128)
        self.sa4 = SA_Layer(128)
        
        # 空间关系感知层
        self.spatial_conv = nn.Conv1d(512, 512, kernel_size=1, bias=False)
        self.spatial_bn = nn.BatchNorm1d(512)
        
        # 多尺度特征融合
        self.conv_fuse = nn.Sequential(
            nn.Conv1d(512, 1024, kernel_size=1, bias=False),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(scale=0.2)
        )
        
        # 骨骼位置预测器
        self.joint_predictor = nn.Sequential(
            nn.Linear(1024, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, num_joints * 3)  # 预测num_joints个关节的3D坐标
        )
        
        self.relu = nn.ReLU()
        
    def execute(self, x):
        # x is expected to be [B, 3, N]
        batch_size, C, N = x.size()
        
        # 保存原始输入用于空间坐标
        x_input = x
        
        # 应用卷积提取特征
        x = self.relu(self.bn1(self.conv1(x)))  # B, 128, N
        x = self.relu(self.bn2(self.conv2(x)))  # B, 128, N
        
        # 应用空间感知的自注意力层
        x1 = self.sa1(x, x_input)
        x2 = self.sa2(x1, x_input)
        x3 = self.sa3(x2, x_input)
        x4 = self.sa4(x3, x_input)
        
        # 连接所有SA层的特征
        x = concat((x1, x2, x3, x4), dim=1)  # B, 512, N
        
        # 增强空间关系感知
        x = self.relu(self.spatial_bn(self.spatial_conv(x)))
        
        # 特征融合和全局池化
        x = self.conv_fuse(x)
        x = jt.max(x, 2)  # 全局最大池化
        x = x.view(batch_size, -1)
        
        # 预测骨骼节点位置
        joints = self.joint_predictor(x)
        
        return joints


# 针对蒙皮权重预测的改进模型
class SkinningTransformer(nn.Module):
    def __init__(self, feat_dim=256, num_joints=22):
        super(SkinningTransformer, self).__init__()
        self.num_joints = num_joints
        self.feat_dim = feat_dim
        
        # 点云特征提取
        self.pct = Point_Transformer(output_channels=feat_dim)
        
        # 顶点特征提取
        self.vertex_mlp = nn.Sequential(
            nn.Linear(3 + feat_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(),
        )
        
        # 关节特征提取
        self.joint_mlp = nn.Sequential(
            nn.Linear(3 + feat_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(),
        )
        
        # 空间关系编码
        self.spatial_encoder = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(),
        )
        
        self.relu = nn.ReLU()
        
    def forward_vertex_mlp(self, x):
        B = x.shape[0]
        return self.vertex_mlp(x.reshape(-1, x.shape[-1])).reshape(B, -1, self.feat_dim)
    
    def forward_joint_mlp(self, x):
        B = x.shape[0]
        return self.joint_mlp(x.reshape(-1, x.shape[-1])).reshape(B, -1, self.feat_dim)
    
    def execute(self, vertices, joints):
        # 提取形状潜在特征
        shape_latent = self.relu(self.pct(vertices.permute(0, 2, 1)))
        
        # 处理顶点特征
        vertices_with_latent = concat([vertices, shape_latent.unsqueeze(1).repeat(1, vertices.shape[1], 1)], dim=-1)
        vertices_latent = self.forward_vertex_mlp(vertices_with_latent)
        
        # 处理关节特征
        joints_with_latent = concat([joints, shape_latent.unsqueeze(1).repeat(1, self.num_joints, 1)], dim=-1)
        joints_latent = self.forward_joint_mlp(joints_with_latent)
        
        # 空间关系编码
        vertices_latent = self.spatial_encoder(vertices_latent.reshape(-1, self.feat_dim)).reshape(vertices_latent.shape)
        
        # 计算权重 (注意力机制)
        weights = nn.bmm(vertices_latent, joints_latent.permute(0, 2, 1)) / np.sqrt(self.feat_dim)
        weights = nn.softmax(weights, dim=-1)
        
        return weights


if __name__ == '__main__':
    
    jt.flags.use_cuda=1
    input_points = init.gauss((16, 3, 1024), dtype='float32')  # B, D, N 


    network = Point_Transformer()
    out_logits = network(input_points)
    print (out_logits.shape)

