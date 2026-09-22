# interiornetStreetlearn_8ptVit模块模型文件注释

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from .modules.extractor import ResidualBlock
from .modules.vision_transformer import _create_vision_transformer
from src.geom.RotationContinuity.sanity_test.code.tools import compute_rotation_matrix_from_ortho6d, compute_pose_from_rotation_matrix

def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """
    将旋转矩阵转换为6D旋转表示（Zhou et al. [1]）
    通过丢弃最后一行实现
    Args:
        matrix: 批量旋转矩阵，形状 (*, 3, 3)
    Returns:
        6D旋转表示，形状 (*, 6)
    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    """
    return matrix[..., :2, :].clone().reshape(*matrix.size()[:-2], 6)

def compute_6d(pose_mtx):
    """计算6D位姿表示"""
    pose_6d = matrix_to_rotation_6d(pose_mtx[...,:3,:3])
    tr = pose_mtx[...,:3,3]
    pose_6d_norm = torch.cat([tr,pose_6d],dim=-1)
    return pose_6d_norm

def compute_normalized_6d(pose_mtx, global_pose_mean, global_pose_std):
    """计算归一化的6D位姿表示"""
    pose_6d = matrix_to_rotation_6d(pose_mtx[...,:3,:3])
    tr = pose_mtx[...,:3,3]
    pose_6d_norm = (torch.cat([tr,pose_6d],dim=-1) - global_pose_mean.to(device=pose_6d.device)) / global_pose_std.to(device=pose_6d.device)
    return pose_6d_norm

class ViTEss(nn.Module):
    def __init__(self, args, global_pose_mean=None, global_pose_std=None):
        super(ViTEss, self).__init__()
        
        # 超参数设置
        self.total_num_features = 192  # 总特征数量
        self.feature_resolution = (24, 24)  # 特征分辨率
        self.pose_size = 9  # 位姿向量大小（3平移 + 6旋转）
        self.num_patches = self.feature_resolution[0] * self.feature_resolution[1]  # patch数量
        extractor_final_conv_kernel_size = max(1, 28-self.feature_resolution[0]+1)  # 特征提取器最终卷积核大小
        self.pool_feat1 = min(96, 4 * args.pool_size)  # 第一池化特征数
        self.pool_feat2 = args.pool_size  # 第二池化特征数
        self.H2 = args.fc_hidden_size  # 全连接层隐藏大小
        self.use_loftr_gating = args.use_loftr_gating  # 是否使用LoFTR门控
        self.global_pose_mean = global_pose_mean  # 全局位姿均值
        self.global_pose_std = global_pose_std  # 全局位姿标准差
        self.use_normalized_6d = args.use_normalized_6d  # 是否使用归一化6D表示
        self.T_pose = args.T_pose  # T位姿参数
        
        # 网络层定义
        self.flatten = nn.Flatten(0,1)  # 扁平化层
        
        # ResNet特征提取器
        self.resnet = models.resnet18(pretrained=True)  # 使用预训练的ResNet18
        self.resnet.fc = nn.Identity()  # 移除全连接层
        self.extractor_final_conv = ResidualBlock(128, self.total_num_features, "batch", kernel_size=extractor_final_conv_kernel_size)
        
        # Fusion Transformer（可选）
        self.fusion_transformer = None
        if args.fusion_transformer:
            self.num_heads = 3  # 注意力头数
            model_kwargs = dict(patch_size=16, embed_dim=self.total_num_features, depth=args.transformer_depth, 
                                num_heads=self.num_heads)
            self.fusion_transformer = _create_vision_transformer("vit_tiny_patch16_384", **model_kwargs)
            
            # 配置Transformer深度
            self.transformer_depth = args.transformer_depth
            self.fusion_transformer.blocks = self.fusion_transformer.blocks[:args.transformer_depth]
            self.fusion_transformer.patch_embed = nn.Identity()
            self.fusion_transformer.head = nn.Identity()
            self.fusion_transformer.cls_token = None
            
            # 位置编码
            self.fusion_transformer.pos_embed = nn.Parameter(torch.zeros([1,self.num_patches,self.total_num_features]))
            nn.init.xavier_uniform_(self.fusion_transformer.pos_embed)
            
            pos_enc = 6  # 位置编码维度
            self.H = int(self.num_heads*2*(self.total_num_features//self.num_heads + pos_enc) * (self.total_num_features//self.num_heads))
        else:
            # 普通池化路径
            self.H = self.pool_feat2 * self.feature_resolution[0] * self.feature_resolution[1]
            self.pool_transformer_output = nn.Sequential(
                nn.Conv2d(self.total_num_features, self.pool_feat1, kernel_size=1, bias=True),
                nn.BatchNorm2d(self.pool_feat1),
                nn.ReLU(),
                nn.Conv2d(self.pool_feat1, self.pool_feat2, kernel_size=1, bias=True),
                nn.BatchNorm2d(self.pool_feat2),
            )
        
        # LoFTR门控网络（可选）
        if self.use_loftr_gating:
            self.moe_predictor = nn.Sequential(
                nn.Linear(self.H+2*self.pose_size+1, self.H2), 
                nn.ReLU(), 
                nn.Linear(self.H2, self.H2), 
                nn.ReLU(),
                nn.Linear(self.H2, 2),
                nn.Sigmoid(),
            )
            self.pose_regressor = nn.Sequential(
                nn.Linear(self.H, self.H2), 
                nn.ReLU(), 
                nn.Linear(self.H2, self.H2), 
                nn.ReLU(),
                nn.Linear(self.H2, self.pose_size),
            )
        else:
            # 直接位姿回归器
            self.pose_regressor = nn.Sequential(
                nn.Linear(self.H, self.H2), 
                nn.ReLU(), 
                nn.Linear(self.H2, self.H2), 
                nn.ReLU(),
                nn.Linear(self.H2, self.pose_size),
            )

    def update_intrinsics(self, input_shape, intrinsics):
        """更新相机内参以适应特征分辨率"""
        sizey, sizex = self.feature_resolution
        scalex = sizex / input_shape[-1]
        scaley = sizey / input_shape[-2]
        xidx = np.array([0,2])
        yidx = np.array([1,3])
        intrinsics[:,:,xidx] = scalex * intrinsics[:,:,xidx]
        intrinsics[:,:,yidx] = scaley * intrinsics[:,:,yidx]
        return intrinsics

    def extract_features(self, images, intrinsics=None):
        """运行特征提取网络"""
        # 图像预处理
        images = images[:, :, [2,1,0]] / 255.0  # BGR to RGB and normalize
        mean = torch.as_tensor([0.485, 0.456, 0.406], device=images.device)
        std = torch.as_tensor([0.229, 0.224, 0.225], device=images.device)
        images = images.sub_(mean[:, None, None]).div_(std[:, None, None])

        # 更新内参
        if intrinsics is not None:
            intrinsics = self.update_intrinsics(images.shape, intrinsics)

        # ResNet特征提取
        input_images = self.flatten(images)
        input_images = F.interpolate(input_images, size=224)  # 调整到ResNet输入尺寸

        x = self.resnet.conv1(input_images)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x) 
        x = self.resnet.layer1(x)  # 64, 56, 56
        x = self.resnet.layer2(x)  # 128, 28, 28       
        
        # 最终卷积层
        x = self.extractor_final_conv(x)  # 192, 24, 24 

        # 重塑为patch序列
        x = x.reshape([input_images.shape[0], -1, self.num_patches])
        if self.fusion_transformer is None:
            features = x[:,:self.total_num_features//2]
        else:
            features = x[:,:self.total_num_features]
        features = features.permute([0,2,1])

        return features, intrinsics

    def forward(self, images, intrinsics=None, inference=False, loftr_num_corr=None, loftr_preds=None):
        """估计两帧之间的SE3位姿"""
        # 特征提取
        features, intrinsics = self.extract_features(images, intrinsics)
        B, _, _, _, _ = images.shape

        # Fusion Transformer处理（如果启用）
        if self.fusion_transformer is not None:
            x = features[:,:,:self.total_num_features]
            x = self.fusion_transformer.patch_embed(x)
            x = x + self.fusion_transformer.pos_embed
            x = self.fusion_transformer.pos_drop(x)

            for layer in range(self.transformer_depth):
                x = self.fusion_transformer.blocks[layer](x, intrinsics=intrinsics)

            features = self.fusion_transformer.norm(x)
        else:
            # 普通池化处理
            reshaped_features = features.reshape([-1,self.feature_resolution[0],self.feature_resolution[1],self.total_num_features])
            features = self.pool_transformer_output(reshaped_features.permute(0,3,1,2))

        # LoFTR门控位姿回归（如果启用）
        if self.use_loftr_gating:
            if self.use_normalized_6d:
                loftr_preds_6d = compute_normalized_6d(loftr_preds.float(), self.global_pose_mean, self.global_pose_std)
            else:
                loftr_preds_6d = compute_6d(loftr_preds.float())
            
            # 归一化对应点数量
            num_correspondences = loftr_num_corr.detach().float().unsqueeze(1) / 500
            
            # 拼接LoFTR预测和对应点数量
            loftr_preds_6d = torch.cat([loftr_preds_6d, num_correspondences],dim=-1)
            pred_reg_6d = self.pose_regressor(features.reshape([B, -1]))
            feats_preds = torch.cat([features.reshape([B, -1]), pred_reg_6d, loftr_preds_6d],dim=-1)
            pred_RT_wt = self.moe_predictor(feats_preds)

            # 混合预测：使用门控权重混合LoFTR预测和回归预测
            pred_T = pred_RT_wt[...,:1] * pred_reg_6d[...,:3] + (1-pred_RT_wt[...,:1]) * loftr_preds_6d[...,:3]
            pred_R = pred_RT_wt[...,1:] * pred_reg_6d[...,3:] + (1-pred_RT_wt[...,1:]) * loftr_preds_6d[...,3:-1]
            pose_preds = torch.cat([pred_T, pred_R], dim=-1)
        else:
            # 直接位姿回归
            pose_preds = self.pose_regressor(features.reshape([B, -1]))
        
        # 处理预测结果
        rot_preds_6d = pose_preds[:, 3:]
        tran_preds = pose_preds[:, :3]
        
        # 反归一化（如果使用了归一化）
        if self.use_normalized_6d:
            preds_6d_unnorm = rot_preds_6d*self.global_pose_std[3:]+self.global_pose_mean[3:]
            tran_preds_unnorm = tran_preds * self.global_pose_std[:3] + self.global_pose_mean[:3]
        else:
            preds_6d_unnorm = rot_preds_6d
            tran_preds_unnorm = tran_preds
        
        # 从6D表示计算旋转矩阵
        rot_preds_mtx = compute_rotation_matrix_from_ortho6d(preds_6d_unnorm)
        rot_preds = compute_pose_from_rotation_matrix(self.T_pose, rot_preds_mtx)
        
        return tran_preds_unnorm, rot_preds, rot_preds_mtx, rot_preds_6d
