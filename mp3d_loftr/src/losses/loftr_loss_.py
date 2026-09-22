# loftr_loss.py - 损失函数详细注释

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from loguru import logger

pose_mean_6d = torch.tensor([-0.34898765,  0.17085525, -0.87944315, 0.50275223, 0.03533648, -0.18179045, -0.03533648, 0.98189617, 0.09313615])
pose_std_6d = torch.tensor([1.94014405, 0.36770130 , 1.88317520, 0.51837117, 0.12717603, 0.65426397, 0.12717603, 0.0188729, 0.09709263])

def rotation_6d_to_matrix(d6):
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalisation per Section B of [1].
    Args:
        d6: 6D rotation representation, of size (*, 6)
    Returns:
        batch of rotation matrices of size (*, 3, 3)
    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)

def matrix_to_rotation_6d(pose_mtx):
    return pose_mtx[..., :2, :].clone().reshape(*pose_mtx.size()[:-2], 6)

def compute_normalized_6d(pose_mtx):
    pose_6d = matrix_to_rotation_6d(pose_mtx[...,:3,:3])
    tr = pose_mtx[...,:3,3]
    pose_6d_norm = (torch.cat([tr,pose_6d],dim=-1) - pose_mean_6d.to(device=pose_6d.device)) / pose_std_6d.to(device=pose_6d.device)
    
    return pose_6d_norm

class LoFTRLoss(nn.Module):
    """FAR方法的损失函数
    
    这个类实现了FAR方法中使用的多种损失函数，包括：
    1. 特征匹配损失（coarse和fine级别）
    2. 几何损失（极线误差、重投影误差）
    3. 位姿损失（旋转、平移、尺度）
    4. 门控损失（用于动态权重学习）
    """
    
    def __init__(self, config):
        """初始化损失函数
        
        Args:
            config: 配置对象
        """
        super().__init__()
        
        # 保存配置
        self.config = config
        
        # 设置损失函数类型和权重
        self.coarse_type = config["loftr"]["loss"]["coarse_type"]
        self.coarse_weight = config["loftr"]["loss"]["coarse_weight"]
        self.fine_type = config["loftr"]["loss"]["fine_type"]
        self.fine_weight = config["loftr"]["loss"]["fine_weight"]
        
        # 设置几何损失权重
        self.rt_weight_rot = config["loftr"]["loss"]["rt_weight_rot"]
        self.rt_weight_tr = config["loftr"]["loss"]["rt_weight_tr"]
        self.scale_weight = config["loftr"]["loss"]["scale_weight"]
        
        # 设置focal损失参数
        self.focal_alpha = config["loftr"]["loss"]["focal_alpha"]
        self.focal_gamma = config["loftr"]["loss"]["focal_gamma"]
        self.pos_weight = config["loftr"]["loss"]["pos_weight"]
        self.neg_weight = config["loftr"]["loss"]["neg_weight"]
        
        # 设置fine损失参数
        self.fine_correct_thr = config["loftr"]["loss"]["fine_correct_thr"]
        
        # 设置门控参数
        self.use_moe = config["loftr"]["regress"]["use_simple_moe"]
        self.use_2wt = config["loftr"]["regress"]["use_2wt"]
        self.use_5050_weight = config["loftr"]["regress"]["use_5050_weight"]
        self.use_1wt = config["loftr"]["regress"]["use_1wt"]
        self.scale_8pt = config["loftr"]["regress"]["scale_8pt"]
        self.save_mlp_feats = config["loftr"]["regress"]["save_mlp_feats"]
        self.save_gating_weights = config["loftr"]["regress"]["save_gating_weights"]
        self.use_pos_embedding = config["loftr"]["regress"]["use_pos_embedding"]
        self.regress_use_num_corr = config["loftr"]["regress"]["regress_use_num_corr"]
        
        # 设置其他参数
        self.use_l1_rt_loss = config["loftr"]["loss"]["use_l1_rt_loss"]
        self.predict_translation_scale = config["loftr"]["predict_translation_scale"]
        self.max_scale_loss = config["trainer"]["max_scale_loss"]
        
        logger.info(f"初始化损失函数：coarse_type={self.coarse_type}, fine_type={self.fine_type}")

    def forward(self, pred, batch):
        """计算损失
        
        Args:
            pred: 模型预测结果
            batch: 真实标签数据
            
        Returns:
            dict: 各种损失值的字典
        """
        loss_dict = {}
        
        # 1. 计算特征匹配损失
        if self.coarse_weight > 0:
            coarse_loss = self.compute_coarse_loss(pred, batch)
            loss_dict["coarse_loss"] = coarse_loss
            
        if self.fine_weight > 0:
            fine_loss = self.compute_fine_loss(pred, batch)
            loss_dict["fine_loss"] = fine_loss
            
        # 2. 计算几何损失
        if self.rt_weight_rot > 0 or self.rt_weight_tr > 0:
            rt_loss = self.compute_rt_loss(pred, batch)
            if self.rt_weight_rot > 0:
                loss_dict["rt_rot_loss"] = rt_loss["rot_loss"]
            if self.rt_weight_tr > 0:
                loss_dict["rt_tr_loss"] = rt_loss["tr_loss"]
                
        # 3. 计算尺度损失
        if self.scale_weight > 0:
            scale_loss = self.compute_scale_loss(pred, batch)
            loss_dict["scale_loss"] = scale_loss
            
        # 4. 计算门控损失
        if self.use_moe:
            moe_loss = self.compute_moe_loss(pred, batch)
            loss_dict["moe_loss"] = moe_loss
            
        # 5. 计算总损失
        total_loss = 0
        for key, loss in loss_dict.items():
            if loss is not None:
                total_loss += loss
        
        loss_dict["total_loss"] = total_loss
        
        return loss_dict

    def compute_coarse_loss(self, pred, batch):
        """计算coarse级别的特征匹配损失
        
        Args:
            pred: 模型预测结果
            batch: 真实标签数据
            
        Returns:
            torch.Tensor: coarse损失值
        """
        # 获取预测和真实的coarse对应关系
        pred_coarse = pred["conf_matrix"]
        gt_coarse = batch["gt_coarse"]
        
        # 根据损失类型计算损失
        if self.coarse_type == "focal":
            # 使用focal loss
            loss = self.focal_loss(pred_coarse, gt_coarse)
        elif self.coarse_type == "cross_entropy":
            # 使用交叉熵损失
            loss = F.binary_cross_entropy_with_logits(pred_coarse, gt_coarse)
        else:
            raise ValueError(f"未知的coarse损失类型：{self.coarse_type}")
        
        return loss * self.coarse_weight

    def compute_fine_loss(self, pred, batch):
        """计算fine级别的特征匹配损失
        
        Args:
            pred: 模型预测结果
            batch: 真实标签数据
            
        Returns:
            torch.Tensor: fine损失值
        """
        # 获取预测和真实的fine对应关系
        pred_fine = pred["fine0"]
        gt_fine = batch["gt_fine"]
        
        # 过滤有效的fine对应点
        valid_mask = (gt_fine.abs().sum(dim=-1) > 0) & (gt_fine.abs().sum(dim=-1) < self.fine_correct_thr)
        
        if self.fine_type == "l2_with_std":
            # 使用L2损失，考虑标准差
            loss = self.l2_loss_with_std(pred_fine[valid_mask], gt_fine[valid_mask])
        elif self.fine_type == "l2":
            # 使用简单的L2损失
            loss = F.mse_loss(pred_fine[valid_mask], gt_fine[valid_mask])
        else:
            raise ValueError(f"未知的fine损失类型：{self.fine_type}")
        
        return loss * self.fine_weight

    def compute_rt_loss(self, pred, batch):
        """计算旋转和平移损失
        
        Args:
            pred: 模型预测结果
            batch: 真实标签数据
            
        Returns:
            dict: 旋转和平移损失
        """
        # 获取预测和真实的位姿
        pred_rt = pred["loftr_rt"]
        gt_rt = batch["T_0to1"]
        
        # 计算旋转误差
        if self.use_l1_rt_loss:
            # 使用L1损失
            rot_loss = F.l1_loss(pred_rt[...,:3,:3], gt_rt[...,:3,:3])
        else:
            # 使用角度损失
            rot_loss = self.angle_loss(pred_rt[...,:3,:3], gt_rt[...,:3,:3])
            
        # 计算平移误差
        tr_loss = F.mse_loss(pred_rt[...,:3,3], gt_rt[...,:3,3])
        
        return {
            "rot_loss": rot_loss * self.rt_weight_rot,
            "tr_loss": tr_loss * self.rt_weight_tr
        }

    def compute_scale_loss(self, pred, batch):
        """计算尺度损失
        
        Args:
            pred: 模型预测结果
            batch: 真实标签数据
            
        Returns:
            torch.Tensor: 尺度损失值
        """
        # 获取预测和真实的尺度
        pred_scale = pred["scale"]
        gt_scale = batch["gt_scale"]
        
        # 限制尺度损失的最大值
        scale_loss = F.mse_loss(pred_scale, gt_scale)
        scale_loss = torch.min(scale_loss, torch.tensor(self.max_scale_loss).to(scale_loss.device))
        
        return scale_loss * self.scale_weight

    def compute_moe_loss(self, pred, batch):
        """计算门控损失（Mixture of Experts）
        
        Args:
            pred: 模型预测结果
            batch: 真实标签数据
            
        Returns:
            torch.Tensor: 门控损失值
        """
        # 获取门控权重和预测结果
        gating_weights = pred["gating_weights"]
        pred_rt = pred["loftr_rt"]
        gt_rt = batch["T_0to1"]
        
        # 计算两种预测的误差
        network_error = self.pose_error(pred_rt, gt_rt)
        solver_error = self.pose_error(pred["solver_rt"], gt_rt)
        
        # 根据门控策略计算损失
        if self.use_2wt:
            # 使用两个权重（网络和求解器）
            loss = (gating_weights * network_error + (1 - gating_weights) * solver_error).mean()
        elif self.use_5050_weight:
            # 使用50-50权重
            loss = 0.5 * network_error + 0.5 * solver_error
        elif self.use_1wt:
            # 使用单权重
            loss = gating_weights * network_error
        else:
            raise ValueError("未知的门控策略")
        
        return loss

    def focal_loss(self, pred, target):
        """计算focal loss
        
        Args:
            pred: 预测值
            target: 目标值
            
        Returns:
            torch.Tensor: focal loss值
        """
        # 计算二元交叉熵
        bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        
        # 计算focal权重
        p = torch.sigmoid(pred)
        pt = torch.where(target == 1, p, 1 - p)
        focal_weight = self.focal_alpha * (1 - pt) ** self.focal_gamma
        
        # 应用权重
        loss = (focal_weight * bce_loss).mean()
        
        return loss

    def l2_loss_with_std(self, pred, target):
        """计算带标准差的L2损失
        
        Args:
            pred: 预测值（包含均值和标准差）
            target: 目标值
            
        Returns:
            torch.Tensor: L2损失值
        """
        # 分离均值和标准差
        pred_mean = pred[...,:2]
        pred_std = pred[...,2:]
        
        # 计算L2损失
        l2_loss = F.mse_loss(pred_mean, target)
        
        # 添加标准差惩罚
        std_loss = pred_std.mean()
        
        return l2_loss + 0.1 * std_loss

    def angle_loss(self, pred_rot, gt_rot):
        """计算旋转角度损失
        
        Args:
            pred_rot: 预测的旋转矩阵
            gt_rot: 真实的旋转矩阵
            
        Returns:
            torch.Tensor: 角度损失值
        """
        # 计算旋转矩阵的差异
        diff = pred_rot - gt_rot
        angle = torch.arccos(torch.clamp((torch.trace(torch.matmul(pred_rot.transpose(-2,-1), gt_rot)) - 1) / 2, -1, 1))
        
        return angle.mean()

    def pose_error(self, pred_rt, gt_rt):
        """计算位姿误差
        
        Args:
            pred_rt: 预测的位姿矩阵
            gt_rt: 真实的位姿矩阵
            
        Returns:
            torch.Tensor: 位姿误差
        """
        # 计算旋转误差
        rot_error = self.angle_loss(pred_rt[...,:3,:3], gt_rt[...,:3,:3])
        
        # 计算平移误差
        tr_error = F.mse_loss(pred_rt[...,:3,3], gt_rt[...,:3,3])
        
        return rot_error + tr_error
