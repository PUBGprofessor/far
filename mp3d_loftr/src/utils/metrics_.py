# metrics.py - 评估指标详细注释

import torch
import numpy as np
from loguru import logger

def compute_symmetrical_epipolar_errors(pred, batch, epi_err_thr=5e-4):
    """计算对称极线误差
    
    Args:
        pred: 模型预测结果
        batch: 真实标签数据
        epi_err_thr: 极线误差阈值
        
    Returns:
        torch.Tensor: 对称极线误差
    """
    # 获取预测的对应点
    pred_corr = pred["corr"]
    
    # 获取相机内参
    K0 = batch["K0"]
    K1 = batch["K1"]
    
    # 获取真实位姿
    T_0to1 = batch["T_0to1"]
    
    # 计算基础矩阵
    F = compute_fundamental_matrix(K0, K1, T_0to1)
    
    # 计算极线误差
    epi_errs = []
    for i in range(pred_corr.shape[0]):
        # 获取对应点
        corr = pred_corr[i]
        
        # 计算极线
        epiline0 = torch.matmul(F[i], corr[...,:2].unsqueeze(-1))
        epiline1 = torch.matmul(F[i].transpose(-2,-1), corr[...,:2].unsqueeze(-1))
        
        # 计算点到极线的距离
        dist0 = torch.abs(torch.sum(corr[...,:2] * epiline0.squeeze(-1), dim=-1)) / torch.sqrt(torch.sum(epiline0.squeeze(-1)**2, dim=-1))
        dist1 = torch.abs(torch.sum(corr[...,:2] * epiline1.squeeze(-1), dim=-1)) / torch.sqrt(torch.sum(epiline1.squeeze(-1)**2, dim=-1))
        
        # 计算对称极线误差
        epi_err = (dist0 + dist1) / 2
        epi_errs.append(epi_err)
    
    return torch.stack(epi_errs)

def compute_fundamental_matrix(K0, K1, T_0to1):
    """计算基础矩阵
    
    Args:
        K0: 相机0内参
        K1: 相机1内参
        T_0to1: 从相机0到相机1的位姿
        
    Returns:
        torch.Tensor: 基础矩阵
    """
    # 计算本质矩阵
    E = compute_essential_matrix(T_0to1)
    
    # 计算基础矩阵
    F = torch.matmul(torch.matmul(K1.transpose(-2,-1), E), K0)
    
    return F

def compute_essential_matrix(T_0to1):
    """计算本质矩阵
    
    Args:
        T_0to1: 从相机0到相机1的位姿
        
    Returns:
        torch.Tensor: 本质矩阵
    """
    # 提取旋转矩阵和平移向量
    R = T_0to1[...,:3,:3]
    t = T_0to1[...,:3,3]
    
    # 构建反对称矩阵
    tx = torch.zeros_like(R)
    tx[...,:,0] = -t[...,2]
    tx[...,:,1] = t[...,0]
    tx[...,:,2] = -t[...,1]
    tx[...,:,0,1] = t[...,2]
    tx[...,:,0,2] = -t[...,1]
    tx[...,:,1,0] = -t[...,2]
    tx[...,:,1,2] = t[...,0]
    tx[...,:,2,0] = t[...,1]
    tx[...,:,2,1] = -t[...,0]
    
    # 计算本质矩阵
    E = torch.matmul(tx, R)
    
    return E

def compute_pose_errors(pred, batch, epi_err_thr=5e-4):
    """计算位姿误差
    
    Args:
        pred: 模型预测结果
        batch: 真实标签数据
        epi_err_thr: 极线误差阈值
        
    Returns:
        dict: 各种位姿误差指标
    """
    metrics = {}
    
    # 获取预测和真实的位姿
    pred_rt = pred["loftr_rt"]
    gt_rt = batch["T_0to1"]
    
    # 计算旋转误差
    rot_errs = []
    for i in range(pred_rt.shape[0]):
        pred_R = pred_rt[i,...,:3,:3]
        gt_R = gt_rt[i,...,:3,:3]
        
        # 计算旋转矩阵的差异
        diff = pred_R - gt_R
        angle = torch.arccos(torch.clamp((torch.trace(torch.matmul(pred_R.transpose(-2,-1), gt_R)) - 1) / 2, -1, 1))
        
        rot_errs.append(angle.item())
    
    metrics["rot_errs"] = rot_errs
    
    # 计算平移误差
    tr_errs = []
    for i in range(pred_rt.shape[0]):
        pred_t = pred_rt[i,...,3]
        gt_t = gt_rt[i,...,3]
        
        tr_err = torch.norm(pred_t - gt_t).item()
        tr_errs.append(tr_err)
    
    metrics["tr_errs"] = tr_errs
    
    # 计算绝对误差
    abs_rot_errs = [torch.abs(torch.tensor(err)) for err in rot_errs]
    abs_tr_errs = [torch.abs(torch.tensor(err)) for err in tr_errs]
    
    metrics["abs_rot_errs"] = abs_rot_errs
    metrics["abs_tr_errs"] = abs_tr_errs
    
    # 计算中值误差
    metrics["rot_median_err"] = np.median(rot_errs)
    metrics["tr_median_err"] = np.median(tr_errs)
    
    # 计算平均误差
    metrics["rot_mean_err"] = np.mean(rot_errs)
    metrics["tr_mean_err"] = np.mean(tr_errs)
    
    # 计算AUC指标
    auc_metrics = compute_auc_metrics(rot_errs, tr_errs)
    metrics.update(auc_metrics)
    
    return metrics

def compute_auc_metrics(rot_errs, tr_errs):
    """计算AUC指标
    
    Args:
        rot_errs: 旋转误差列表
        tr_errs: 平移误差列表
        
    Returns:
        dict: AUC指标
    """
    auc_metrics = {}
    
    # 旋转误差AUC
    rot_auc_5 = np.mean([err < 5 for err in rot_errs])
    rot_auc_10 = np.mean([err < 10 for err in rot_errs])
    rot_auc_20 = np.mean([err < 20 for err in rot_errs])
    
    auc_metrics["rot_auc_5"] = rot_auc_5
    auc_metrics["rot_auc_10"] = rot_auc_10
    auc_metrics["rot_auc_20"] = rot_auc_20
    
    # 平移误差AUC
    tr_auc_0.1 = np.mean([err < 0.1 for err in tr_errs])
    tr_auc_0.25 = np.mean([err < 0.25 for err in tr_errs])
    tr_auc_0.5 = np.mean([err < 0.5 for err in tr_errs])
    
    auc_metrics["tr_auc_0.1"] = tr_auc_0.1
    auc_metrics["tr_auc_0.25"] = tr_auc_0.25
    auc_metrics["tr_auc_0.5"] = tr_auc_0.5
    
    return auc_metrics

def aggregate_metrics(metrics, epi_err_thr=5e-4):
    """聚合指标
    
    Args:
        metrics: 指标字典
        epi_err_thr: 极线误差阈值
        
    Returns:
        dict: 聚合后的指标
    """
    aggregated = {}
    
    # 基本统计
    aggregated["rot_mean_err"] = np.mean(metrics["rot_errs"])
    aggregated["rot_median_err"] = np.median(metrics["rot_errs"])
    aggregated["tr_mean_err"] = np.mean(metrics["tr_errs"])
    aggregated["tr_median_err"] = np.median(metrics["tr_errs"])
    
    # AUC指标
    auc_keys = [k for k in metrics.keys() if "auc" in k]
    for key in auc_keys:
        aggregated[key] = np.mean(metrics[key])
    
    # 数据集大小
    aggregated["dset size"] = len(metrics["rot_errs"])
    
    return aggregated

def aggregate_metrics_interiornet_streetlearn(metrics, epi_err_thr=5e-4):
    """聚合InteriorNet和StreetLearn指标
    
    Args:
        metrics: 指标字典
        epi_err_thr: 极线误差阈值
        
    Returns:
        dict: 聚合后的指标
    """
    aggregated = {}
    
    # 基本统计
    aggregated["rot_mean_err"] = np.mean(metrics["rot_errs"])
    aggregated["rot_median_err"] = np.median(metrics["rot_errs"])
    aggregated["tr_mean_err"] = np.mean(metrics["tr_errs"])
    aggregated["tr_median_err"] = np.median(metrics["tr_errs"])
    
    # AUC指标
    auc_keys = [k for k in metrics.keys() if "auc" in k]
    for key in auc_keys:
        aggregated[key] = np.mean(metrics[key])
    
    # 数据集大小
    aggregated["dset size"] = len(metrics["rot_errs"])
    
    return aggregated
