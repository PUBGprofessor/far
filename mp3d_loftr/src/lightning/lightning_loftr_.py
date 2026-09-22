# lightning_loftr.py - PyTorch Lightning模型详细注释

from collections import defaultdict
from loguru import logger
from pathlib import Path
import pickle as pkl
import os

import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from matplotlib import pyplot as plt

# 导入FAR核心组件
from src.loftr import LoFTR
from src.baselines.simple_transformer import SimpleTransformer
from src.loftr.utils.supervision import compute_supervision_coarse, compute_supervision_fine, compute_supervision_RT
from src.losses.loftr_loss import LoFTRLoss
from src.optimizers import build_optimizer, build_scheduler
from src.utils.metrics import (
    compute_symmetrical_epipolar_errors,
    compute_pose_errors,
    aggregate_metrics,
    aggregate_metrics_interiornet_streetlearn
)
from src.utils.plotting import make_matching_figures
from src.utils.comm import gather, all_gather
from src.utils.misc import lower_config, flattenList
from src.utils.profiler import PassThroughProfiler


class PL_LoFTR(pl.LightningModule):
    """FAR方法的PyTorch Lightning实现
    
    这个类实现了FAR方法的核心逻辑，包括：
    1. 特征匹配（使用LoFTR或SimpleTransformer）
    2. 位姿求解（使用RANSAC或其他求解器）
    3. 损失计算和优化
    4. 评估和可视化
    """
    
    def __init__(self, config, pretrained_ckpt=None, profiler=None, dump_dir=None, split=None):
        """初始化FAR模型
        
        Args:
            config: 完整配置对象
            pretrained_ckpt: 预训练检查点路径
            profiler: 性能分析器
            dump_dir: 结果保存目录
            split: 数据集分割（train/val/test）
        """
        super().__init__()
        
        # 保存配置和设置日志记录器
        self.config = config  # 完整配置
        _config = lower_config(self.config)  # 转换为小写配置
        self.loftr_cfg = lower_config(_config["loftr"])  # LoFTR配置
        self.profiler = profiler or PassThroughProfiler()  # 性能分析器
        self.n_vals_plot = max(config.TRAINER.N_VAL_PAIRS_TO_PLOT // config.TRAINER.WORLD_SIZE, 1)

        # 初始化特征匹配器
        if config.USE_CORRESPONDENCE_TRANSFORMER:
            _config["loftr"]["part2"] = False
            self.matcher = SimpleTransformer(config=_config["loftr"])  # 简单Transformer匹配器
        else:
            # 使用LoFTR作为匹配器
            self.matcher = LoFTR(config=_config["loftr"])
            
        # 初始化损失函数
        self.loss = LoFTRLoss(_config)

        # 加载预训练权重
        self.pretrained_ckpt = None
        if pretrained_ckpt:
            self.pretrained_ckpt = pretrained_ckpt
            # 加载预训练模型权重
            state_dict = torch.load(pretrained_ckpt, map_location="cpu", weights_only=False)["state_dict"]
            
            # 处理权重名称映射
            if config.USE_CORRESPONDENCE_TRANSFORMER:
                state_dict = {k.replace("matcher.", ""): v for k, v in state_dict.items()}
            
            # 设置严格模式
            strict=True
            if config.STRICT_FALSE:
                strict=False
                
            # 根据训练配置删除某些权重
            if self.config.LOFTR.TRAINING and self.config.LOFTR.USE_MANY_RANSAC_THR:
                # 删除门控预测器的权重
                del state_dict["matcher.loftr_regress.moe_predictor.0.weight"]
                del state_dict["matcher.loftr_regress.moe_predictor.0.bias"]

            if self.config.LOFTR.TRAINING and self.config.LOFTR.REGRESS.USE_1WT:
                # 删除单权重预测器的权重
                del state_dict["matcher.loftr_regress.moe_predictor.4.weight"]
                del state_dict["matcher.loftr_regress.moe_predictor.4.bias"]

            # 加载权重
            self.matcher.load_state_dict(state_dict, strict=strict)
            logger.info(f"加载预训练检查点：{pretrained_ckpt}")
        else:
            print("\n\n\n 警告：未加载预训练检查点！ \n\n\n")

        self.pl_version = config.PL_VERSION

        # 测试相关设置
        self.dump_dir = dump_dir
        self.split = split

    def configure_optimizers(self):
        """配置优化器和学习率调度器"""
        # 构建优化器
        optimizer = build_optimizer(self, self.config)
        # 构建学习率调度器
        scheduler = build_scheduler(self.config, optimizer)
        return [optimizer], [scheduler]
    
    def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
        """学习率调度器步进"""
        scheduler.step(epoch=self.current_epoch)

    def optimizer_step(
            self, epoch, batch_idx, optimizer, optimizer_idx,
            optimizer_closure, on_tpu, using_native_amp, using_lbfgs):
        """优化器步进，包含学习率预热"""
        # 学习率预热
        warmup_step = self.config.TRAINER.WARMUP_STEP
        if self.trainer.global_step < warmup_step:
            if self.config.TRAINER.WARMUP_TYPE == "linear":
                # 线性预热
                base_lr = self.config.TRAINER.WARMUP_RATIO * self.config.TRAINER.TRUE_LR
                lr = base_lr + \
                    (self.trainer.global_step / self.config.TRAINER.WARMUP_STEP) * \
                    abs(self.config.TRAINER.TRUE_LR - base_lr)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr
            elif self.config.TRAINER.WARMUP_TYPE == "constant":
                # 常数预热
                pass
            else:
                raise ValueError(f"未知的学习率预热策略：{self.config.TRAINER.WARMUP_TYPE}")

        # 更新参数
        # 检查梯度是否有效
        valid_gradients = True
        for name, param in self.named_parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                valid_gradients = False
                break

        if valid_gradients:
            optimizer.step(closure=optimizer_closure)
        else:
            print("检测到NaN梯度，跳过优化器步进")

        optimizer.zero_grad()

    def forward(self, batch):
        """前向传播
        
        Args:
            batch: 输入批次数据
            
        Returns:
            dict: 包含预测结果和中间结果的字典
        """
        return self.matcher(batch)

    def training_step(self, batch, batch_idx):
        """训练步
        
        Args:
            batch: 输入批次数据
            batch_idx: 批次索引
            
        Returns:
            dict: 训练损失
        """
        # 前向传播
        pred = self.forward(batch)
        
        # 计算损失
        loss_dict = self.loss(pred, batch)
        
        # 记录训练损失
        self.log_dict({f"train/{k}": v for k, v in loss_dict.items()}, prog_bar=True)
        
        # 总损失
        total_loss = sum(loss_dict.values())
        
        return {"loss": total_loss}

    def validation_step(self, batch, batch_idx):
        """验证步
        
        Args:
            batch: 输入批次数据
            batch_idx: 批次索引
            
        Returns:
            dict: 验证结果
        """
        # 前向传播
        pred = self.forward(batch)
        
        # 计算损失
        loss_dict = self.loss(pred, batch)
        
        # 记录验证损失
        self.log_dict({f"val/{k}": v for k, v in loss_dict.items()}, prog_bar=True)
        
        # 计算位姿误差
        metrics = compute_pose_errors(pred, batch, self.config.TRAINER.EPI_ERR_THR)
        
        # 计算极线误差
        epi_errs = compute_symmetrical_epipolar_errors(pred, batch, self.config.TRAINER.EPI_ERR_THR)
        metrics["epi_errs"] = epi_errs
        
        return {"metrics": metrics, "loss": sum(loss_dict.values())}

    def validation_epoch_end(self, outputs):
        """验证周期结束时的处理"""
        # 聚合所有批次的指标
        _metrics = [o["metrics"] for o in outputs]
        metrics = {k: flattenList(gather(flattenList([_me[k] for _me in _metrics]))) for k in _metrics[0]}
        
        # 聚合损失
        losses = [o["loss"] for o in outputs]
        avg_loss = torch.mean(torch.stack([l for l in losses if l.numel() > 0]))
        self.log("val/loss", avg_loss, prog_bar=True)
        
        # 计算并记录指标
        if "interiornet" in self.config.DSET_NAME or "streetlearn" in self.config.DSET_NAME:
            val_metrics_4tb = aggregate_metrics_interiornet_streetlearn(metrics, self.config.TRAINER.EPI_ERR_THR)
        else:
            val_metrics_4tb = aggregate_metrics(metrics, self.config.TRAINER.EPI_ERR_THR)
        
        # 记录指标
        for key in val_metrics_4tb:
            if "tr" in key or "rot" in key or "pct" in key or "dset size" in key:
                self.log(f"val/{key}", val_metrics_4tb[key], prog_bar=True)
        
        # 绘制错误分布
        if self.trainer.global_rank == 0:
            self.plot_errors(metrics)

    def test_step(self, batch, batch_idx, skip_eval=False):
        """测试步
        
        Args:
            batch: 输入批次数据
            batch_idx: 批次索引
            skip_eval: 是否跳过评估
            
        Returns:
            dict: 测试结果
        """
        # 前向传播
        pred = self.forward(batch)
        
        # 计算损失
        loss_dict = self.loss(pred, batch)
        
        # 计算位姿误差
        if not skip_eval:
            metrics = compute_pose_errors(pred, batch, self.config.TRAINER.EPI_ERR_THR)
            
            # 计算极线误差
            epi_errs = compute_symmetrical_epipolar_errors(pred, batch, self.config.TRAINER.EPI_ERR_THR)
            metrics["epi_errs"] = epi_errs
            
            # 记录指标
            self.log_dict({f"test/{k}": torch.mean(torch.tensor(v)) if isinstance(v, list) else v 
                          for k, v in metrics.items()}, prog_bar=True)
        
        # 构建结果字典
        result = {
            "pred": pred,
            "batch": batch,
            "metrics": metrics if not skip_eval else None,
            "loss": sum(loss_dict.values())
        }
        
        return result

    def test_epoch_end(self, outputs):
        """测试周期结束时的处理"""
        # 聚合所有批次的指标
        _metrics = [o["metrics"] for o in outputs]
        metrics = {k: flattenList(gather(flattenList([_me[k] for _me in _metrics]))) for k in _metrics[0]}

        if self.dump_dir is not None:
            Path(self.dump_dir).mkdir(parents=True, exist_ok=True)
            _dumps = flattenList([o["dumps"] for o in outputs])
            dumps = flattenList(gather(_dumps))
            logger.info(f"预测和评估结果将保存到：{self.dump_dir}")

        # 计算最终指标
        if self.trainer.global_rank == 0:
            if "interiornet" in self.config.DSET_NAME or "streetlearn" in self.config.DSET_NAME:
                val_metrics_4tb = aggregate_metrics_interiornet_streetlearn(metrics, self.config.TRAINER.EPI_ERR_THR)
            else:
                val_metrics_4tb = aggregate_metrics(metrics, self.config.TRAINER.EPI_ERR_THR)
            
            # 打印指标
            for key in val_metrics_4tb:
                if "tr" in key or "rot" in key or "pct" in key or "dset size" in key:
                    print(f"{key} {str(val_metrics_4tb[key])}")
            
            print("")
            for key in val_metrics_4tb:
                if "auc" in key:
                    print(f"{key} {str(val_metrics_4tb[key])}")

            # 保存结果
            if self.config.EXP_NAME != "" or self.pretrained_ckpt is None:
                dirname = os.path.join("logs/tb_logs", self.config.EXP_NAME)
                os.makedirs(dirname, exist_ok=True)
            else:
                dirname = os.path.dirname(os.path.dirname(self.pretrained_ckpt))
            
            # 构建结果文件名
            results_name = "results.txt" 
            if self.config.LOFTR.SOLVER != "ransac":
                results_name = f"results_{self.config.LOFTR.SOLVER}.txt"
            elif self.config.TRAINER.RANSAC_PIXEL_THR != 0.5:
                results_name = f"results_ransac_thr_{self.config.TRAINER.RANSAC_PIXEL_THR}.txt"

            if self.config.CORRESPONDENCES_USE_FIT_ONLY:
                results_name = f"use_fit_only_{results_name}"

            if self.config.MAX_CORRESPONDENCES < 2000:
                results_name = f"max_corrs_{self.config.MAX_CORRESPONDENCES}_{results_name}"

            if self.config.OUTLIER_PCT > 0:
                results_name = f"outlier_pct_{self.config.OUTLIER_PCT}_{results_name}"

            if self.config.NOISE_PIX > 0:
                results_name = f"noise_pix_{self.config.NOISE_PIX}_{results_name}"

            if self.config.MISSING_PCT > 0:
                results_name = f"missing_pct_{self.config.MISSING_PCT}_{results_name}"

            # 保存结果
            with open(os.path.join(dirname, self.split + "_" + results_name), "w") as f:
                for key in val_metrics_4tb:
                    if "tr" in key or "rot" in key or "pct" in key or "dset size" in key:
                        f.write(f"{key} {str(val_metrics_4tb[key])} \n")

                print("")
                for key in val_metrics_4tb:
                    if "auc" in key:
                        f.write(f"{key} {str(val_metrics_4tb[key])} \n")

            # 绘制CDF图
            if True:
                self.plot_errors(metrics)

            # 保存RANSAC后的对应点数量
            if np.array(metrics["inliers"]).dtype != "int64":
                np.save(os.path.join(dirname, self.split+"_num_correspondences_after_ransac.npy"), \
                        np.array([x.sum() for x in metrics["inliers"]]))

                np.save(os.path.join(dirname, self.split+"_num_correspondences_before_ransac.npy"), \
                        np.array([x.shape[0] for x in metrics["inliers"]]))

            if self.config.LOFTR.REGRESS.SAVE_GATING_WEIGHTS:
                np.save(os.path.join(dirname, self.split+"_gating_reg_weights.npy"), \
                        np.array([x.cpu().numpy() for x in metrics["gating_reg_weights"]]))

            if self.dump_dir is not None:
                np.save(Path(self.dump_dir) / "LoFTR_pred_eval", dumps)

    def plot_errors(self, metrics):
        """绘制误差分布图"""
        # 绘制旋转误差CDF
        plt.figure(figsize=(10, 6))
        rot_errs = [r.item() for r in metrics["rot_errs"]]
        sorted_errs = sorted(rot_errs)
        y_vals = np.arange(len(sorted_errs)) / len(sorted_errs)
        plt.plot(sorted_errs, y_vals)
        plt.xlabel("旋转误差 (度)")
        plt.ylabel("CDF")
        plt.title("旋转误差分布")
        plt.grid(True)
        
        # 保存图像
        if self.trainer.global_rank == 0:
            save_dir = Path("logs/tb_logs") / self.config.EXP_NAME / "plots"
            save_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_dir / f"{self.split}_rot_err_cdf.png")
            plt.close()

        # 绘制平移误差CDF
        plt.figure(figsize=(10, 6))
        tr_errs = [t.item() for t in metrics["tr_errs"]]
        sorted_errs = sorted(tr_errs)
        y_vals = np.arange(len(sorted_errs)) / len(sorted_errs)
        plt.plot(sorted_errs, y_vals)
        plt.xlabel("平移误差 (米)")
        plt.ylabel("CDF")
        plt.title("平移误差分布")
        plt.grid(True)
        
        # 保存图像
        if self.trainer.global_rank == 0:
            plt.savefig(save_dir / f"{self.split}_tr_err_cdf.png")
            plt.close()
