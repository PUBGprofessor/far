# train.py - 训练脚本详细注释

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # 设置OpenBLAS线程数为1，避免并行计算问题
import numpy as np

import math
import argparse
import pprint
from distutils.util import strtobool
from pathlib import Path
from loguru import logger as loguru_logger

import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.plugins import DDPPlugin

from src.config.default import get_cfg_defaults
from src.utils.misc import get_rank_zero_only_logger, setup_gpus
from src.utils.profiler import build_profiler
from src.lightning.data import Mp3dDataModule, Mp3dLightDataModule
from src.lightning.lightning_loftr import PL_LoFTR

loguru_logger = get_rank_zero_only_logger(loguru_logger)
import warnings 
warnings.filterwarnings("error", category=RuntimeWarning)  # 忽略运行时警告

import sys
import os
sys.path.append("third_party/prior_ransac")  # 添加prior_ransac模块到Python路径

def parse_args():
    """解析命令行参数"""
    # 初始化自定义解析器，添加到pl.Trainer解析器中
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # 数据配置路径
    parser.add_argument(
        "data_cfg_path", type=str, help="数据配置文件路径")
    parser.add_argument(
        "main_cfg_path", type=str, help="主配置文件路径")
    
    # 实验配置
    parser.add_argument(
        "--exp_name", type=str, default="default_exp_name", help="实验名称")
    parser.add_argument(
        "--batch_size", type=int, default=4, help="每个GPU的批大小")
    parser.add_argument(
        "--num_workers", type=int, default=4, help="数据加载器的工作进程数")
    parser.add_argument(
        "--pin_memory", type=lambda x: bool(strtobool(x)),
        nargs="?", default=True, help="是否将数据加载到固定内存")
    
    # 模型配置
    parser.add_argument(
        "--ckpt_path", type=str, default=None,
        help="预训练检查点路径，有助于使用预训练的仅coarse LoFTR")
    parser.add_argument(
        "--disable_ckpt", action="store_true",
        help="禁用检查点保存（用于调试）")
    
    # 性能配置
    parser.add_argument(
        "--profiler_name", type=str, default=None,
        help="选项：[inference, pytorch]，或留空")
    parser.add_argument(
        "--parallel_load_data", action="store_true",
        help="使用多个进程加载数据集")
    parser.add_argument(
        "--find_unused_parameters", action="store_true", help="如果为true，调试DDP")
    
    # 损失函数权重配置
    parser.add_argument(
        "--rt_weight_rot", default=0.0, type=float, help="旋转损失的权重")
    parser.add_argument(
        "--rt_weight_tr", default=0.0, type=float, help="平移损失的权重")
    parser.add_argument(
        "--scale_weight", default=0.0, type=float, help="尺度损失的权重")
    parser.add_argument(
        "--fine_weight", default=1.0, type=float, help="fine对应点损失的权重")
    parser.add_argument(
        "--coarse_weight", default=1.0, type=float, help="coarse对应点损失的权重")
    
    # 训练策略配置
    parser.add_argument(
        "--predict_translation_scale", default=False, action="store_true", help="预测平移尺度")
    parser.add_argument(
        "--use_40pct_dset", default=False, action="store_true", help="使用40%的数据集")
    parser.add_argument(
        "--ckpt_every_n_epochs", type=int, default=1, help="每多少个epoch保存一次检查点")
    parser.add_argument(
        "--num_warmup_steps", type=int, default=None, help="预热步数")
    parser.add_argument(
        "--lr", type=float, default=None, help="学习率")
    parser.add_argument(
        "--use_one_cycle_lr", default=False, action="store_true", help="使用单周期学习率")
    parser.add_argument(
        "--use_pred_corr", default=False, action="store_true", help="使用预测的对应点输入到corr transformer")
    parser.add_argument(
        "--correspondences_use_fit_only", action="store_true", help="如果为true，只训练有>5个地面真值对应点的情况")
    parser.add_argument(
        "--regress_rt", default=False, action="store_true", help="使用transformer回归RT（无尺度）")    
    parser.add_argument(
        "--regress_loftr_layers", default=0, type=int, help="使用LoFTR + EMM回归RT（无尺度）")    
    parser.add_argument(
        "--use_old_slow_warmup", default=False, action="store_true", help="如果为true，使用旧的慢速预热")
    parser.add_argument(
        "--thr", default=0.2, type=float, help="coarse对应点中conf_matrix的阈值")
    parser.add_argument(
        "--max_scale_loss", default=999, type=float, help="尺度损失的最大值")
    parser.add_argument(
        "--use_pos_embedding", action="store_true", help="如果为true，emm位姿回归器使用位置编码")
    parser.add_argument(
        "--num_coarse_loftr_layers", default=4, type=int, help="coarse loftr层数")
    parser.add_argument(
        "--loss_fine_type", default="l2_with_std", type=str, help="fine损失类型")
    parser.add_argument(
        "--use_l1_rt_loss", default=False, action="store_true", help="使用L1 RT损失")
    parser.add_argument(
        "--use_correspondence_transformer", default=False, action="store_true", help="使用对应点transformer")
    parser.add_argument(
        "--max_correspondences", default=2000, type=int, help="最大对应点数量")
    parser.add_argument(
        "--outlier_pct", default=0, type=float, help="离群点百分比")
    parser.add_argument(
        "--noise_pix", default=0, type=float, help="噪声像素数")
    parser.add_argument(
        "--missing_pct", default=0, type=float, help="缺失百分比")
    parser.add_argument(
        "--strict_false", default=False, action="store_true", help="严格模式")
    parser.add_argument(
        "--no_save_feats", default=False, action="store_true", help="不保存特征")
    parser.add_argument(
        "--no_save_preds", default=False, action="store_true", help="不保存预测")
    parser.add_argument(
        "--no_save_numcorr", default=False, action="store_true", help="不保存对应点数量")
    parser.add_argument(
        "--solver", default="prior_ransac", type=str, help="求解器类型")
    parser.add_argument(
        "--use_large_dset", default=False, action="store_true", help="使用大数据集")
    parser.add_argument(
        "--use_pred_corr", default=False, action="store_true", help="使用预测的对应点")
    
    # 添加PyTorch Lightning参数
    parser = pl.Trainer.add_argparse_args(parser)
    return parser.parse_args()

def main():
    """主训练函数"""
    # 解析参数
    args = parse_args()
    
    # 初始化默认配置并合并主配置和数据配置
    config = get_cfg_defaults()
    config.merge_from_file(args.main_cfg_path)
    config.merge_from_file(args.data_cfg_path)
    pl.seed_everything(config.TRAINER.SEED)  # 设置随机种子以保证可重复性

    # 设置训练参数
    config.TRAINER.CANONICAL_BS = args.batch_size
    config.TRAINER.NUM_WORKERS = args.num_workers
    config.TRAINER.PIN_MEMORY = args.pin_memory
    config.TRAINER.DISABLE_CKPT = args.disable_ckpt
    config.TRAINER.PROFILER_NAME = args.profiler_name
    config.TRAINER.PARALLEL_LOAD_DATA = args.parallel_load_data
    config.TRAINER.FIND_UNUSED_PARAMETERS = args.find_unused_parameters
    config.TRAINER.CKPT_EVERY_N_EPOCHS = args.ckpt_every_n_epochs
    config.TRAINER.NUM_WARMUP_STEPS = args.num_warmup_steps
    config.TRAINER.TRUE_LR = args.lr
    config.TRAINER.USE_ONE_CYCLE_LR = args.use_one_cycle_lr
    config.TRAINER.USE_PRED_CORR = args.use_pred_corr
    config.TRAINER.CORRESPONDENCES_USE_FIT_ONLY = args.correspondences_use_fit_only
    config.TRAINER.REGRESS_RT = args.regress_rt
    config.TRAINER.REGRESS_LOFTR_LAYERS = args.regress_loftr_layers
    config.TRAINER.USE_OLD_SLOW_WARMUP = args.use_old_slow_warmup
    config.TRAINER.THR = args.thr
    config.TRAINER.MAX_SCALE_LOSS = args.max_scale_loss
    config.TRAINER.USE_POS_EMBEDDING = args.use_pos_embedding
    config.LOFTR.NUM_COARSE_LOFTR_LAYERS = args.num_coarse_loftr_layers
    config.LOFTR.LOSS.FINE_TYPE = args.loss_fine_type
    config.LOFTR.LOSS.USE_L1_RT_LOSS = args.use_l1_rt_loss
    config.USE_CORRESPONDENCE_TRANSFORMER = args.use_correspondence_transformer
    config.CORRESPONDENCES_USE_FIT_ONLY = args.correspondences_use_fit_only
    config.MAX_CORRESPONDENCES = args.max_correspondences
    config.OUTLIER_PCT = args.outlier_pct
    config.NOISE_PIX = args.noise_pix
    config.MISSING_PCT = args.missing_pct
    config.STRICT_FALSE = args.strict_false
    config.NO_SAVE_FEATS = args.no_save_feats
    config.NO_SAVE_PREDS = args.no_save_preds
    config.NO_SAVE_NUMCORR = args.no_save_numcorr
    config.LOFTR.SOLVER = args.solver
    config.USE_LARGE_DSET = args.use_large_dset
    config.USE_PRED_CORR = args.use_pred_corr
    config.PL_VERSION = pl.__version__

    # 设置损失权重
    config.LOFTR.LOSS.RT_WEIGHT_ROT = args.rt_weight_rot
    config.LOFTR.LOSS.RT_WEIGHT_TR = args.rt_weight_tr
    config.LOFTR.LOSS.SCALE_WEIGHT = args.scale_weight
    config.LOFTR.LOSS.FINE_WEIGHT = args.fine_weight
    config.LOFTR.LOSS.COARSE_WEIGHT = args.coarse_weight
    config.LOFTR.PREDICT_TRANSLATION_SCALE = args.predict_translation_scale

    # 根据参数调整模型配置
    if args.solver == "prior_ransac":
        import os
        os.environ["OPENBLAS_NUM_THREADS"] = "1"  # 优化RANSAC计算

    if args.num_coarse_loftr_layers < 4:
        config.LOFTR.COARSE.LAYER_NAMES = ["self", "cross"] * args.num_coarse_loftr_layers

    if args.regress_loftr_layers > 1:
        config.LOFTR.REGRESS.LAYER_NAMES = ["self", "cross"] * args.regress_loftr_layers

    # 设置学习率调度器
    if args.use_one_cycle_lr:
        config.TRAINER.SCHEDULER = "OneCycleLR"
        if args.use_large_dset:
            config.TRAINER.STEPS = args.max_epochs * 8740
        elif args.use_40pct_dset:
            config.TRAINER.STEPS = args.max_epochs * 1700
        else:
            config.TRAINER.STEPS = args.max_epochs * 3665
        print(f"使用 {config.TRAINER.STEPS} 步")
        config.TRAINER.PCT_WARMUP = config.TRAINER.WARMUP_STEP / config.TRAINER.STEPS
        config.TRAINER.WARMUP_TYPE = "constant"

    if args.use_old_slow_warmup:
        config.TRAINER.WARMUP_STEP /= _scaling

    print(f"使用 {config.TRAINER.WARMUP_STEP} 预热步")

    # 初始化模型
    profiler = build_profiler(args.profiler_name)
    model = PL_LoFTR(config, pretrained_ckpt=args.ckpt_path, profiler=profiler)
    loguru_logger.info(f"LoFTR LightningModule初始化完成！")

    # 初始化数据模块
    if config.USE_CORRESPONDENCE_TRANSFORMER:
        data_module = Mp3dLightDataModule(args, config)
    else:
        data_module = Mp3dDataModule(args, config)
        loguru_logger.info(f"LoFTR Mp3d DataModule初始化完成！")
    
    # 设置TensorBoard日志记录器
    logger = TensorBoardLogger(save_dir="logs/tb_logs", name=args.exp_name, default_hp_metric=False)
    ckpt_dir = Path(logger.log_dir) / "checkpoints"
    
    print("准备设置模型")

    # 设置模型检查点回调
    if pl.__version__ == "1.6.0":
        ckpt_callback = ModelCheckpoint(monitor="rot_mean_err", verbose=True, save_top_k=5, mode="min",
                                        save_last=True,
                                        dirpath=str(ckpt_dir),
                                        filename="{epoch}-{rot_mean_err:.2f}-{rot_median_err:.2f}-{tr_abs_mean_err:.2f}")
    else:
        ckpt_callback = ModelCheckpoint(monitor="rot_mean_err", verbose=True, save_top_k=5, mode="min",
                                        save_last=True,
                                        dirpath=str(ckpt_dir),
                                        filename="{epoch}-{rot_mean_err:.2f}-{rot_median_err:.2f}-{tr_abs_mean_err:.2f}",
                                        every_n_val_epochs=args.ckpt_every_n_epochs)

    print("准备设置学习率监控")

    # 设置学习率监控器
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks = [lr_monitor]
    if not args.disable_ckpt:
        callbacks.append(ckpt_callback)
    
    print("准备初始化训练器")

    # 初始化PyTorch Lightning训练器
    if pl.__version__ == "1.6.0":
        # Lightning Trainer (旧版本)
        trainer = pl.Trainer.from_argparse_args(
            args,
            gradient_clip_val=config.TRAINER.GRADIENT_CLIPPING,
            callbacks=callbacks,
            logger=logger,
            sync_batchnorm=config.TRAINER.WORLD_SIZE > 0,
            replace_sampler_ddp=False,  # 使用自定义采样器
            profiler=profiler)        
    else:
        # Lightning Trainer (新版本)
        trainer = pl.Trainer.from_argparse_args(
            args,
            plugins=DDPPlugin(find_unused_parameters=args.find_unused_parameters,
                            num_nodes=args.num_nodes,
                            sync_batchnorm=config.TRAINER.WORLD_SIZE > 0),
            gradient_clip_val=config.TRAINER.GRADIENT_CLIPPING,
            callbacks=callbacks,
            logger=logger,
            sync_batchnorm=config.TRAINER.WORLD_SIZE > 0,
            replace_sampler_ddp=False,  # 使用自定义采样器
            reload_dataloaders_every_epoch=False,  # 避免重复样本！
            weights_summary="full",
            profiler=profiler)
    loguru_logger.info(f"Trainer初始化完成！")
    loguru_logger.info(f"开始训练！")
    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    main()
