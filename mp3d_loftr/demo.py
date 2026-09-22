# demo.py - 演示脚本详细注释

import pytorch_lightning as pl
import argparse

from src.config.default import get_cfg_defaults

from src.lightning.lightning_loftr import PL_LoFTR
import sys
import torch
import cv2
import numpy as np
sys.path.append("third_party/prior_ransac")  # 添加prior_ransac模块到Python路径

def parse_args():
    """解析命令行参数"""
    # 初始化自定义解析器，添加到pl.Trainer解析器中
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # 配置文件路径
    parser.add_argument(
        "data_cfg_path", type=str, help="数据配置文件路径")
    parser.add_argument(
        "main_cfg_path", type=str, help="主配置文件路径")
    
    # 模型配置
    parser.add_argument(
        "--ckpt_path", type=str, default="", help="检查点路径")
    parser.add_argument(
        "--exp_name", type=str, default="", help="实验名称，如果未提供则使用检查点名称")
    parser.add_argument(
        "--batch_size", type=int, default=1, help="每个GPU的批大小")
    parser.add_argument(
        "--num_workers", type=int, default=2, help="数据加载器的工作进程数")
    
    # 相机内参配置
    parser.add_argument(
        "--fx", type=float, default=517.97, help="焦距x")
    parser.add_argument(
        "--fy", type=float, default=517.97, help="焦距y")
    parser.add_argument(
        "--cx", type=float, default=320, help="主点x")
    parser.add_argument(
        "--cy", type=float, default=240, help="主点y")
    
    # 图像尺寸配置
    parser.add_argument(
        "--h", type=int, default=640, help="图像高度")
    parser.add_argument(
        "--w", type=int, default=480, help="图像宽度")
    
    # 输入图像路径
    parser.add_argument(
        "--img_path0", type=str, default="test", help="图像0路径")
    parser.add_argument(
        "--img_path1", type=str, default="test", help="图像1路径")
    
    # 添加PyTorch Lightning参数
    parser = pl.Trainer.add_argparse_args(parser)
    return parser.parse_args()

if __name__ == "__main__":
    # 解析参数
    args = parse_args()

    # 初始化默认配置并合并主配置和数据配置
    config = get_cfg_defaults()
    config.merge_from_file(args.main_cfg_path)
    config.merge_from_file(args.data_cfg_path)
    pl.seed_everything(config.TRAINER.SEED)  # 设置随机种子以保证可重复性

    # 设置不相关的参数（用于演示）
    config.LOFTR.FEAT_SIZE = 128
    config.LOFTR.NUM_ENCODER_LAYERS = 6
    config.LOFTR.NUM_BANDS = 10
    config.LOFTR.CORRESPONDENCE_TRANSFORMER_USE_POS_ENCODING = False
    config.LOFTR.CORRESPONDENCE_TF_USE_GLOBAL_AVG_POOLING = False
    config.LOFTR.CORRESPONDENCE_TF_USE_FEATS = False
    config.LOFTR.CTF_CAT_FEATS = False
    config.LOFTR.NUM_HEADS = 8
    config.USE_PRED_CORR = False
    config.STRICT_FALSE = False
    config.EVAL_FIT_ONLY = False

    # 设置相关参数（用于演示）
    config.LOFTR.PREDICT_TRANSLATION_SCALE = False
    config.LOFTR.REGRESS_RT = True
    config.LOFTR.REGRESS_LOFTR_LAYERS = 1
    config.LOFTR.REGRESS.USE_POS_EMBEDDING = True
    config.LOFTR.REGRESS.REGRESS_USE_NUM_CORRES = True
    config.LOFTR.COARSE.LAYER_NAMES = ["self", "cross"] * 3

    # 设置配置参数
    config.LOFTR.FROM_SAVED_PREDS = None
    config.LOFTR.SOLVER = "prior_ransac"
    config.LOFTR.USE_MANY_RANSAC_THR = True

    config.LOFTR.FINE_PRED_STEPS = 2
    config.LOFTR.REGRESS.SAVE_MLP_FEATS = False
    config.LOFTR.REGRESS.USE_SIMPLE_MOE = True
    config.LOFTR.REGRESS.USE_2WT = True
    config.LOFTR.REGRESS.USE_5050_WEIGHT = False
    config.LOFTR.REGRESS.USE_1WT = False
    config.LOFTR.REGRESS.SCALE_8PT = True
    config.LOFTR.REGRESS.SAVE_GATING_WEIGHTS = False
    config.LOFTR.TRAINING = False

    # 设置其他配置参数
    config.LOAD_PREDICTIONS_PATH = None
    config.EXP_NAME = args.exp_name
    config.USE_CORRESPONDENCE_TRANSFORMER = False
    config.CORRESPONDENCES_USE_FIT_ONLY = False
    config.EVAL_SPLIT = "test"
    config.PL_VERSION = pl.__version__

    # 加载图像
    image0 = cv2.imread(args.img_path0, cv2.IMREAD_GRAYSCALE)  # 读取灰度图像
    image0 = cv2.resize(image0, (args.w, args.h))  # 调整图像尺寸
    image0 = torch.from_numpy(image0).float()[None].unsqueeze(0).cuda() / 255  # 转换为张量并归一化
    
    image1 = cv2.imread(args.img_path1, cv2.IMREAD_GRAYSCALE)  # 读取灰度图像
    image1 = cv2.resize(image1, (args.w, args.h))  # 调整图像尺寸
    image1 = torch.from_numpy(image1).float()[None].unsqueeze(0).cuda() / 255  # 转换为张量并归一化

    def get_intrinsics(fx, fy, cx, cy):
        """获取相机内参矩阵"""
        K = [[fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]]
        K = np.array(K)
        K = torch.from_numpy(K.astype(np.double)).unsqueeze(0).cuda()
        return K, K
        
    K_0, K_1 = get_intrinsics(float(args.fx), float(args.fy), float(args.cx), float(args.cy))

    # 创建未使用的张量（占位符）
    depth0 = depth1 = torch.tensor([]).unsqueeze(0).cuda()
    T_0to1 = T_1to0 = torch.tensor([]).unsqueeze(0).cuda()
    scene_name = torch.tensor([]).unsqueeze(0).cuda()
    loaded_preds = torch.tensor([]).unsqueeze(0).cuda()
    lightweight_numcorr = torch.tensor([0]).unsqueeze(0).cuda()

    # 构建输入批次
    batch = {
        "image0": image0,   # (1, h, w) - 图像0
        "image1": image1,   # (1, h, w) - 图像1
        "K0": K_0,         # (3, 3) - 相机0内参
        "K1": K_1,         # (3, 3) - 相机1内参
        # 以下字段未使用
        "depth0": depth0,   # (h, w) - 深度图0
        "depth1": depth1,   # (h, w) - 深度图1
        "T_0to1": T_0to1,   # (4, 4) - 真实位姿0到1
        "T_1to0": T_1to0,   # (4, 4) - 真实位姿1到0
        "dataset_name": ["mp3d"],  # 数据集名称
        "scene_id": scene_name,     # 场景ID
        "pair_id": 0,               # 对ID
        "pair_names": (args.img_path0, args.img_path1),  # 图像对名称
        "loaded_predictions": loaded_preds,  # 加载的预测结果
        "lightweight_numcorr": lightweight_numcorr,  # 轻量级对应点数量
    }

    # 初始化模型
    model = PL_LoFTR(config, pretrained_ckpt=args.ckpt_path, split="test").eval().cuda()
    
    # 前向传播
    batch = model.test_step(batch, batch_idx=0, skip_eval=True)

    # 输出结果
    print("预测的位姿是：\n", np.round(batch["loftr_rt"].cpu().numpy(),4))
