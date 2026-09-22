# mp3d_loftr模块配置文件注释

from yacs.config import CfgNode as CN

_CN = CN()

##############  LoFTR Pipeline  ##############
_CN.LOFTR = CN()
_CN.LOFTR.BACKBONE_TYPE = "ResNetFPN"  # 特征提取网络类型，使用ResNetFPN作为骨干网络
_CN.LOFTR.RESOLUTION = (8, 2)  # 分辨率配置，(coarse_level, fine_level)的分辨率
_CN.LOFTR.FINE_WINDOW_SIZE = 5  # fine匹配窗口大小，必须是奇数
_CN.LOFTR.FINE_CONCAT_COARSE_FEAT = True  # 是否在fine级别concat coarse特征

# 1. LoFTR-backbone (local feature CNN) config
_CN.LOFTR.RESNETFPN = CN()
_CN.LOFTR.RESNETFPN.INITIAL_DIM = 128  # ResNetFPN初始维度
_CN.LOFTR.RESNETFPN.BLOCK_DIMS = [128, 196, 256]  # ResNetFPN各层维度 [s1, s2, s3]

# 2. LoFTR-coarse module config
_CN.LOFTR.COARSE = CN()
_CN.LOFTR.COARSE.D_MODEL = 256  # coarse transformer模型维度
_CN.LOFTR.COARSE.D_FFN = 256  # coarse feed-forward网络维度
_CN.LOFTR.COARSE.NHEAD = 8  # coarse多头注意力头的数量
_CN.LOFTR.COARSE.LAYER_NAMES = ["self", "cross"] * 4  # coarse transformer层结构
_CN.LOFTR.COARSE.ATTENTION = "linear"  # 注意力类型：linear或full
_CN.LOFTR.COARSE.TEMP_BUG_FIX = True  # 临时bug修复标志

# 3. Coarse-Matching config
_CN.LOFTR.MATCH_COARSE = CN()
_CN.LOFTR.MATCH_COARSE.THR = 0.2  # 匹配阈值
_CN.LOFTR.MATCH_COARSE.BORDER_RM = 2  # 边缘移除像素数
_CN.LOFTR.MATCH_COARSE.MATCH_TYPE = "dual_softmax"  # 匹配类型：dual_softmax或sinkhorn
_CN.LOFTR.MATCH_COARSE.DSMAX_TEMPERATURE = 0.1  # dual softmax温度参数
_CN.LOFTR.MATCH_COARSE.SKH_ITERS = 3  # sinkhorn迭代次数
_CN.LOFTR.MATCH_COARSE.SKH_INIT_BIN_SCORE = 1.0  # sinkhorn初始化二进制分数
_CN.LOFTR.MATCH_COARSE.SKH_PREFILTER = False  # sinkhorn预过滤
_CN.LOFTR.MATCH_COARSE.TRAIN_COARSE_PERCENT = 0.2  # 训练时使用的coarse匹配百分比
_CN.LOFTR.MATCH_COARSE.TRAIN_PAD_NUM_GT_MIN = 200  # 训练时最小GT填充数量
_CN.LOFTR.MATCH_COARSE.SPARSE_SPVS = True  # 是否使用稀疏SPVS

# 4. LoFTR-fine module config
_CN.LOFTR.FINE = CN()
_CN.LOFTR.FINE.D_MODEL = 128  # fine transformer模型维度
_CN.LOFTR.FINE.D_FFN = 128  # fine feed-forward网络维度
_CN.LOFTR.FINE.NHEAD = 8  # fine多头注意力头的数量
_CN.LOFTR.FINE.LAYER_NAMES = ["self", "cross"] * 1  # fine transformer层结构
_CN.LOFTR.FINE.ATTENTION = "linear"  # 注意力类型

# 5. LoFTR-regress module config
_CN.LOFTR.REGRESS = CN()
_CN.LOFTR.REGRESS.D_MODEL = 256  # regress transformer模型维度
_CN.LOFTR.REGRESS.D_FFN = 256  # regress feed-forward网络维度
_CN.LOFTR.REGRESS.NHEAD = 8  # regress多头注意力头的数量
_CN.LOFTR.REGRESS.LAYER_NAMES = ["self", "cross"]  # regress transformer层结构
_CN.LOFTR.REGRESS.ATTENTION = "linear"  # 注意力类型
_CN.LOFTR.REGRESS.TEMP_BUG_FIX = False  # 临时bug修复标志
_CN.LOFTR.REGRESS.USE_POS_EMBEDDING = False  # 是否使用位置编码

# 6. LoFTR Losses
_CN.LOFTR.LOSS = CN()
# coarse-level损失
_CN.LOFTR.LOSS.COARSE_TYPE = "focal"  # coarse损失类型：focal或cross_entropy
_CN.LOFTR.LOSS.COARSE_WEIGHT = 1.0  # coarse损失权重
# focal loss参数
_CN.LOFTR.LOSS.FOCAL_ALPHA = 0.25  # focal loss的alpha参数
_CN.LOFTR.LOSS.FOCAL_GAMMA = 2.0  # focal loss的gamma参数
_CN.LOFTR.LOSS.POS_WEIGHT = 1.0  # 正样本权重
_CN.LOFTR.LOSS.NEG_WEIGHT = 1.0  # 负样本权重

# fine-level损失
_CN.LOFTR.LOSS.FINE_TYPE = "l2_with_std"  # fine损失类型
_CN.LOFTR.LOSS.FINE_WEIGHT = 1.0  # fine损失权重
_CN.LOFTR.LOSS.FINE_CORRECT_THR = 1.0  # fine正确阈值

# RT (Rotation and Translation)损失
_CN.LOFTR.LOSS.RT_WEIGHT_ROT = 0.0  # 旋转损失权重
_CN.LOFTR.LOSS.RT_WEIGHT_TR = 0.0  # 平移损失权重

# Scale损失
_CN.LOFTR.LOSS.SCALE_WEIGHT = 0.0  # 尺度损失权重

##############  Dataset  ##############
_CN.DATASET = CN()
# 1. data config
# training and validating
_CN.DATASET.TRAINVAL_DATA_SOURCE = None  # 训练验证数据源
_CN.DATASET.TRAIN_DATA_ROOT = None  # 训练数据根目录
_CN.DATASET.TRAIN_POSE_ROOT = None  # 训练位姿数据目录
_CN.DATASET.TRAIN_NPZ_ROOT = None  # 训练NPZ文件目录
_CN.DATASET.TRAIN_LIST_PATH = None  # 训练数据列表路径
_CN.DATASET.TRAIN_INTRINSIC_PATH = None  # 训练相机内参路径

# testing
_CN.DATASET.TEST_DATA_SOURCE = None  # 测试数据源
_CN.DATASET.TEST_DATA_ROOT = None  # 测试数据根目录
_CN.DATASET.TEST_POSE_ROOT = None  # 测试位姿数据目录

# 2. dataset config
_CN.DATASET.AUGMENTATION_TYPE = None  # 数据增强类型

# MegaDepth options
_CN.DATASET.MGDPT_IMG_RESIZE = 640  # 图像调整大小
_CN.DATASET.MGDPT_IMG_PAD = True  # 图像填充为正方形
_CN.DATASET.MGDPT_DEPTH_PAD = True  # 深度图填充
_CN.DATASET.MGDPT_DF = 8  # 深度因子

##############  Trainer  ##############
_CN.TRAINER = CN()
_CN.TRAINER.WORLD_SIZE = 1  # 世界大小（GPU数量）
_CN.TRAINER.CANONICAL_BS = 64  # 标准批大小
_CN.TRAINER.CANONICAL_LR = 6e-3  # 标准学习率
_CN.TRAINER.SCALING = None  # 自动计算缩放
_CN.TRAINER.FIND_LR = False  # 是否使用学习率查找器

# optimizer
_CN.TRAINER.OPTIMIZER = "adamw"  # 优化器类型
_CN.TRAINER.TRUE_LR = None  # 实际学习率（自动计算）
_CN.TRAINER.ADAM_DECAY = 0.  # Adam优化器衰减
_CN.TRAINER.ADAMW_DECAY = 0.1  # AdamW优化器衰减

# learning rate scheduler
_CN.TRAINER.SCHEDULER = "MultiStepLR"  # 学习率调度器类型
_CN.TRAINER.SCHEDULER_INTERVAL = "epoch"  # 调度器间隔
_CN.TRAINER.MSLR_MILESTONES = [3, 6, 9, 12]  # MultiStepLR里程碑
_CN.TRAINER.MSLR_GAMMA = 0.5  # MultiStepLR衰减因子

# plotting related
_CN.TRAINER.ENABLE_PLOTTING = True  # 是否启用绘图
_CN.TRAINER.N_VAL_PAIRS_TO_PLOT = 32  # 验证对数量
_CN.TRAINER.PLOT_MODE = "evaluation"  # 绘图模式

# geometric metrics and pose solver
_CN.TRAINER.EPI_ERR_THR = 5e-4  # 极线误差阈值
_CN.TRAINER.POSE_GEO_MODEL = "E"  # 几何模型：E（Essential Matrix）、F（Fundamental Matrix）、H（Homography）
_CN.TRAINER.POSE_ESTIMATION_METHOD = "RANSAC"  # 位姿估计方法
_CN.TRAINER.RANSAC_PIXEL_THR = 0.5  # RANSAC像素阈值
_CN.TRAINER.RANSAC_CONF = 0.99999  # RANSAC置信度
_CN.TRAINER.RANSAC_MAX_ITERS = 10000  # RANSAC最大迭代次数

# data sampler
_CN.TRAINER.DATA_SAMPLER = "scene_balance"  # 数据采样器类型

# gradient clipping
_CN.TRAINER.GRADIENT_CLIPPING = 0.5  # 梯度裁剪值

# reproducibility
_CN.TRAINER.SEED = 66  # 随机种子

# Matterport3D specific settings
_CN.DATASET.TRAINVAL_DATA_SOURCE = "mp3d"
_CN.DATASET.TEST_DATA_SOURCE = "mp3d"
_CN.DATASET.TRAIN_DATA_JSON = "mp3d_planercnn_json/cached_set_train.json"
_CN.DATASET.VAL_DATA_JSON = "mp3d_planercnn_json/cached_set_val.json"
_CN.DATASET.TEST_DATA_JSON = "mp3d_planercnn_json/cached_set_test.json"
_CN.DEPTH_DIR = "observations"
_CN.DATA_DIR = "mp3d_rpnet_v4_sep20"

def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _CN.clone()
