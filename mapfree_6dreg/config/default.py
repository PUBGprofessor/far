# mapfree_6dreg模块配置文件注释

from yacs.config import CfgNode as CN

_CN = CN()

##############  Model  ##############
_CN.MODEL = None  # 模型类型：Regression（回归）或FeatureMatching（特征匹配）
_CN.DEBUG = False  # 调试模式

# Regression model options
_CN.ENCODER = CN()
_CN.ENCODER.TYPE = None   # 编码器类型：ResNet或ResUNet
_CN.ENCODER.NUM_BLOCKS = None  # 每层块数量，用-分隔，如3-3-3
_CN.ENCODER.BLOCK_TYPE = None  # 块类型：0:PreactBlock, 1:PreactBlockBottleneck
_CN.ENCODER.NOT_CONCAT = None  # ResUNet选项
_CN.ENCODER.NUM_OUT_LAYERS = None  # ResUNet选项

_CN.AGGREGATOR = CN()
_CN.AGGREGATOR.TYPE = None  # 聚合器类型：CorrelationVolumeWarping或CorrelationVolumeWarpingQKV
_CN.AGGREGATOR.POSITION_ENCODER = None   # 是否添加位置编码
_CN.AGGREGATOR.POSITION_ENCODER_IM1 = None   # 是否为图像1添加均匀位置编码
_CN.AGGREGATOR.MAX_SCORE_CHANNEL = None  # 是否添加最大分数通道
_CN.AGGREGATOR.NORMALISE_DOT = False     # 是否在点积前归一化特征
_CN.AGGREGATOR.RESIDUAL_ATT = False      # 是否使用残差注意力
_CN.AGGREGATOR.CV_OUTLAYERS = 0          # 相关体积压缩层数
_CN.AGGREGATOR.CV_HALF_CHANNELS = False  # 是否使用半通道计算相关体积
_CN.AGGREGATOR.UPSAMPLE_POS_ENC = 0      # 位置编码上采样通道数
_CN.AGGREGATOR.DUSTBIN = False           # 是否创建dustbin处理未匹配特征

_CN.HEAD = CN()
_CN.HEAD.TYPE = None     # 头部类型：ProcrustesResBlockMLP或DirectResBlockMLP
_CN.BACKPROJECT_ANCHORS = None    # 是否将锚点反投影到3D
_CN.HEAD.ADD_BASIS = False        # 是否为MLP锚点添加正交基
_CN.HEAD.NUM_PTS = 6              # 要估计的点数：3、6或更多
_CN.HEAD.AVG_POOL = False         # 是否使用全局平均池化
_CN.HEAD.BATCH_NORM = True        # 是否使用批归一化
_CN.HEAD.SEPARATE_SCALE = True    # 是否分别回归尺度

# Feature Matching Options
_CN.FEATURE_MATCHING = None  # 特征匹配类型：SIFT或Precomputed
_CN.POSE_SOLVER = None  # 位姿求解器：EssentialMatrix、EssentialMatrixMetric、Procrustes、PNP

# SIFT options
_CN.SIFT = CN()
_CN.SIFT.NUM_FEATURES = None  # SIFT特征数量
_CN.SIFT.RATIO_THRESHOLD = None  # 比例阈值

# Pre-computed feature matching options
_CN.MATCHES_FILE_PATH = None    # 预计算对应点文件路径

# EMAT RANSAC options
_CN.EMAT_RANSAC = CN()
_CN.EMAT_RANSAC.PIX_THRESHOLD = None  # 像素阈值
_CN.EMAT_RANSAC.SCALE_THRESHOLD = None  # 尺度阈值
_CN.EMAT_RANSAC.CONFIDENCE = None  # 置信度

# Procrustes RANSAC options
_CN.PROCRUSTES = CN()
_CN.PROCRUSTES.MAX_CORR_DIST = None  # 最大对应点距离
_CN.PROCRUSTES.REFINE = False      # 是否用ICP优化位姿

# PNP RANSAC options
_CN.PNP = CN()
_CN.PNP.RANSAC_ITER = None  # RANSAC迭代次数
_CN.PNP.REPROJECTION_INLIER_THRESHOLD = None  # 重投影内点阈值（像素）
_CN.PNP.CONFIDENCE = None  # 置信度

##############  Dataset  ##############
_CN.DATASET = CN()
# 1. data config
_CN.DATASET.DATA_SOURCE = None # 数据源：ScanNet、7Scenes、MapFree
_CN.DATASET.SCENES = None      # 要使用的场景列表
_CN.DATASET.DATA_ROOT = None   # 数据集根目录
_CN.DATASET.NPZ_ROOT = None    # NPZ文件目录
_CN.DATASET.MIN_OVERLAP_SCORE = None  # 最小重叠分数
_CN.DATASET.MAX_OVERLAP_SCORE = None  # 最大重叠分数
_CN.DATASET.AUGMENTATION_TYPE = None  # 数据增强类型
_CN.DATASET.BLACK_WHITE = False       # 是否转换为黑白图像

# 2. dataset specific settings
_CN.DATASET.PAIRS_TXT = CN()          # 训练/验证/测试对文件路径
_CN.DATASET.PAIRS_TXT.TRAIN = None
_CN.DATASET.PAIRS_TXT.VAL = None
_CN.DATASET.PAIRS_TXT.TEST = None
_CN.DATASET.PAIRS_TXT.ONE_NN = False  # 是否只保留最高相似度的参考图像

_CN.DATASET.HEIGHT = None  # 图像高度
_CN.DATASET.WIDTH = None   # 图像宽度

# depth options
_CN.DATASET.ESTIMATED_DEPTH = None  # 估计深度图路径

############# TRAINING #############
_CN.TRAINING = CN()
# Data Loader settings
_CN.TRAINING.BATCH_SIZE = None  # 批大小
_CN.TRAINING.NUM_WORKERS = None  # 工作进程数
_CN.TRAINING.SAMPLER = None  # 采样器类型：random或scene_balance
_CN.TRAINING.N_SAMPLES_SCENE = None  # 每个场景样本数
_CN.TRAINING.SAMPLE_WITH_REPLACEMENT = None  # 是否有放回采样

# Training settings
_CN.TRAINING.LR = None  # 学习率
_CN.TRAINING.LR_STEP_INTERVAL = None  # 学习率调度间隔
_CN.TRAINING.LR_STEP_GAMMA = None  # 学习率衰减因子
_CN.TRAINING.VAL_INTERVAL = None  # 验证间隔
_CN.TRAINING.VAL_BATCHES = None  # 验证批数
_CN.TRAINING.LOG_INTERVAL = None  # 日志记录间隔
_CN.TRAINING.EPOCHS = None  # 训练轮数
_CN.TRAINING.GRAD_CLIP = 0.   # 梯度裁剪值

# Loss settings
_CN.TRAINING.ROT_LOSS = "rot_frobenius_loss"  # 旋转损失类型
_CN.TRAINING.TRANS_LOSS = "trans_l2_loss"     # 平移损失类型
_CN.TRAINING.LAMBDA = 1.0  # 平移损失缩放因子

cfg = _CN
