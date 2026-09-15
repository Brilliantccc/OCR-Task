"""
表格识别系统配置文件
支持：表格检测（TrackA）+ 结构识别（TrackB）
"""
import os
import torch

# ==================== 路径配置 ====================
# AutoDL 用户: 请将数据集放在 /root/autodl-tmp/data/ 或 /root/data/ 下
# 然后修改 DATA_DIR 为实际路径，例如:
#   DATA_DIR = "/root/autodl-tmp/data/ICDAR2019_cTDaR-master"
# 或通过软链接: ln -s /root/autodl-tmp/data/ICDAR2019_cTDaR-master <项目目录>/data/
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "ICDAR2019_cTDaR-master")

# TrackA: 表格检测
# 实际结构: training/TRACKA/ground_truth/ (图片+XML混合)
#           test/TRACKA/ (图片)
#           test_ground_truth/TRACKA/ (XML)
TRACKA_DIR = os.path.join(DATA_DIR, "training", "TRACKA")
TRACKA_TRAIN_IMG = os.path.join(TRACKA_DIR, "ground_truth")
TRACKA_TRAIN_ANN = os.path.join(TRACKA_DIR, "ground_truth")
TRACKA_TEST_IMG = os.path.join(DATA_DIR, "test", "TRACKA")
TRACKA_TEST_ANN = os.path.join(DATA_DIR, "test_ground_truth", "TRACKA")

# TrackB: 结构识别
TRACKB_DIR = os.path.join(DATA_DIR, "training", "TRACKB1")
TRACKB_TRAIN_IMG = os.path.join(TRACKB_DIR, "ground_truth")
TRACKB_TRAIN_ANN = os.path.join(TRACKB_DIR, "ground_truth")
TRACKB_TEST_IMG = os.path.join(DATA_DIR, "test", "TRACKB1")
TRACKB_TEST_ANN = os.path.join(DATA_DIR, "test_ground_truth", "TRACKB1")

# 模型保存路径
# AutoDL 用户: 如系统盘空间不足，可改为 /root/autodl-tmp/runs
MODEL_DIR = os.path.join(BASE_DIR, "runs")


def get_next_version():
    """自动获取下一个版本号"""
    import re
    if not os.path.exists(MODEL_DIR):
        return 1
    versions = []
    for item in os.listdir(MODEL_DIR):
        match = re.match(r'v(\d+)', item)
        if match:
            versions.append(int(match.group(1)))
    return max(versions) + 1 if versions else 1


# 版本管理
VERSION = None
VERSION_DIR = None


def get_next_version_by_task(task):
    """获取指定任务的下一个版本号"""
    import re
    task_dir = os.path.join(MODEL_DIR, task)
    if not os.path.exists(task_dir):
        return 1
    versions = []
    for item in os.listdir(task_dir):
        match = re.match(r'v(\d+)', item)
        if match:
            versions.append(int(match.group(1)))
    return max(versions) + 1 if versions else 1


def init_version(task=None):
    """初始化版本号（在train.py中调用）
    task: 'detection' 或 'structure'
    """
    global VERSION, VERSION_DIR
    if task:
        VERSION = get_next_version_by_task(task)
        VERSION_DIR = os.path.join(MODEL_DIR, task, f"v{VERSION}")
    else:
        VERSION = get_next_version()
        VERSION_DIR = os.path.join(MODEL_DIR, f"v{VERSION}")


# ==================== 类别配置 ====================
# TrackA: 表格检测（2类：背景 + 表格）
DETECTION_CLASSES = {"background": 0, "table": 1}
DETECTION_NUM_CLASSES = len(DETECTION_CLASSES)

# TrackB: 结构识别（3类：实际数据只有 table 和 cell）
STRUCTURE_CLASSES = {
    "background": 0,
    "table": 1,
    "cell": 2,
}
STRUCTURE_NUM_CLASSES = len(STRUCTURE_CLASSES)

# ==================== 训练配置 ====================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 检测模型训练参数（RTX 3090 24GB 优化）
DETECTION_BATCH_SIZE = 16  # 24GB显存可以跑到16
DETECTION_LR = 5e-5  # 降低学习率，更稳定收敛
DETECTION_EPOCHS = 100  # 增加轮数

# 结构识别训练参数（RTX 3090 24GB 优化）
STRUCTURE_BATCH_SIZE = 8  # 小数据集用小 batch
STRUCTURE_LR = 1e-4  # 降低学习率
STRUCTURE_EPOCHS = 100  # 增加轮数

# 通用参数（15 vCPU）
WORKERS = 8
PIN_MEMORY = True
IMAGE_SIZE = 1024  # Faster R-CNN 输入尺寸（正方形）
IMG_CHANNELS = 3

# 学习率调度
LR_WARMUP_EPOCHS = 5
LR_MIN = 1e-7

# 早停（配合100 epochs）
EARLY_STOP_PATIENCE = 20

# 数据划分（如果数据集没有预划分）
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# ==================== 数据增强配置 ====================
# 文档/表格场景: 增强不宜过强，保持文字清晰可读
AUGMENTATION = {
    "horizontal_flip_prob": 0.5,
    "vertical_flip_prob": 0.2,       # 文档倒置概率低，降低
    "brightness_range": (0.8, 1.2),  # 轻微调整，保护文字对比度
    "contrast_range": (0.8, 1.2),    # 同上
    "saturation_range": (0.8, 1.2),  # 同上
    "noise_prob": 0.2,               # 文档图像噪声概率降低
    "blur_prob": 0.1,                # 模糊会破坏文字，大幅降低
}

# ==================== 后处理配置 ====================
NMS_IOU_THRESHOLD = 0.3  # cell 可能相邻，降低 NMS 阈值
DETECTION_SCORE_THRESHOLD = 0.3  # 默认置信度阈值（动态阈值会覆盖）
MAX_DETECTIONS = 2000  # 每张图最多 2224 个 cell

# Faster R-CNN 内部 NMS 阈值（结构识别用，密集 cell 场景需要更低的值）
# 默认0.5 太严格，会抑制相邻 cell；0.3 更宽松，保留更多检测
STRUCTURE_NMS_THRESH = 0.3

# Soft-NMS 配置（保留密集 cell 检测，提升 Recall）
NMS_METHOD = 'soft'       # 'hard' 或 'soft'
SOFT_NMS_SIGMA = 0.5      # 高斯衰减系数（0.3=保守, 0.5=通用, 0.7=宽松）
SOFT_NMS_SCORE_THRESH = 0.001  # Soft-NMS 最低保留分数

# ==================== 日志配置 ====================
LOG_INTERVAL = 10
SAVE_INTERVAL = 5


def init_dirs():
    """创建必要的目录"""
    os.makedirs(MODEL_DIR, exist_ok=True)
    if VERSION_DIR:
        os.makedirs(VERSION_DIR, exist_ok=True)


if __name__ == "__main__":
    print("Configuration Test:")
    print(f"  Base Dir:    {BASE_DIR}")
    print(f"  Data Dir:    {DATA_DIR}")
    print(f"  TrackA Dir:  {TRACKA_DIR}")
    print(f"  TrackB Dir:  {TRACKB_DIR}")
    print(f"  Device:      {DEVICE}")
    print(f"  Detection:   {DETECTION_NUM_CLASSES} classes")
    print(f"  Structure:   {STRUCTURE_NUM_CLASSES} classes")
