"""
OCR系统配置文件
"""
import os

# ==================== 路径配置 ====================
# 使用相对路径避免Windows中文路径问题
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "ICDAR-2019-SROIE")
IMG_DIR = os.path.join(DATA_DIR, "img_fixed")  # 使用修复后的图像
BOX_DIR = os.path.join(DATA_DIR, "box")
KEY_DIR = os.path.join(DATA_DIR, "key")

# 模型保存路径
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


def init_version():
    """初始化版本号（在train.py中调用）"""
    global VERSION, VERSION_DIR
    VERSION = get_next_version()
    VERSION_DIR = os.path.join(MODEL_DIR, f"v{VERSION}")

# ==================== 数据配置 ====================
# 字符集（收据中常见的字符）
CHARS = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,-/()&$:;#@!?'\"+=%*[]{}\\|<>~`^ "
CHAR_TO_IDX = {ch: i + 1 for i, ch in enumerate(CHARS)}  # 0 用于 CTC blank
IDX_TO_CHAR = {i + 1: ch for i, ch in enumerate(CHARS)}
NUM_CLASSES = len(CHARS) + 1  # +1 for CTC blank

# 图像配置
IMG_HEIGHT = 32
IMG_WIDTH = 640  # 增大宽度以支持长文本
IMG_CHANNELS = 1  # 灰度图

# ==================== 训练配置 ====================
# 设备配置（优先使用GPU）
DEVICE = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") or __import__("torch").cuda.is_available() else "cpu"

# 训练参数（RTX 3090 24GB优化）
BATCH_SIZE = 128
LEARNING_RATE = 0.001
NUM_EPOCHS = 100
WORKERS = 8
PIN_MEMORY = True

# 宽度分桶配置
WIDTH_BUCKETS = [256, 512, 1024, 2048]  # 分桶边界

# 数据划分
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# 学习率调度（余弦退火 + 预热）
LR_WARMUP_EPOCHS = 5  # 预热 epochs
LR_MIN = 1e-6  # 最小学习率

# 早停配置
EARLY_STOP_PATIENCE = 15

# ==================== 数据增强配置 ====================
AUGMENTATION = {
    "rotation_range": 5,
    "scale_range": (0.9, 1.1),
    "noise_prob": 0.2,
    "blur_prob": 0.1,
    "brightness_range": (0.8, 1.2),
}

# ==================== 日志配置 ====================
LOG_INTERVAL = 10  # 每10个batch打印一次
SAVE_INTERVAL = 5  # 每5个epoch保存一次

def init_dirs():
    """创建必要的目录"""
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(VERSION_DIR, exist_ok=True)
