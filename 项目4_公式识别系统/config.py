"""
公式识别系统配置文件
数据集: CROHME (手写公式) + HME100K (10万手写公式)

支持两套架构（MODEL_ARCH 切换）:
  - vit_transformer : ViT 编码器 + Transformer 解码器（默认，对齐执行计划备选 A/C）
  - cnn_lstm        : ResNet-34 + Bahdanau Attention + LSTM（基线，用于 A/B 对比）
"""
import os
import torch

# ==================== 路径配置 ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# CROHME
CROHME_DIR = os.path.join(DATA_DIR, "CROHME")
CROHME_TRAIN_IMG = os.path.join(CROHME_DIR, "training", "images")
CROHME_TRAIN_LABEL = os.path.join(CROHME_DIR, "training", "labels.txt")
CROHME_EVAL_IMG = os.path.join(CROHME_DIR, "evaluation", "images")
CROHME_EVAL_LABEL = os.path.join(CROHME_DIR, "evaluation", "labels.txt")
CROHME_VOCAB = os.path.join(CROHME_DIR, "words_dict.txt")

# HME100K
HME100K_DIR = os.path.join(DATA_DIR, "HME100K")
HME100K_TRAIN_IMG = os.path.join(HME100K_DIR, "train_images")
HME100K_TRAIN_LABEL = os.path.join(HME100K_DIR, "train_labels.txt")
HME100K_TEST_IMG = os.path.join(HME100K_DIR, "test_images")
HME100K_TEST_LABEL = os.path.join(HME100K_DIR, "test_labels.txt")
# 官方难度划分（仅 test 集，easy+medium+hard = test 全集）
HME100K_SUBSET_DIR = os.path.join(HME100K_DIR, "subset")

# 模型保存路径
MODEL_DIR = os.path.join(BASE_DIR, "runs")

# ==================== 架构选择 ====================
# 'vit_transformer' | 'cnn_lstm'
MODEL_ARCH = 'vit_transformer'

# ==================== 特殊 Token ====================
PAD_TOKEN = 0
SOS_TOKEN = 1
EOS_TOKEN = 2
UNK_TOKEN = 3
SPECIAL_TOKENS = ["<PAD>", "<SOS>", "<EOS>", "<UNK>"]

# ==================== 图像配置 ====================
# 固定高度（公式图像通常宽扁）。
#
# 从 64 提到 128 的原因：H=64 / patch=16 时只有 4 行 patch，而实测的错误几乎
# 全是**单个字形混淆**（5↔3、7↔1、9↔0、e↔R、n↔1）——结构、上下标位置都对，
# 只是认错字符，说明图像分辨率不足。参考实现 pix2tex 用的也是 128 高度。
#
# ⚠️ 代价比"高度翻倍"要大：图像等比缩放，宽度也翻倍，所以 patch 总数是 **4 倍**
#    （行 4→8、列数也翻倍），每步耗时约为原来的 3.3 倍。
IMG_HEIGHT = 128
# 宽度上限：等比缩放到 H=128 后，覆盖 99% 样本需要约 2048，但那会让 ViT 的
# O(patches²) 注意力把显存顶爆（batch 32 实测外推约 24GB，正好卡在 3090 上限）。
# 1536 对应宽高比 12，只压扁约 1% 的样本，batch 32 峰值约 18.5GB，留有余量。
# 实测依据见 docs/执行计划.md 的分辨率小节。
IMG_MAX_WIDTH = 1536
IMG_CHANNELS = 1          # 灰度输入（CNN 基线用）
VIT_IN_CHANNELS = 3       # ViT 复制成 3 通道以复用 ImageNet 预训练权重

# ==================== 序列配置 ====================
MAX_SEQ_LEN = 200         # LaTeX 序列最大长度（实测 P99=57, max=186）
MIN_FREQ = 2              # 词表最低出现频次

# ==================== 训练配置（RTX 3090 24GB） ====================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
BASE_LR = 3e-4            # peak learning rate
ENCODER_LR = 1e-4         # Encoder 使用更低学习率（预训练 backbone）
WARMUP_EPOCHS = 5
EPOCHS = 80
WEIGHT_DECAY = 0.01
DROPOUT = 0.3             # CNN+LSTM 基线的 dropout
GRAD_CLIP_MAX_NORM = 1.0

# Transformer 子层 dropout（执行计划备选 A 用 0.1，比 LSTM 基线低）
TRANSFORMER_DROPOUT = 0.1

# Teacher Forcing 比例调度 —— 仅 cnn_lstm 基线使用。
# Transformer 解码器训练是单次并行前向 + 因果掩码，不存在逐步喂入，
# 这三个配置对 vit_transformer 完全无效。
TF_RATIO_START = 1.0
TF_RATIO_END = 0.3
TF_RATIO_EPOCHS = 40

# 早停（监控 ExpRate）
EARLY_STOP_PATIENCE = 10

# ==================== 模型配置（共用） ====================
LABEL_SMOOTHING = 0.1

# ---- cnn_lstm 基线 ----
EMBED_DIM = 256           # Decoder embedding 维度
DECODER_DIM = 512         # LSTM hidden 维度
ATTN_DIM = 256            # Attention 中间层维度
ENCODER_DIM = 512         # ResNet-34 layer4 输出维度
NUM_DECODER_LAYERS = 1    # LSTM 层数（此前是死配置，从未被 model.py 读取）

# ---- vit_transformer ----
VIT_PATCH_SIZE = 16       # patch 边长
VIT_HIDDEN = 768          # ViT-B/16 hidden 维度
VIT_LAYERS = 12           # ViT-B/16 层数
VIT_HEADS = 12            # ViT-B/16 头数
D_MODEL = 512             # Transformer 解码器维度（执行计划备选 A）
NUM_HEADS = 8
FFN_DIM = 2048
TRANSFORMER_LAYERS = 4

# ==================== 推理配置 ====================
BEAM_WIDTH = 5            # 此前是死配置，beam_search 硬编码 5 从不读它
MAX_DECODE_LEN = 200
# beam search 长度惩罚（GNMT 风格 ((5+len)/6)^alpha），0 表示关闭
BEAM_LEN_PENALTY = 0.6
# 验证时是否用 beam search 解码。
# 默认 False：贪心快，但**选出的 best.pth 是贪心口径下最优**，
# 与推理时开 --beam 的口径不一致。要严格一致就设为 True（慢很多）。
EVAL_USE_BEAM = False

# ==================== 数据加载 ====================
NUM_WORKERS = 8
PERSISTENT_WORKERS = True
PREFETCH_FACTOR = 2

# 按图像宽度分桶组批。
# collate 会把 batch 内所有图补齐到批内最大宽度，而公式宽度差异极大
# （H=64 时中位 244px、最长 1200px）。实测随机组批让 ViT 多算 2.50 倍的
# 空白 patch，按宽度分桶后可降到 1.03 倍 —— 编码器理论提速约 144%。
# 关闭它可用于对比验证（结果不应有实质差异，只是慢）。
BUCKET_BY_WIDTH = True
# 分桶粒度：每个"超级桶"含多少个 batch。桶内打乱后切批，
# 既保证批内宽度相近，又让每轮的组合有变化。
BUCKET_BATCHES_PER_BUCKET = 10
# 扫描图像尺寸的线程数（只读文件头，8 线程实测 74502 张约 8 秒）
WIDTH_SCAN_THREADS = 16


def get_teacher_forcing_ratio(epoch):
    """线性衰减 Teacher Forcing 比例（仅 cnn_lstm 基线使用）"""
    if epoch >= TF_RATIO_EPOCHS:
        return TF_RATIO_END
    ratio = TF_RATIO_START - (TF_RATIO_START - TF_RATIO_END) * epoch / TF_RATIO_EPOCHS
    return max(ratio, TF_RATIO_END)


# ==================== 架构相关派生配置 ====================
def arch_config(arch=None):
    """
    返回指定架构的派生配置（patch 网格尺寸等），供 model / dataset 共用

    Returns:
        dict: {'arch', 'img_height', 'patch_size', 'patch_rows', 'max_patches'}
    """
    arch = arch or MODEL_ARCH
    patch = VIT_PATCH_SIZE if arch == 'vit_transformer' else None
    rows = IMG_HEIGHT // patch if patch else None
    return {
        'arch': arch,
        'img_height': IMG_HEIGHT,
        'patch_size': patch,
        'patch_rows': rows,
        'max_patches': rows * (IMG_MAX_WIDTH // patch) if patch else None,
    }


if __name__ == "__main__":
    print("=" * 62)
    print("公式识别系统 — 配置摘要")
    print("=" * 62)
    print(f"  架构: {MODEL_ARCH}")
    for k, v in {
        'DEVICE': DEVICE, 'BATCH_SIZE': BATCH_SIZE, 'BASE_LR': BASE_LR,
        'ENCODER_LR': ENCODER_LR, 'EPOCHS': EPOCHS, 'WARMUP_EPOCHS': WARMUP_EPOCHS,
        'IMG_HEIGHT': IMG_HEIGHT, 'IMG_MAX_WIDTH': IMG_MAX_WIDTH,
        'MAX_SEQ_LEN': MAX_SEQ_LEN, 'LABEL_SMOOTHING': LABEL_SMOOTHING,
        'EARLY_STOP_PATIENCE': EARLY_STOP_PATIENCE, 'BEAM_WIDTH': BEAM_WIDTH,
        'NUM_WORKERS': NUM_WORKERS,
    }.items():
        print(f"  {k:22s} = {v}")

    print("-" * 62)
    print("  cnn_lstm 基线:")
    for k, v in {'EMBED_DIM': EMBED_DIM, 'DECODER_DIM': DECODER_DIM,
                 'ATTN_DIM': ATTN_DIM, 'ENCODER_DIM': ENCODER_DIM,
                 'NUM_DECODER_LAYERS': NUM_DECODER_LAYERS,
                 'DROPOUT': DROPOUT}.items():
        print(f"    {k:20s} = {v}")

    print("  vit_transformer:")
    for k, v in {'VIT_PATCH_SIZE': VIT_PATCH_SIZE, 'VIT_HIDDEN': VIT_HIDDEN,
                 'VIT_LAYERS': VIT_LAYERS, 'VIT_HEADS': VIT_HEADS,
                 'D_MODEL': D_MODEL, 'NUM_HEADS': NUM_HEADS,
                 'FFN_DIM': FFN_DIM, 'TRANSFORMER_LAYERS': TRANSFORMER_LAYERS,
                 'TRANSFORMER_DROPOUT': TRANSFORMER_DROPOUT}.items():
        print(f"    {k:20s} = {v}")

    ac = arch_config()
    print(f"  patch 网格: {ac['patch_rows']} 行 × 最多 "
          f"{IMG_MAX_WIDTH // VIT_PATCH_SIZE} 列 = {ac['max_patches']} patches")
    print("=" * 62)
    print(f"TF ratio 调度（仅 cnn_lstm）: {TF_RATIO_START} → {TF_RATIO_END} "
          f"(epoch 0 → {TF_RATIO_EPOCHS})")
