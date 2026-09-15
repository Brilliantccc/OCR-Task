# Simple OCR System

> 基于 CRNN + CTC 的端到端文字识别系统，专为收据/发票场景文本识别设计。

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch" alt="PyTorch">
  <img src="https://img.shields.io/badge/Python-3.8+-3776AB?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License">
</p>

---

## 项目简介

本项目实现了从零搭建的 CRNN（Convolutional Recurrent Neural Network）文字识别模型，配合 CTC（Connectionist Temporal Classification）损失函数，实现无需字符级标注的端到端 OCR 训练。

**核心能力：**
- 在 ICDAR-2019-SROIE 收据数据集上训练
- 字符准确率达到 **95.53%**，词准确率达到 **92.74%**
- 支持断点续训、微调、自动版本管理

## 系统架构

```
┌─────────────────────────────────────────────────────────┐
│                    输入：收据图像                         │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│              文本行裁剪（基于 Bounding Box）              │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│         预处理：灰度化 → 动态宽度 → 归一化                │
│         输出：(batch, 1, 32, width)                      │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                  CRNN 模型                               │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │ CNN 特征提取  │→│ BiLSTM 序列建模│→│ FC + Softmax  │  │
│  │ (7层卷积+BN)  │  │ (2层双向LSTM) │  │ (字符分类)     │  │
│  └──────────────┘  └──────────────┘  └───────────────┘  │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│            CTC 解码 → 贪婪解码输出文本                    │
└─────────────────────────────────────────────────────────┘
```

## 功能特性

### 模型与训练
- **CRNN 架构**：7 层 CNN 特征提取 + 2 层双向 LSTM 序列建模
- **CTC Loss**：端到端训练，无需字符级对齐标注
- **余弦退火 + 预热**：5 epoch 预热 + 余弦退火学习率调度
- **梯度裁剪**：max_norm=5.0，防止梯度爆炸
- **早停机制**：patience=15，防止过拟合

### 数据处理
- **动态宽度**：保持文本行原始宽高比，避免字符变形
- **宽度分桶采样**：按宽度范围分组（256/512/1024/2048），减少 padding 浪费
- **数据增强**：旋转（±5°）、缩放、高斯噪声、模糊、亮度调整
- **JPEG 修复**：自动修复损坏的 JPEG 图像

### 工程特性
- **断点续训**：`--resume` 从上次停止的地方继续
- **微调模式**：`--finetune` 基于预训练模型微调（默认 0.1x 学习率）
- **自动版本管理**：每次训练自动递增版本号（v1, v2, ...）
- **训练历史**：自动保存 JSON + CSV 格式的训练曲线数据
- **tqdm 进度条**：实时展示 loss、准确率等指标

## 环境要求

- Python 3.8+
- CUDA 11.x+（GPU 训练，也支持 CPU）
- RTX 3090 24GB 或同等显存的 GPU（推荐）

## 安装

```bash
# 克隆仓库
git clone https://github.com/Brilliantccc/Simple-OCR-System.git
cd Simple-OCR-System

# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate   # Linux/Mac
# venv\Scripts\activate    # Windows

# 安装依赖
pip install -r requirements.txt
```

## 数据准备

本项目使用 **ICDAR-2019-SROIE** 收据数据集。

### 下载数据集

1. 访问 [ICDAR 2019 SROIE](https://rrc.cvc.uab.es/) 注册账号
2. 下载 `Task1 & Task3 - Text Detection and Recognition` 数据
3. 解压至 `data/ICDAR-2019-SROIE/` 目录

### 数据目录结构

```
data/
└── ICDAR-2019-SROIE/
    ├── img_fixed/    # 收据图像（.jpg）
    ├── box/          # 文本框标注（.csv，格式：x1,y1,...,x4,y4,text）
    └── key/          # 关键信息标注
```

> ⚠️ 模型训练只需 `img_fixed/` 和 `box/` 目录。如果图像有损坏，请先运行 `python fix_jpeg.py` 修复。

## 使用方法

### 训练模型

```bash
# 标准训练（自动创建新版本目录）
python train.py

# 指定训练轮数
python train.py --epochs 50

# 断点续训（从上次停止的地方继续）
python train.py --resume

# 微调模式（基于已有模型）
python train.py --finetune runs/v1/best_model.pth
python train.py --finetune runs/v1/best_model.pth --finetune_lr 0.00001
```

训练产物保存在 `runs/vN/` 目录下：
- `best_model.pth` — 验证集最优模型
- `last.pth` — 最后一个 epoch 的模型
- `history.json` / `history.csv` — 训练历史

### 评估模型

```bash
# 在测试集上评估（默认使用最新版本的最佳模型）
python evaluate.py --test

# 指定模型路径评估
python evaluate.py --model runs/v1/best_model.pth --test

# 可视化预测结果（保存到 prediction_results.png）
python evaluate.py --model runs/v1/best_model.pth --visualize
```

### 单张图像推理

```bash
python evaluate.py --model runs/v1/best_model.pth --image path/to/receipt_line.jpg
```

### Python API 调用

```python
import torch
from model import CRNN
from utils import load_checkpoint, decode_output

# 加载模型
model = CRNN()
load_checkpoint(model, None, "runs/v1/best_model.pth")
model.eval()

# 准备输入（灰度图像，高度 32，动态宽度）
# images: torch.Tensor, shape (batch, 1, 32, width)

# 推理
with torch.no_grad():
    outputs = model(images)
    texts = decode_output(outputs)

print(texts)
```

## 配置参数

主要参数在 [config.py](config.py) 中配置：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `IMG_HEIGHT` | 32 | 输入图像高度 |
| `IMG_CHANNELS` | 1 | 灰度图 |
| `BATCH_SIZE` | 128 | 批大小（RTX 3090 优化） |
| `LEARNING_RATE` | 0.001 | 初始学习率 |
| `NUM_EPOCHS` | 100 | 最大训练轮数 |
| `WIDTH_BUCKETS` | [256, 512, 1024, 2048] | 宽度分桶边界 |
| `LR_WARMUP_EPOCHS` | 5 | 学习率预热轮数 |
| `EARLY_STOP_PATIENCE` | 15 | 早停耐心值 |
| `TRAIN_RATIO` | 0.8 | 训练集比例 |
| `VAL_RATIO` | 0.1 | 验证集比例 |
| `TEST_RATIO` | 0.1 | 测试集比例 |

## 项目结构

```
.
├── README.md               # 本文件
├── requirements.txt        # Python 依赖
├── config.py               # 配置文件（超参数、路径、字符集）
├── model.py                # CRNN 模型定义（CNN + BiLSTM + FC）
├── dataset.py              # 数据集加载（动态宽度、分桶采样、数据增强）
├── train.py                # 训练脚本（支持断点续训/微调）
├── evaluate.py             # 评估与推理脚本
├── utils.py                # 工具函数（CTC解码、准确率计算、checkpoint）
├── fix_jpeg.py             # JPEG 图像修复工具
├── data/                   # 数据集目录
│   └── ICDAR-2019-SROIE/
│       ├── img_fixed/      # 修复后的收据图像
│       ├── box/            # 文本框标注（CSV）
│       └── key/            # 关键信息标注
└── runs/                   # 训练产物
    ├── v1/
    │   ├── best_model.pth  # 最佳模型权重
    │   ├── last.pth        # 最后 epoch 模型
    │   ├── history.json    # 训练历史
    │   └── history.csv
    └── v2/
        └── ...
```

## 训练结果

在 ICDAR-2019-SROIE 测试集上的评估结果：

| 指标 | 值 |
|------|-----|
| **字符准确率 (Char Acc)** | **95.53%** |
| **词准确率 (Word Acc)** | **92.74%** |
| **编辑距离 (Edit Dist)** | **0.11** |

### 样本预测

```
✓ 预测: UNIHAKKA INTERNATIONAL SDN BHD         | 真实: UNIHAKKA INTERNATIONAL SDN BHD
✓ 预测: INPUT TAX CLAIMS, ON THE BASIC OF THE  | 真实: INPUT TAX CLAIMS, ON THE BASIC OF THE
✓ 预测: 27/02/18 21:22                          | 真实: 27/02/18 21:22
✓ 预测: RM 0.30                                 | 真实: RM 0.30
✓ 预测: 8.80                                    | 真实: 8.80
```

## 预训练模型

训练好的模型权重已发布到 ModelScope：

| 模型 | 平台 | 链接 |
|------|------|------|
| CRNN-v2 | ModelScope | [Brilliantccc/OCR-CRNN-v2](https://www.modelscope.cn/models/Brilliantccc/OCR-CRNN-v2) |

### 快速使用

```python
from modelscope.pipelines import pipeline

ocr = pipeline('ocr', model='Brilliantccc/OCR-CRNN-v2')
result = ocr('path/to/receipt.jpg')
print(result)
```

或手动加载模型权重：

```python
import torch
from model import CRNN
from utils import load_checkpoint, decode_output

# 从 ModelScope 下载模型后
model = CRNN()
load_checkpoint(model, None, "best_model.pth")
model.eval()

with torch.no_grad():
    outputs = model(images)
    texts = decode_output(outputs)
```

## 参考资料

- [CRNN 论文](https://arxiv.org/abs/1507.05717) — *An End-to-End Trainable Neural Network for Image-based Sequence Recognition*
- [CRNN 参考实现](https://github.com/bgshih/crnn)
- [ICDAR 2019 SROIE](https://rrc.cvc.uab.es/) — 收据 OCR 数据集
- [CTC Loss](https://distill.pub/2017/ctc/) — Connectionist Temporal Classification 详解

## 许可证

[MIT License](LICENSE)
