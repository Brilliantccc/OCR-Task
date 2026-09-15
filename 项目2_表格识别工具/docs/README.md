# 表格识别工具 — 使用指南

基于 Faster R-CNN 的表格检测与结构识别系统，支持 ICDAR 2019 cTDaR 数据集。

**模型权重**: [ModelScope - table-detection-and-structure](https://www.modelscope.cn/models/Brilliantccc/table-detection-and-structure)
**源码**: [GitHub - table-detection-and-structure](https://github.com/Brilliantccc/table-detection-and-structure)

## 目录

- [项目概述](#项目概述)
- [项目结构](#项目结构)
- [环境配置](#环境配置)
- [数据准备](#数据准备)
- [训练](#训练)
  - [表格检测训练 (TrackA)](#1-表格检测训练-tracka--在文档中找表格)
  - [结构识别训练 (TrackB)](#2-结构识别训练-trackb--在表格内找-cell)
  - [合并版训练脚本](#3-合并版训练脚本兼容旧用法)
- [评估](#评估)
- [推理](#推理)
- [配置参数](#配置参数)
- [Soft-NMS 说明](#soft-nms-说明)
- [推荐训练流程](#推荐训练流程)
- [常见问题](#常见问题)

---

## 项目概述

本系统分为两个阶段：

| 阶段 | 任务 | 模型 | 类别 |
|------|------|------|------|
| TrackA | 表格检测 | Faster R-CNN (ResNet50-FPN) | 背景 + 表格 (2类) |
| TrackB | 结构识别 | Faster R-CNN (ResNet50-FPN) | 背景 + 表格 + cell (3类) |

两个任务共享同一个模型类 `TableDetector`，通过 `num_classes` 参数区分。

---

## 项目结构

```
项目2_表格识别工具/
├── config.py              # 全局配置（路径、超参、NMS参数）
├── model.py               # 模型定义（TableDetector）
├── dataset.py             # 数据集加载与增强
├── utils.py               # 工具函数（NMS、IoU、可视化、checkpoint）
├── train_detection.py     # 训练：表格检测 (TrackA)
├── train_structure.py     # 训练：结构识别 (TrackB)
├── train_combined.py      # 训练：合并版（兼容旧用法）
├── evaluate.py            # 评估：检测 or 结构识别
├── inference.py           # 推理：端到端表格识别
├── docs/
│   └── README.md          # 本文档
└── runs/                  # 模型保存目录
    ├── detection/
    │   ├── v1/
    │   └── v2/
    └── structure/
        ├── v1/
        └── v2/
```

---

## 环境配置

### 推荐环境（AutoDL）

| 项目 | 版本 |
|------|------|
| PyTorch | 2.1.0 |
| Python | 3.10 |
| CUDA | 12.1 |
| GPU | RTX 3090 (24GB) × 1 |
| CPU | 15 vCPU Intel Xeon Platinum 8358P |

### 安装依赖

```bash
# AutoDL 镜像已预装 PyTorch, 只需安装其余依赖
pip install -r requirements.txt

# 如需手动安装 PyTorch (非 AutoDL 环境):
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
```

主要依赖：

| 包 | 用途 |
|---|------|
| torch / torchvision | 模型训练与推理（AutoDL 已预装） |
| opencv-python | 图像处理与数据增强 |
| numpy | 数值计算 |
| tqdm | 进度条 |
| Pillow | 图像读写 |

### AutoDL 数据挂载

AutoDL 的数据盘通常在 `/root/autodl-tmp/`。有两种方式配置数据路径：

**方式一：软链接（推荐）**

```bash
# 假设数据集已上传到 /root/autodl-tmp/data/ICDAR2019_cTDaR-master/
cd 项目2_表格识别工具/
ln -s /root/autodl-tmp/data/ICDAR2019_cTDaR-master ./data/
```

**方式二：修改 config.py**

```python
# 编辑 config.py，将 DATA_DIR 改为实际路径
DATA_DIR = "/root/autodl-tmp/data/ICDAR2019_cTDaR-master"
```

### AutoDL 训练注意事项

1. **显存**：RTX 3090 24GB，当前配置 batch_size=16 (检测) / 8 (结构) 可正常运行
2. **数据盘**：模型 checkpoints 建议保存到数据盘 `/root/autodl-tmp/`，避免系统盘空间不足
3. **后台训练**：使用 `nohup` 或 `tmux` 保持训练不中断：
   ```bash
   tmux new -s train
   python train_detection.py
   # Ctrl+B, D 断开; tmux attach -t train 重新连接
   ```
4. **GPU 监控**：另开终端运行 `watch -n 1 nvidia-smi` 查看显存和利用率

---

## 数据准备

数据集使用 **ICDAR 2019 cTDaR**（Competition on Table Detection and Recognition）。

### 目录结构

```
data/ICDAR2019_cTDaR-master/
├── training/
│   ├── TRACKA/
│   │   └── ground_truth/    # 图片 + XML 标注（表格检测）
│   └── TRACKB1/
│       └── ground_truth/    # 图片 + XML 标注（结构识别）
├── test/
│   ├── TRACKA/
│   └── TRACKB1/
└── test_ground_truth/
    ├── TRACKA/
    └── TRACKB1/
```

### 标注格式

XML 文件使用 VOC 风格，关键标签：

- **TrackA**: `<table>` → `<Coords points="x1,y1 x2,y2 x3,y3 x4,y4"/>`
- **TrackB**: `<cell>` / `<row>` / `<column>` → 同上格式

坐标为四边形顶点，自动转换为 `[x1, y1, x2, y2]` 矩形框。

---

## 训练

### 1. 表格检测训练 (TrackA — 在文档中找表格)

```bash
# 新训练（默认 100 epochs）
python train_detection.py

# 断点续训（默认续训最新版本）
python train_detection.py --resume

# 断点续训指定版本
python train_detection.py --resume --version 1   # 续训 v1
python train_detection.py --resume --version 3   # 续训 v3

# 微调（从已有模型开始，学习率自动 ×0.1）
python train_detection.py --finetune runs/detection/v1/best_model.pth

# 自定义训练轮数
python train_detection.py --epochs 50
```

**模型配置：**
- 架构：`TableDetector(num_classes=2, pretrained=True)`
- 优化器：AdamW (lr=5e-5, weight_decay=1e-4)
- 学习率调度：线性预热 5 epoch + 余弦退火
- 混合精度训练（AMP）
- 梯度裁剪：max_norm=5.0
- 早停：20 epochs 无改善则停止

**保存路径：** `runs/detection/v{N}/best_model.pth`

---

### 2. 结构识别训练 (TrackB — 在表格内找 cell)

```bash
# 新训练
python train_structure.py

# 从检测模型迁移 backbone（推荐，可加速收敛）
python train_structure.py --backbone runs/detection/v1/best_model.pth

# 断点续训（默认续训最新版本）
python train_structure.py --resume

# 断点续训指定版本
python train_structure.py --resume --version 1   # 续训 v1

# 微调
python train_structure.py --finetune runs/structure/v1/best_model.pth

# 自定义训练轮数
python train_structure.py --epochs 80
```

**模型配置：**
- 架构：`TableDetector(num_classes=3, pretrained=True, detections_per_img=2000)`
- 优化器：AdamW (lr=1e-4, weight_decay=1e-4)
- `detections_per_img=2000`：表格内 cell 数量可能很多，提高检测上限
- 采样器：`batch_size_per_image=512`, `positive_fraction=0.5`

**保存路径：** `runs/structure/v{N}/best_model.pth`

---

### 3. 合并版训练脚本（兼容旧用法）

```bash
# 表格检测
python train_combined.py --task detection

# 结构识别
python train_combined.py --task structure

# 结构识别 + backbone 迁移
python train_combined.py --task structure --backbone runs/detection/v1/best_model.pth

# 断点续训
python train_combined.py --task detection --resume

# 断点续训指定版本
python train_combined.py --task structure --resume --version 1
```

> **建议**：优先使用独立的 `train_detection.py` 和 `train_structure.py`，逻辑更清晰。

---

## 评估

```bash
# 评估表格检测（自动查找最新版本）
python evaluate.py --task detection

# 评估结构识别（自动查找最新版本）
python evaluate.py --task structure

# 指定 checkpoint 路径
python evaluate.py --task detection --checkpoint runs/detection/v2/best_model.pth

# 附带可视化结果
python evaluate.py --task structure --visualize --num_vis 20
```

### 输出指标

| 指标 | 说明 |
|------|------|
| AP@0.50 | IoU=0.5 时的 Average Precision |
| AP@0.75 | IoU=0.75 时的 Average Precision |
| AP@0.50:0.95 | IoU 从 0.5 到 0.95 的平均 AP（COCO 标准） |
| Precision | 精确率（IoU=0.5） |
| Recall | 召回率（IoU=0.5） |
| F1-Score | F1 分数 |
| TP / FP / FN | 真阳性 / 假阳性 / 假阴性数量 |

### 评估阈值差异

| 任务 | Score 阈值 | NMS 方法 |
|------|-----------|----------|
| 检测 (TrackA) | 0.5 | 由 config 控制 |
| 结构 (TrackB) | 0.3 | 由 config 控制（默认 Soft-NMS） |

结构识别使用更低的 score 阈值，因为 cell 更小、更密集。

### 可视化输出

使用 `--visualize` 时，结果保存到 `runs/{task}/vis/` 目录：
- 绿色框：Ground Truth
- 红色框：预测结果（附置信度分数）

---

## 推理

端到端表格识别：检测表格 → 裁剪 → 识别结构 → 生成 HTML + JSON。

```bash
# 单张图片
python inference.py \
    --input test.jpg \
    --detector runs/detection/v1/best_model.pth \
    --structure runs/structure/v1/best_model.pth \
    --visualize

# 批量处理整个目录
python inference.py \
    --input ./test_images/ \
    --output ./results/ \
    --detector runs/detection/v1/best_model.pth \
    --structure runs/structure/v1/best_model.pth

# 指定 GPU
python inference.py \
    --input test.jpg \
    --detector model_det.pth \
    --structure model_str.pth \
    --device cuda:0
```

### 输出文件

每张输入图片生成两个文件：

**JSON 结构化数据：**
```json
{
  "tables": [
    {
      "index": 0,
      "bbox": [x1, y1, x2, y2],
      "detection_score": 0.95,
      "structure": {
        "cells": [{"class": "cell", "bbox": [...], "confidence": 0.88}, ...],
        "rows": [...],
        "columns": [...],
        "headers": [...]
      },
      "html": "<table border='1'>...</table>"
    }
  ]
}
```

**HTML 可视化表格：** 可直接在浏览器中打开查看。

---

## 配置参数

所有参数在 [config.py](../config.py) 中配置。

### 训练参数

| 参数 | 检测默认值 | 结构默认值 | 说明 |
|------|-----------|-----------|------|
| `BATCH_SIZE` | 16 | 8 | 批大小 |
| `LR` | 5e-5 | 1e-4 | 学习率 |
| `EPOCHS` | 100 | 100 | 训练轮数 |
| `LR_WARMUP_EPOCHS` | 5 | 5 | 预热轮数 |
| `LR_MIN` | 1e-7 | 1e-7 | 最低学习率 |
| `EARLY_STOP_PATIENCE` | 20 | 20 | 早停轮数 |
| `IMAGE_SIZE` | 1024 | 1024 | 输入图片尺寸 |
| `WORKERS` | 8 | 8 | 数据加载线程数 |

### 后处理参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `NMS_METHOD` | `'soft'` | NMS 方法：`'soft'` 或 `'hard'` |
| `SOFT_NMS_SIGMA` | `0.5` | Soft-NMS 高斯衰减系数 |
| `SOFT_NMS_SCORE_THRESH` | `0.001` | Soft-NMS 最低保留分数 |
| `NMS_IOU_THRESHOLD` | `0.3` | Hard-NMS 的 IoU 阈值 |
| `DETECTION_SCORE_THRESHOLD` | `0.3` | 默认置信度阈值（推理时） |
| `MAX_DETECTIONS` | `2000` | 每张图最大检测数 |

### 动态阈值

训练验证时使用按面积分桶的动态阈值（在 `train_structure.py` 中）：

| Cell 面积 (px²) | Score 阈值 | IoU 阈值 |
|-----------------|-----------|---------|
| < 2,000 | 0.15 | 0.15 |
| < 8,000 | 0.20 | 0.20 |
| < 20,000 | 0.30 | 0.30 |
| < 50,000 | 0.40 | 0.40 |
| ≥ 50,000 | 0.50 | 0.50 |

---

## Soft-NMS 说明

### 为什么用 Soft-NMS？

表格中相邻 cell 的 IoU 容易超过 0.3，标准 Hard-NMS 会直接删除相邻检测框，导致 Recall 降低。

**Hard-NMS**：IoU > 阈值 → 直接删除
**Soft-NMS**：IoU 越大 → 分数衰减越多，但不直接删除

衰减公式：`score_i *= exp(-(iou²) / sigma)`

### 切换方式

在 `config.py` 中修改：

```python
# 使用 Soft-NMS（推荐，适合密集 cell 场景）
NMS_METHOD = 'soft'
SOFT_NMS_SIGMA = 0.5      # 0.3=保守  0.5=通用  0.7=宽松

# 使用 Hard-NMS
NMS_METHOD = 'hard'
NMS_IOU_THRESHOLD = 0.3
```

### Sigma 参数调优

| Sigma | 行为 | 适用场景 |
|-------|------|---------|
| 0.3 | 衰减快，接近 Hard-NMS | 目标间隔较大 |
| 0.5 | 平衡（默认） | 通用场景 |
| 0.7 | 衰减慢，保留更多框 | 极密集 cell |

---

## 推荐训练流程

```bash
# ========== Step 1: 训练表格检测模型 ==========
python train_detection.py --epochs 100
# 产出: runs/detection/v1/best_model.pth

# ========== Step 2: 评估检测模型 ==========
python evaluate.py --task detection
# 查看 mAP、Precision、Recall

# ========== Step 3: 用检测 backbone 迁移学习，训练结构识别 ==========
python train_structure.py --backbone runs/detection/v1/best_model.pth --epochs 100
# 产出: runs/structure/v1/best_model.pth

# ========== Step 4: 评估结构识别模型 ==========
python evaluate.py --task structure --visualize
# 查看 cell 检测效果

# ========== Step 5: 端到端推理 ==========
python inference.py \
    --input test.jpg \
    --detector runs/detection/v1/best_model.pth \
    --structure runs/structure/v1/best_model.pth \
    --visualize
```

---

## 常见问题

### Q: 显存不足 (OOM)

降低 batch size，在 `config.py` 中修改：

```python
DETECTION_BATCH_SIZE = 8    # 原 16
STRUCTURE_BATCH_SIZE = 4    # 原 8
```

### Q: 如何继续之前的训练？

```bash
# 续训最新版本（自动查找）
python train_detection.py --resume
python train_structure.py --resume

# 续训指定版本
python train_detection.py --resume --version 1   # 续训 v1
python train_structure.py --resume --version 2   # 续训 v2
```

自动加载 `runs/{task}/v{N}/best_model.pth`，从上次中断的 epoch 继续。不指定 `--version` 时默认续训最新版本。

### Q: 如何从新版本开始训练？

删除旧版本目录或让脚本自动分配新版本号（默认行为，每次运行自动 +1）。

### Q: 检测模型训练完后，想用新数据微调？

```bash
python train_detection.py --finetune runs/detection/v1/best_model.pth
```

学习率自动降为原来的 0.1 倍。

### Q: 结构识别一定要从检测模型迁移 backbone 吗？

不是必须的，但推荐。迁移 backbone 可以利用检测模型已学到的文档特征，通常能：

- 收敛更快（减少约 30% 训练时间）
- 初始 mAP 更高

```bash
python train_structure.py --backbone runs/detection/v1/best_model.pth
```

### Q: Soft-NMS 和 Hard-NMS 怎么选？

- **密集 cell 检测**（结构识别）→ Soft-NMS（默认已配置）
- **稀疏目标检测**（表格检测）→ Hard-NMS 即可，差异不大

在 `config.py` 中修改 `NMS_METHOD` 即可切换，无需改代码。
