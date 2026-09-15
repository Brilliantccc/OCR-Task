# 表格识别工具 (Table Recognition Tool)

[![ModelScope](https://img.shields.io/badge/ModelScope-模型权重-blue)](https://www.modelscope.cn/models/Brilliantccc/table-detection-and-structure)
[![GitHub](https://img.shields.io/badge/GitHub-源码-black)](https://github.com/Brilliantccc/table-detection-and-structure)

基于 Faster R-CNN 的表格检测与结构识别系统，支持 ICDAR 2019 cTDaR 数据集。

## 功能

- **表格检测 (TrackA)** — 在文档页面中定位表格区域
- **结构识别 (TrackB)** — 在表格内检测 cell / row / column
- **端到端推理** — 检测 → 裁剪 → 结构识别 → 输出 HTML + JSON
- **Soft-NMS** — 高斯衰减 NMS，保留密集 cell 检测
- **动态阈值** — 按目标面积分桶自适应 score / IoU 阈值

## 模型架构

| 任务 | 模型 | Backbone | 类别 |
|------|------|----------|------|
| 表格检测 | Faster R-CNN | ResNet50-FPN | 背景 + 表格 (2类) |
| 结构识别 | Faster R-CNN | ResNet50-FPN | 背景 + 表格 + cell (3类) |

两个任务共享 `TableDetector` 类，通过 `num_classes` 参数区分。

## 项目结构

```
├── config.py              # 全局配置（路径、超参、NMS参数）
├── model.py               # 模型定义（TableDetector）
├── dataset.py             # 数据集加载与增强
├── utils.py               # 工具函数（NMS、IoU、可视化）
├── train_detection.py     # 训练：表格检测 (TrackA)
├── train_structure.py     # 训练：结构识别 (TrackB)
├── train_combined.py      # 训练：合并版（兼容旧用法）
├── evaluate.py            # 评估：检测 or 结构
├── inference.py           # 推理：端到端表格识别
├── docs/README.md         # 详细使用指南
├── requirements.txt       # 依赖
└── data/                  # 数据集目录（需自行下载）
```

## 快速开始

### 环境要求

| 项目 | 版本 |
|------|------|
| Python | 3.10+ |
| PyTorch | 2.1.0+ |
| CUDA | 12.1+ |
| GPU | RTX 3090 (24GB) 或同等 |

### 安装

```bash
pip install -r requirements.txt
```

### 数据准备

下载 ICDAR 2019 cTDaR 数据集，放到 `data/ICDAR2019_cTDaR-master/` 目录下：

```bash
mkdir -p data
cd data
git clone --depth 1 https://github.com/cndplab-founder/ICDAR2019_cTDaR.git
```

### 训练

```bash
# Step 1: 训练表格检测模型
python train_detection.py

# Step 2: 用检测模型的 backbone 初始化，训练结构识别
python train_structure.py --backbone runs/detection/v1/best_model.pth

# 断点续训（自动找最新版本）
python train_structure.py --resume

# 断点续训指定版本
python train_structure.py --resume --version 1
```

### 评估

```bash
python evaluate.py --task detection
python evaluate.py --task structure --visualize
```

### 推理

```bash
python inference.py \
    --input test.jpg \
    --detector runs/detection/v1/best_model.pth \
    --structure runs/structure/v1/best_model.pth \
    --visualize
```

## 训练效果

| 模型 | 任务 | mAP@0.50 |
|------|------|:--------:|
| 检测模型 | 在文档中找表格 | **0.9078** |
| 结构识别模型 | 在表格内找 cell | **0.7511** |

结构识别模型详细指标（ICDAR 2019 cTDaR TrackB 测试集）：

| 指标 | 值 |
|------|-----|
| AP@0.50 | 0.7511 |
| AP@0.75 | 0.6342 |
| AP@0.50:0.95 | 0.5643 |
| Precision | 0.9191 |
| Recall | 0.6574 |
| F1-Score | 0.7665 |

## 模型下载

预训练模型权重：[ModelScope - table-detection-and-structure](https://www.modelscope.cn/models/Brilliantccc/table-detection-and-structure)

## 详细文档

使用指南、配置参数、AutoDL 部署说明等请参考 [docs/README.md](docs/README.md)。

## 许可证

MIT License
