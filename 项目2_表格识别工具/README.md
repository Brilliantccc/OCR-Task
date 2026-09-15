# 项目2：表格识别工具

基于 Faster R-CNN + ResNet18 的表格检测和结构识别系统。

## 功能特性

- ✅ 表格检测（Faster R-CNN + ResNet50-FPN）
- ✅ 表格结构识别（ResNet18 Backbone）
- ✅ 输出 HTML / JSON 格式
- ✅ 端到端推理流水线

## 技术栈

| 组件 | 模型 | 说明 |
|------|------|------|
| 表格检测 | Faster R-CNN | ResNet50-FPN backbone，预训练 COCO |
| 结构识别 | ResNet18 | 自定义检测头，识别 cell/row/column/header |

## 数据集

- ICDAR2019_cTDaR（TrackA: 检测 + TrackB: 结构识别）

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 下载数据集

```bash
cd data
git clone --depth 1 https://github.com/cndplab-founder/ICDAR2019_cTDaR.git
```

### 训练

```bash
# 训练表格检测模型
python train.py --task detection

# 训练结构识别模型
python train.py --task structure

# 断点续训
python train.py --task detection --resume
```

### 评估

```bash
# 评估检测模型
python evaluate.py --task detection --visualize

# 评估结构识别模型
python evaluate.py --task structure
```

### 推理

```bash
# 单张图片
python inference.py --input path/to/image.jpg --output output/

# 批量处理
python inference.py --input path/to/images/ --output output/ --visualize
```

## 文件结构

```
项目2_表格识别工具/
├── config.py          # 配置文件
├── dataset.py         # 数据集加载（VOC XML解析）
├── model.py           # 模型定义（Faster R-CNN + ResNet18）
├── train.py           # 训练脚本
├── evaluate.py        # 评估脚本（mAP计算）
├── inference.py       # 推理脚本（端到端表格识别）
├── utils.py           # 工具函数
├── requirements.txt   # 依赖
└── data/              # 数据集目录
```

## 评估指标

| 指标 | 目标值 | 说明 |
|------|--------|------|
| mAP | > 0.85 | 平均精度均值 (IoU=0.5) |
| Precision | - | 精确率 |
| Recall | - | 召回率 |
| F1-Score | - | F1分数 |

## 训练历史

模型保存在 `runs/{task}/v{N}/` 目录下，按任务分开保存：

```
runs/
├── detection/
│   └── v1/
│       ├── best_model.pth    ← 最佳检测模型
│       ├── last.pth
│       ├── history.json
│       └── history.csv
└── structure/
    └── v1/
        ├── best_model.pth    ← 最佳结构识别模型
        ├── last.pth
        ├── history.json
        └── history.csv
```

每次训练自动递增版本号，不会覆盖之前的模型。

## 许可证

MIT License
