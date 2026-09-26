# OCR-Task

文档智能识别项目集合，包含 4 个子项目。

## ⚠ 克隆方式

项目1/2/3/4 是**独立仓库**，通过 git submodule 引用。直接 `git clone` 会得到空的子目录，
必须带上 `--recursive`：

```bash
git clone --recursive https://github.com/Brilliantccc/OCR-Task.git
cd OCR-Task
```

已经克隆过但子目录为空的话：

```bash
git submodule update --init --recursive
```

## 项目概览

| 项目 | 描述 | 状态 |
|------|------|------|
| [项目1：简单OCR系统](项目1_简单OCR系统/) | CRNN+CTC 文字识别 | ✅ 已完成 |
| [项目2：表格识别工具](项目2_表格识别工具/) | Faster R-CNN 表格检测 + cell 结构识别 | ✅ 已完成 |
| [项目3：文档版面分析系统](项目3_文档版面分析系统/) | RT-DETR 版面元素检测（11 类） | ✅ 已完成 |
| [项目4：公式识别系统](项目4_公式识别系统/) | ViT + Transformer 公式识别（HME100K ExpRate 0.6325） | ✅ 已完成 |

## 项目架构

```
OCR-Task/                        （本仓库）
├── 项目1_简单OCR系统/            # submodule → Simple-OCR-System
├── 项目2_表格识别工具/           # submodule → table-detection-and-structure
├── 项目3_文档版面分析系统/       # submodule → document-layout-analysis
└── 项目4_公式识别系统/           # submodule → latex-formula-recognition
```

四个项目各自有完整的数据、训练脚本和发布权重，拆成独立仓库后
issue / release / star 都更清晰。本仓库作为入口，统一收拢四个 submodule 的指针。

## 相关仓库与模型

| 项目 | GitHub | ModelScope |
|------|--------|-----------|
| 项目1 | [Simple-OCR-System](https://github.com/Brilliantccc/Simple-OCR-System) | [OCR-CRNN-v2](https://www.modelscope.cn/models/Brilliantccc/OCR-CRNN-v2) |
| 项目2 | [table-detection-and-structure](https://github.com/Brilliantccc/table-detection-and-structure) | [table-detection-and-structure](https://www.modelscope.cn/models/Brilliantccc/table-detection-and-structure) |
| 项目3 | [document-layout-analysis](https://github.com/Brilliantccc/document-layout-analysis) | [rtdetr-r50vd-doclaynet-layout](https://www.modelscope.cn/models/Brilliantccc/rtdetr-r50vd-doclaynet-layout) · [faster-rcnn-r50fpn-doclaynet-layout](https://www.modelscope.cn/models/Brilliantccc/faster-rcnn-r50fpn-doclaynet-layout) |
| 项目4 | [latex-formula-recognition](https://github.com/Brilliantccc/latex-formula-recognition) | [latex-formula-recognition](https://www.modelscope.cn/models/Brilliantccc/latex-formula-recognition) |

## 技术栈

- **深度学习**: PyTorch
- **计算机视觉**: OpenCV
- **文字识别**: CRNN, CTC Loss
- **文档版面分析**: RT-DETR, Faster R-CNN
- **表格**: 检测 + 结构识别两阶段
- **公式识别**: ViT + Transformer Encoder-Decoder

## 快速开始

克隆（注意 `--recursive`，见上），然后进入具体项目：

```bash
cd 项目3_文档版面分析系统

# 安装依赖
pip install -r requirements.txt

# 训练 / 评估 / 推理
python train.py --detector rtdetr --epochs 60
python evaluate.py --weights runs/rtdetr/v1/best.pth --split test
python inference.py --weights runs/rtdetr/v1/best.pth --input page.png
```

各项目的详细用法、环境要求和已训练权重见各自 README。

## 许可证

MIT License —— 见 [LICENSE](LICENSE)。各子项目使用的数据集与预训练权重有各自的许可证，
详见对应子项目 README：

- 项目1：ICDAR-2019-SROIE（ICDAR RRC，学术用途）
- 项目2：ICDAR-2019-cTDaR（ICDAR 竞赛，学术用途）
- 项目3：[DocLayNet](https://github.com/DS4SD/DocLayNet)（IBM Research，CDLA-Permissive-1.0）
  + PekingU/rtdetr_r50vd_coco_o365（Apache-2.0）
- 项目4：**HME100K 来源于好未来（TAL Education）AI 开放平台**；CROHME 为 CROHME 竞赛数据
