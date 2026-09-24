# OCR-Task

文档智能识别项目集合，包含 5 个子项目。

## ⚠ 克隆方式

项目1/2/3 是**独立仓库**，通过 git submodule 引用。直接 `git clone` 会得到空的子目录，
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
| [项目2：表格识别工具](项目2_表格识别工具/) | 表格检测 + 结构识别 | 🚧 进行中 |
| [项目3：文档版面分析系统](项目3_文档版面分析系统/) | RT-DETR 版面元素检测（11 类） | ✅ 已完成 |
| [项目4：公式识别系统](项目4_公式识别系统/) | Encoder-Decoder 公式识别 | 🚧 进行中 |
| [项目5：端到端文档解析系统](项目5_端到端文档解析系统/) | VLM 完整文档解析 | 🚧 进行中 |

## 项目架构

```
OCR-Task/                        （本仓库）
├── 项目1_简单OCR系统/            # submodule → Simple-OCR-System
├── 项目2_表格识别工具/           # submodule → table-detection-and-structure
├── 项目3_文档版面分析系统/       # submodule → document-layout-analysis
├── 项目4_公式识别系统/           # 本仓库内
└── 项目5_端到端文档解析系统/     # 本仓库内
```

前三个拆成独立仓库是因为它们各自有完整的数据、训练脚本和发布权重，独立仓库的
issue / release / star 都更清晰。

## 相关仓库与模型

| 项目 | GitHub | ModelScope |
|------|--------|-----------|
| 项目1 | [Simple-OCR-System](https://github.com/Brilliantccc/Simple-OCR-System) | [OCR-CRNN-v2](https://www.modelscope.cn/models/Brilliantccc/OCR-CRNN-v2) |
| 项目2 | [table-detection-and-structure](https://github.com/Brilliantccc/table-detection-and-structure) | [table-detection-and-structure](https://www.modelscope.cn/models/Brilliantccc/table-detection-and-structure) |
| 项目3 | [document-layout-analysis](https://github.com/Brilliantccc/document-layout-analysis) | [rtdetr-r50vd-doclaynet-layout](https://www.modelscope.cn/models/Brilliantccc/rtdetr-r50vd-doclaynet-layout) · [faster-rcnn-r50fpn-doclaynet-layout](https://www.modelscope.cn/models/Brilliantccc/faster-rcnn-r50fpn-doclaynet-layout) |

## 技术栈

- **深度学习**: PyTorch
- **计算机视觉**: OpenCV
- **文字识别**: CRNN, CTC Loss
- **文档版面分析**: RT-DETR, Faster R-CNN
- **表格**: 检测 + 结构识别两阶段
- **公式识别**: Encoder-Decoder (Transformer)
- **端到端解析**: VLM（视觉语言模型）

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

MIT License —— 见 [LICENSE](LICENSE)。各子项目使用的数据集与预训练权重有各自的许可证
（如项目3 的 DocLayNet、项目4 的 HME100K），详见对应子项目 README。
