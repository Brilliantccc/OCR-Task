# OCR-Task

文档智能识别项目集合，包含5个子项目。

## 项目概览

| 项目 | 描述 | 状态 |
|------|------|------|
| [项目1：简单OCR系统](项目1_简单OCR系统/) | CRNN+CTC文字识别 | ✅ 已完成 |
| [项目2：表格识别工具](项目2_表格识别工具/) | YOLO/DETR表格检测 | 🚧 进行中 |
| [项目3：文档版面分析系统](项目3_文档版面分析系统/) | LayoutLM版面分析 | 🚧 进行中 |
| [项目4：公式识别系统](项目4_公式识别系统/) | Encoder-Decoder公式识别 | 🚧 进行中 |
| [项目5：端到端文档解析系统](项目5_端到端文档解析系统/) | VLM完整文档解析 | 🚧 进行中 |

## 项目架构

```
OCR-Task/
├── 项目1_简单OCR系统/          # CRNN + CTC
├── 项目2_表格识别工具/          # YOLO/DETR
├── 项目3_文档版面分析系统/      # LayoutLM
├── 项目4_公式识别系统/          # Encoder-Decoder
└── 项目5_端到端文档解析系统/    # VLM
```

## 技术栈

- **深度学习**: PyTorch
- **计算机视觉**: OpenCV
- **OCR**: CRNN, CTC Loss
- **检测**: YOLO, DETR
- **多模态**: LayoutLM, VLM

## 快速开始

```bash
# 克隆仓库
git clone https://github.com/Brilliantccc/OCR-Task.git
cd OCR-Task

# 进入具体项目
cd 项目1_简单OCR系统

# 安装依赖
pip install -r requirements.txt

# 训练
python train.py
```

## 许可证

MIT License
