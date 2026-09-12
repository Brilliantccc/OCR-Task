# 项目1：简单OCR系统

基于CRNN+CTC的端到端文字识别系统。

## 功能特性

- ✅ CRNN模型（CNN + LSTM）
- ✅ CTC Loss训练
- ✅ 数据增强（旋转、缩放、噪声、模糊）
- ✅ 断点续训 / 微调模式
- ✅ YOLO风格训练进度展示
- ✅ GPU训练支持

## 技术栈

- PyTorch 2.1
- OpenCV
- CTC Loss

## 安装与运行

```bash
# 安装依赖
pip install -r requirements.txt

# 训练
python train.py

# 评估
python evaluate.py --test
```

## 项目结构

```
├── config.py           # 配置文件
├── dataset.py          # 数据集加载
├── model.py            # CRNN模型
├── train.py            # 训练脚本
├── evaluate.py         # 评估脚本
├── utils.py            # 工具函数
└── requirements.txt    # 依赖
```

## 许可证

MIT License
