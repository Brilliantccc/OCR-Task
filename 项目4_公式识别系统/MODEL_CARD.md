# latex-formula-recognition

手写 / 印刷数学公式图像 → LaTeX 的端到端识别模型。

> 本文件是 **ModelScope 模型页**用的模型卡。
> 上传权重时请连同本文件的内容填到模型页的 README 中（或直接作为 README.md 上传）。

## 模型描述

输入一张公式图像，输出对应的 LaTeX 字符串。基于 ViT-B/16 编码器 + Transformer 解码器，
共 **103.2M** 参数。

| | |
|---|---|
| 任务 | 数学公式识别（MER, Mathematical Expression Recognition） |
| 输入 | 灰度公式图像，高度归一化到 **128**，宽度可变 |
| 输出 | LaTeX 字符串（含空格分隔的 token） |
| 参数量 | 103.2M |
| 词表 | 199 个 token |
| 架构 | 自写 ViT-B/16（2D sin-cos 位置编码）+ 4 层 Transformer 解码器 |

## 权重文件

```
best.pth      394 MB    模型权重（不含优化器状态）
vocab.json    3.3 KB    词表
```

> ⚠️ **两个文件必须一起下载并放在同一目录**。词表与权重一一配对，
> 代码会从权重所在目录查找 `vocab.json`，并用 checkpoint 里记录的 `vocab_size`
> 做交叉校验——缺失或大小不符都会**立即报错**（而不是静默输出错误结果）。

## 实测指标

在 **HME100K test**（24,607 张，未参与任何模型选择，指标无偏）：

| 指标 | 数值 |
|------|------|
| **ExpRate（表达式完全匹配率）** | **0.6325** |
| BLEU-4 | **0.9261** |
| 编辑距离 | 0.0419 |

按 HME100K 官方难度划分：

| 难度 | 样本数 | ExpRate |
|------|--------|---------|
| easy | 7,721 | 0.7586 |
| medium | 10,450 | 0.6372 |
| hard | 6,436 | 0.4736 |

在 CROHME eval（986 张）上 ExpRate 为 0.5243——
该数字**偏高**，因为它同时用于选择 checkpoint 和报告结果，
且 CROHME 只占训练数据的 10.6%。

环境：PyTorch 2.1.2 + CUDA 12.1 + RTX 3090，训练 40 轮（早停），约 5.2 小时。

## 使用方法

```bash
git clone https://github.com/Brilliantccc/latex-formula-recognition
cd latex-formula-recognition
pip install -r requirements-autodl.txt

# 把 best.pth 和 vocab.json 放在同一目录下（例如 runs/v2/）
python inference.py --weights runs/v2/best.pth --input test.jpg
```

代码中调用：

```python
from inference import FormulaPredictor

predictor = FormulaPredictor('runs/v2/best.pth')
latex = predictor.predict('test.jpg')          # 只要 LaTeX 字符串
pred, path, info = predictor.predict_and_visualize('test.jpg')   # 附渲染对比图
```

推理默认使用**贪心解码**。实测 beam search 仅提升 0.6 个百分点（BLEU 反而略降），
而耗时是其数倍，因此不推荐。

> **用自己的代码加载本权重时，建议加 `weights_only=True`**
>
> ```python
> ckpt = torch.load('best.pth', map_location='cpu', weights_only=True)
> ```
>
> `best.pth` 里只有张量、字符串、数字和列表，全在 PyTorch 的安全白名单内，
> 实测可用安全模式加载。`.pth` 默认走 pickle 反序列化，**加载时会执行文件里的任意代码**——
> 虽然本权重是自行训练的、可放心，但养成用安全模式的习惯能避免将来踩坑。
>
> 仓库代码中 `inference.py` / `evaluate.py` / `analyze_errors.py` 均已使用 `weights_only=True`；
> 只有断点续训路径（`utils.load_checkpoint`）例外，因为 `latest.pth` 含优化器状态。

## 训练数据

| 数据集 | 规模 | 用途 |
|--------|------|------|
| HME100K | 74,502 训练 / 24,607 测试 | 训练主力 + 主测试集 |
| CROHME | 8,835 训练 / 986 评估 | 训练 + 验证集 |

**本模型不分发数据集**，请通过各数据集官方渠道获取。

### 数据来源与致谢

**HME100K**（Handwritten Mathematical Expression 100K）
- 来源：**好未来（TAL Education）AI 开放平台**
- 官方页面：https://ai.100tal.com/dataset
- 论文：Yuan et al., *Syntax-Aware Network for Handwritten Mathematical Expression Recognition*, arXiv:2203.01601

**CROHME**（Competition on Recognition of Online Handwritten Mathematical Expressions）
- 来源：CROHME 竞赛官方（http://crohme.liris.cnrs.fr/）
- 实际下载地址（官方站点常不可访问）：https://aistudio.baidu.com/datasetdetail/174727

感谢以上数据集的发布方，没有它们本模型无法完成。

## 局限性

- **手写而非印刷**：训练数据以手写公式为主，印刷体（尤其排版复杂的学术公式）表现未经验证
- **长公式较弱**：HME100K 上超过 40 个 token 的公式 ExpRate 仅 0.2783
- **括号分组错误**：81.5% 的错误含括号，其中多数是「括号数量对、位置错」
  （如 `\frac{a b}{c}` 应为 `\frac{a}{b c}`），属于结构建模问题
- **易混字形**：`3↔2`、`3↔5`、`2↔1` 等数字混淆仍有残留
- **输出为 token 流**：结果是空格分隔的 LaTeX（如 `\frac { a } { b }`），
  语义正确但排版冗余，可直接编译
- 领域较窄：仅训练于 CROHME + HME100K，对其他来源的公式泛化能力未评估

## 许可证

MIT License

## 引用

如果本模型对你有帮助，请引用其使用的数据集：

```bibtex
@article{yuan2022syntax,
  title={Syntax-Aware Network for Handwritten Mathematical Expression Recognition},
  author={Yuan, Ye and Liu, Xiao and Dikubab, Wondimu and Liu, Hui and Ji, Zhilong and Wu, Zhongqin and Bai, Xiang},
  journal={arXiv preprint arXiv:2203.01601},
  year={2022}
}
```
