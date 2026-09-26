# 项目4：公式识别系统

[![ModelScope](https://img.shields.io/badge/ModelScope-模型权重-blue)](https://www.modelscope.cn/models/Brilliantccc/latex-formula-recognition)
[![GitHub](https://img.shields.io/badge/GitHub-源码-black)](https://github.com/Brilliantccc/latex-formula-recognition)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

手写 / 印刷数学公式图像 → LaTeX 的端到端识别系统。

**预训练权重**：[ModelScope - latex-formula-recognition](https://www.modelscope.cn/models/Brilliantccc/latex-formula-recognition)
（`best.pth` + `vocab.json` 两个文件必须一起下载并放在同一目录，详见下文「词表与权重必须放在一起」）
模型卡见 [MODEL_CARD.md](MODEL_CARD.md)。

## 功能特性

- ✅ 手写公式识别（CROHME + HME100K）
- ✅ 印刷公式识别
- ✅ 输出 LaTeX 格式
- ✅ 公式可视化渲染（预测结果直接渲染成排版后的公式图）
- ✅ 结果导出（JSON / LaTeX / TXT / Markdown）
- ✅ 双架构可切换，便于 A/B 对比

## 技术栈

本项目提供两套架构，通过 `config.py` 的 `MODEL_ARCH` 切换：

| | `vit_transformer`（默认） | `cnn_lstm`（基线） |
|---|---|---|
| 编码器 | ViT-B/16，**自写**，2D sin-cos 位置编码 | ResNet-34 |
| 解码器 | Transformer（d_model 512 / 8 头 / 4 层） | LSTM + Bahdanau Attention |
| 参数量 | 103.2M | 24.5M |
| 训练方式 | 单次并行前向 + 因果掩码 | Teacher Forcing 逐步解码 |
| 输入尺寸 | 高度 **128**，宽度可变（patch 16 → 8 × W/16 网格） | 高度 128，宽度可变 |

> **图像高度用 128 而不是更省算力的 64**：早期用 64 时实测的错误几乎全是
> **单个字形混淆**（5↔3、7↔1、9↔0、e↔R），结构和上下标位置都对，只是认错字符。
> 这是分辨率不足的典型症状——H=64 / patch=16 时只有 4 行 patch，一个手写数字
> 大约只占 1 个 patch。改成 128（8 行）后 CROHME ExpRate 从 0.425 提升到 0.524，
> 且上述那些具体混淆全部消失。详见 [docs/执行计划.md](docs/执行计划.md) 的分辨率小节。

> **为什么自写 ViT 而不用 torchvision 的 `vit_b_16`**：后者在源码里硬断言
> 输入必须是固定方形尺寸（`_process_input` 里的 `torch._assert(h == self.image_size)`），
> 而公式图像的宽高比实测最高达 16:1，用固定方形会被严重压扁。
> 因此自建同构模块、**逐块搬运** torchvision 的预训练权重（14/14 命中），
> 位置编码换成与分辨率无关的 2D sin-cos。
>
> 权重逐块拷贝而不是整体 `load_state_dict`，还有一个好处：checkpoint 里不会
> 嵌入 torchvision 版本相关的子模块命名（如 `encoder_layer_0`），
> 本地 torchvision 0.28 与训练环境 0.16 之间可以互传权重。

## 评估指标

**主指标是 ExpRate（表达式完全匹配率）**，即预测 LaTeX 与真实标注**完全一致**的比例。
这是数学公式识别（MER）领域的标准口径，也是 `best.pth` 的选择依据。

| 指标 | 说明 | 目标 |
|------|------|------|
| **ExpRate（主）** | 完全匹配率 | > 50% |
| BLEU-4 | 辅助参考 | > 0.85 |
| 编辑距离 | 预测与真实的字符级差异 | < 0.15 |

> 早期版本以 BLEU 为主指标，现在改为 ExpRate：BLEU 对公式这类**结构性序列**偏弱
> （一个符号位置错、括号层级错，BLEU 可能仍然很高），而 ExpRate 直接反映"能不能用"。
> BLEU 保留在评估输出里作为参考，并在 ExpRate 同分时用于 `best.pth` 的决胜。

评估结果会按 **HME100K 官方难度划分**（easy / medium / hard）分档报告，
以及按公式长度分档。

## 实测结果

`vit_transformer` + `IMG_HEIGHT=128`，训练 40 轮（早停，约 5.2 小时 / RTX 3090）：

| 指标 | HME100K test (24607) | CROHME eval (986) | 目标 |
|------|---------------------|-------------------|------|
| **ExpRate（主）** | **0.6325** | 0.5243 | > 0.50 |
| BLEU-4 | **0.9261** | 0.7881 | > 0.85 |
| 编辑距离 | **0.0419** | 0.0910 | < 0.15 |

HME100K 按官方难度分档：

| 难度 | 样本数 | ExpRate |
|------|--------|---------|
| easy | 7721 | 0.7586 |
| medium | 10450 | 0.6372 |
| hard | 6436 | 0.4736 |

> ### ⚠️ 关于两个数据集数字的可比性
>
> **报数请以 HME100K test 为准**，理由是它无选择偏差：
> - `best.pth` 是按 **CROHME** 的验证 ExpRate 选出来的
> - 而 **HME100K test 从未参与任何模型选择**，所以 0.6325 是无偏估计
> - **CROHME 的 0.5243 偏高**：它既用于选模型、又用于报数。该集合只有 986 个样本，
>   ExpRate 的标准误约 **1.6 个百分点**，而 0.5243 是 40 轮高度相关评估中的**最大值**，
>   必然向上偏。训练末段 10 轮的典型水平是 0.49~0.50
>
> 要拿到无偏的 CROHME 数字，需要把 CROHME eval 拆成验证/测试两半并重新训练。
>
> 另外，CROHME 只占训练数据的 **10.6%**（8835 / 83337）——这是刻意的设计取舍，
> 但意味着 CROHME 上衡量的是跨分布表现，不代表同分布泛化能力（HME100K 的 0.6325 才是）。

### 剩余错误是什么（想继续优化前请先读）

用 `analyze_errors.py` 对 HME100K test 做过完整归类（详见
[docs/执行计划.md](docs/执行计划.md) §4.8），结论是**没有便宜的修复手段了**：

| 曾经的优化假设 | 实测结果 |
|---|---|
| 大小写 / l↔1 混淆是主要矛盾 | 占错误 **3.8%** |
| 括号方向搞反 | 占错误 **0.9%** |
| 括号配平后处理 | 天花板 **+0.95 个点**（预测括号不配平的仅占 0.95% 样本） |

真实的错误以两类为主，**都不是后处理能解决的**：

- **括号分组错**：81.5% 的错误含括号，但只有 2.6% 括号不配平——
  模型知道该有多少括号，只是分组错了（如 `\frac{a b}{c}` 应为 `\frac{a}{b c}`）。
  输出本身是合法的平衡括号，后处理无从判断该在哪断开
- **数字/字形混淆**：`3↔2`、`3↔5`、`2↔1` 等，需要重新看图，只能靠模型能力

继续提升需要重训实验（更高分辨率 / 更大容量 / 更多数据）。

## 数据集

| 数据集 | 规模 | 用途 |
|--------|------|------|
| CROHME | train 8835 / eval 986 | 训练 + **验证集** |
| HME100K | train 74502 / test 24607 | 训练主力 + 测试 |

> **数据划分是刻意这样设计的**：HME100K 占训练数据主导，CROHME 作为验证和标准评测集。
> 注意 CROHME 只占训练数据的 10.6%，所以验证指标是在这个分布上测的。
>
> 数据集不入 git（见 `.gitignore`），需自行下载后放到 `data/` 下。

### 数据来源与致谢

**HME100K**（Handwritten Mathematical Expression 100K）
- 来源：**好未来（TAL Education）AI 开放平台**
- 官方页面：https://ai.100tal.com/dataset
- 论文：Yuan et al., *Syntax-Aware Network for Handwritten Mathematical Expression Recognition*, arXiv:2203.01601
- 用途：本项目的主力训练集（74,502 张）与主测试集（24,607 张）
- 本仓库**不分发**该数据集，请通过上述官方渠道获取

**CROHME**（Competition on Recognition of Online Handwritten Mathematical Expressions）
- 来源：CROHME 竞赛官方（http://crohme.liris.cnrs.fr/）
- **本项目实际使用的下载地址**：https://aistudio.baidu.com/datasetdetail/174727
  （官方站点经常无法访问，AI Studio 上的这份镜像可用）
- 用途：验证集与评测集
- 原始格式为 INKML 在线笔迹，本项目使用其渲染后的图像版本

感谢以上数据集的发布方，没有它们本项目无法完成。

## 环境

训练在 AutoDL（**PyTorch 2.1.2** + CUDA 12.1 + RTX 3090）上跑，
本地开发机是 Windows + torch 2.13 / torchvision 0.28（也有 CUDA，可跑小规模验证）。

两边 torch 相差一个大版本，实测确认过两件事：

- `torch.amp.GradScaler` 在 2.1.2 上**不存在**，必须回退到 `torch.cuda.amp`
  （`train.py` 里已做防御性导入）
- 权重可跨版本互传：本仓库的 checkpoint 在 torch 2.1.2 上训练、在 torch 2.13 上加载，
  评估指标**逐位复现**（这是自写 ViT 模块、不把 torchvision 内部命名写进 state_dict 的原因）

### 安装依赖

```bash
# AutoDL（镜像已预装 torch，只装额外依赖）
pip install -r requirements-autodl.txt

# 全新环境（含 torch / torchvision）
pip install -r requirements.txt
```

> ### ⚠️ 不要放宽 `numpy` 和 `opencv-python` 的版本上界
>
> `requirements*.txt` 里写的是 `numpy>=1.24,<2` 和 `opencv-python>=4.8,<5`，
> **两个上界都不能去掉**：
>
> - `torch 2.1.x` 需要 `numpy<2`（numpy 2.0 改了 ABI，torch 直到 2.3 才支持）
> - `opencv-python 5.x` 的依赖声明是 `numpy>=2`（Python ≥3.9），装上就会把 numpy 拉过界
>
> 实测确认过：本机装的是 `opencv-python 5.0.0.93`，其 `METADATA` 明确写着
> `numpy>=2; python_version >= "3.9"`。一旦在 AutoDL 上让 pip 自由升级，
> **torch 会在 import 阶段直接崩**。
>
> 分开两个文件也是为此：AutoDL 镜像里的 torch 版本已配好 CUDA，
> 让 pip 去动它很容易把环境搞坏。

## 使用方式

### 训练

```bash
python train.py --arch vit_transformer --epochs 60
```

常用参数：

| 参数 | 说明 |
|------|------|
| `--arch` | `vit_transformer` / `cnn_lstm`，默认取 `config.MODEL_ARCH` |
| `--epochs` / `--batch_size` / `--lr` | 覆盖 config 默认值 |
| `--resume runs/v1/latest.pth` | 断点续训 |
| `--version N` | 指定版本号（默认自动递增） |

实测参考（RTX 3090，`IMG_HEIGHT=128`、batch 32、宽度分桶组批）：

- **约 8 分钟 / epoch**，训练 + 验证合计
- 第 30 轮左右到达平台期，**早停（patience 10）会在第 40 轮前后自动结束**，约 5.2 小时
- 显存峰值约 9.6 GB / 24 GB，batch 有加大余地（但调 batch 需同步调学习率）

> 如果你把 `IMG_HEIGHT` 改回 64：patch 总数减到 1/4、每 epoch 约 4.3 分钟，
> 但 CROHME ExpRate 会从 0.524 掉到 0.425，**不推荐**。

训练产物（**权重与词表放在一起**）：

```
runs/v{N}/
├── vocab.json          # 词表，与权重一一配对，必须一起拷贝
├── best.pth            # ExpRate 最高的权重（只含模型权重，不含优化器状态）
├── latest.pth          # 最后一个 epoch（含优化器状态，供断点续训）
└── train_history.json  # 训练曲线
```

> `best.pth` 刻意**不含**优化器状态：AdamW 在 103M 参数上要多占约 826MB，
> 对推理毫无用处。所以它只有约 394MB，而 `latest.pth` 约 1.2GB。
> **续训请用 `latest.pth`**——用 `best.pth` 续训会用随机初始化的优化器状态。

### 验证（不训练也能跑）

改完模型/训练代码后，**先本地跑这个**：

```bash
python verify_pipeline.py              # 结构检查（快）
python verify_pipeline.py --overfit    # 追加小样本过拟合测试
```

包含四项核心检查：

- **标签泄漏检查**：篡改 `targets[j:]` 后，早期位置的输出必须逐元素不变。
  并行前向 + 因果掩码的张量对齐极易 off-by-one，一旦泄漏，模型会「抄答案」——
  训练 loss 正常下降、推理却完全无效，**不报错、不 NaN，只有专项测试能发现**。
- **2D 位置编码顺序**：行列分量必须与 patch 展平顺序一致（搞反是静默错误）
- **padding mask**：右侧补白的 patch 必须被交叉注意力忽略
- **过拟合测试**：十几个真实样本应能被拟合到 ExpRate≈100%，
  这是验证「tokenizer → dataset → model → loss → decode」整链路无误的最有效手段

### 评估

```bash
python evaluate.py --weights runs/v1/best.pth --split all
python evaluate.py --weights runs/v1/best.pth --split test --beam
```

`--split`：`eval`(CROHME) / `test`(HME100K) / `all`。`--beam` 用 beam search 解码（更准但慢）。

### 推理

```bash
# 单张图片
python inference.py --weights runs/v1/best.pth --input test.jpg --beam

# 整个目录
python inference.py --weights runs/v1/best.pth --input ./images --output_dir ./results --beam
```

| 参数 | 说明 |
|------|------|
| `--beam` | 使用 Beam Search（宽 5），比贪心解码更准，长公式差距明显 |
| `--arch` | 一般不必传，架构会从 checkpoint 自动识别 |
| `--no_render` | 只输出 LaTeX 源码，不渲染公式图 |
| `--no_formula_png` | 不额外保存单张公式渲染图 |
| `--export` | 批量模式导出格式：`json,tex,txt,md,none`（默认 `json,tex`） |

单图推理产生两个文件：

- `xxx_result.jpg` —— 对比图：**输入图像 | 预测公式渲染 | 真实公式渲染**
- `xxx_formula.png` —— 单独的公式渲染图，适合直接嵌入文档

代码里调用：

```python
from inference import FormulaPredictor

predictor = FormulaPredictor('runs/v1/best.pth')

latex = predictor.predict('test.jpg', use_beam=True)          # 只要 LaTeX
pred, result_path, info = predictor.predict_and_visualize('test.jpg')   # 代码 + 渲染图
records = predictor.run_batch('./images', export=('json', 'tex'))       # 批量
```

## ⚠️ 词表与权重必须放在一起

词表 `vocab.json` 由 `train.py` 生成到**权重所在的版本目录**，
推理和评估只会从权重同目录查找，**没有共享的默认路径可以回退**。

这是刻意的设计：词表配错不会报错，只会让输出全错。所以：

- 拷贝权重时**务必连 `vocab.json` 一起拷**
- checkpoint 里记录了 `vocab_size`，加载时会交叉校验；
  词表缺失或大小对不上都会**立即报错**并说明怎么修

词表只从**训练集**标签统计（不包含测试集），避免数据泄漏。

## 公式渲染说明

渲染使用 **matplotlib 自带的 mathtext 引擎**，无需安装 TeX 发行版，离线、跨平台可用。
代价是 mathtext 只支持 LaTeX 数学模式的一个子集：

- 渲染失败时逐级降级：**清洗排版命令 → 摘掉引擎不认识的部分 → 退化为等宽源码**，
  降级情况会标在图上（`[partial]` / `[text]`），不会静默丢内容
- `\begin{array}` / `\cases` 等环境、`\displaystyle`、`\Big` 系列不支持，会被清洗（内容保留、排版简化）
- `\sum` / `\int` 的上下限渲染在右侧，而非真实 LaTeX 显示模式的上方/下方
- 公式含中文时自动切换中文字体，代价是 `\alpha` 等希腊字母会退化成拉丁形近字母，
  此时标记为 `[latex+cjk]`

### 导出的 .tex 怎么编译

```bash
xelatex results.tex
```

导出文档内含中文，**pdflatex 无法处理，必须用 xelatex**。
未安装 `ctex` 宏包时可注释掉该行，代价是中文不显示。

需要 100% 排版保真时，用导出的 `results.tex` 配合 xelatex 渲染。

## 许可证

MIT License
