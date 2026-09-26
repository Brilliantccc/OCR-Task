"""
工具函数
BLEU 计算、编辑距离、精确匹配率、Beam Search、checkpoint、可视化、LaTeX 渲染
"""
import os
import re
import json
import numpy as np
import torch

from config import PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, DEVICE


# ==================== 依赖检查 ====================
# 下面这些包是在函数内部惰性 import 的（nltk 在 compute_bleu、tqdm 在批量推理、
# matplotlib/PIL 在渲染时）。惰性导入本身没问题，但**失败时机很糟**：
# nltk 缺失会等到第一个 epoch 训练完、开始验证时才炸，整轮算力白烧。
# 所以在训练/评估启动时先统一探一遍。
_LAZY_DEPS = {
    'nltk': 'BLEU 计算',
    'tqdm': '批量推理进度条',
    'matplotlib': '公式渲染',
    'PIL': '公式图保存',
}


def check_dependencies(deps=None, strict=True):
    """
    在启动时探测惰性导入的依赖，避免跑到一半才失败

    Args:
        deps: 要检查的包名列表，默认全部
        strict: True 时缺包直接抛错；False 时只打印警告
    Returns:
        list[str]: 缺失的包名
    """
    import importlib

    names = list(deps) if deps else list(_LAZY_DEPS)
    missing = []
    for name in names:
        try:
            importlib.import_module(name)
        except ImportError:
            missing.append(name)

    if missing:
        detail = '\n'.join(f'  - {n:12s} 用于 {_LAZY_DEPS.get(n, "?")}' for n in missing)
        msg = (f"缺少依赖包:\n{detail}\n"
               f"  安装: pip install {' '.join(missing)}\n"
               f"  或直接: pip install -r requirements.txt")
        if strict:
            raise ImportError(msg)
        print(f'[WARN] {msg}')
    return missing


# ==================== 评估指标 ====================

def tokenize_for_bleu(latex_str):
    """将 LaTeX 字符串分词为 token 列表（用于 BLEU 计算）"""
    from tokenizer import tokenize_latex
    return tokenize_latex(latex_str)


def compute_bleu(predictions, references):
    """
    计算 corpus-level BLEU-4
    Args:
        predictions: list of str, 预测 LaTeX
        references: list of str, 真实 LaTeX
    Returns:
        float, BLEU-4 score
    """
    import nltk
    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

    pred_tokens = [tokenize_for_bleu(p) for p in predictions]
    ref_tokens = [[tokenize_for_bleu(r)] for r in references]

    smoother = SmoothingFunction().method1
    bleu = corpus_bleu(ref_tokens, pred_tokens,
                       weights=(0.25, 0.25, 0.25, 0.25),
                       smoothing_function=smoother)
    return bleu


def _levenshtein_distance(s1, s2):
    """内置 Levenshtein 编辑距离（无需额外依赖）"""
    m, n = len(s1), len(s2)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if s1[i - 1] == s2[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


def compute_edit_distance(predictions, references):
    """
    计算平均归一化编辑距离 (0=完全相同, 1=完全不同)
    """
    if not predictions:
        return 0.0
    total = 0.0
    for pred, ref in zip(predictions, references):
        if len(pred) == 0 and len(ref) == 0:
            total += 0.0
        else:
            dist = _levenshtein_distance(pred, ref)
            total += dist / max(len(pred), len(ref))
    return total / len(predictions)


def compute_exprate(predictions, references):
    """
    计算表达式精确匹配率
    """
    if not predictions:
        return 0.0
    correct = sum(1 for p, r in zip(predictions, references) if p.strip() == r.strip())
    return correct / len(predictions)


# ==================== AverageMeter ====================

class AverageMeter:
    """跟踪统计量的均值和计数"""

    def __init__(self, name=''):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# ==================== Checkpoint ====================

def save_checkpoint(path, epoch, model, optimizer, val_bleu, val_loss=0.0,
                    train_history=None, scheduler=None, extra=None,
                    val_exprate=None, with_optimizer=True, with_history=False):
    """
    保存 checkpoint

    会写入 model.arch（架构标识）—— 两套架构的 state_dict 键名完全不同，
    没有标识的话加载错权重只会报一堆 cryptic 的 key 不匹配。

    Args:
        with_optimizer: 是否存优化器状态。AdamW 在 103M 参数上要多占约 826MB
            （exp_avg + exp_avg_sq 各 413MB），是模型权重本身的两倍。
            best.pth 是给推理/部署用的，不需要优化器状态，传 False 可让文件
            从 ~1.24GB 降到 ~413MB（下载量少 3 倍）。
            断点续训请从 latest.pth 恢复，那个仍然带完整优化器状态。
        with_history: 是否附带训练历史。同样只给 latest.pth 用。
    """
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'arch': getattr(model, 'arch', None),
        # 记录词表大小：加载时用它校验 vocab.json 有没有配错
        'vocab_size': getattr(model, 'vocab_size', None),
        'val_bleu': val_bleu,
        'val_loss': val_loss,
    }
    if with_optimizer and optimizer is not None:
        state['optimizer_state_dict'] = optimizer.state_dict()
    if with_optimizer and scheduler is not None:
        state['scheduler_state_dict'] = scheduler.state_dict()
    if val_exprate is not None:
        state['val_exprate'] = val_exprate
    if with_history and train_history is not None:
        state['train_history'] = train_history
    if extra is not None:
        state.update(extra)
    torch.save(state, path)


def load_checkpoint(path, model, optimizer=None, map_location=None,
                    check_arch=True):
    """
    加载 checkpoint

    check_arch=True 时先比对架构标识，不匹配立即给出清晰报错，
    而不是让 load_state_dict 抛出一长串 key 不匹配。
    """
    # ⚠️ 这里**刻意**保留 weights_only=False，不要"顺手统一"改成 True：
    # 本函数只被 train.py 的 --resume 调用，加载的是含**优化器状态**的 latest.pth
    # （optimizer.state_dict() 带 int 键的嵌套结构，不在 weights_only 的安全白名单语义内）。
    # 而推理/评估路径加载的是 best.pth（只有张量+元信息），那三处已改用
    # weights_only=True —— 它们才是"下载别人的权重来跑"的高风险路径。
    ckpt = torch.load(path, map_location=map_location or DEVICE, weights_only=False)

    ckpt_arch = ckpt.get('arch')
    model_arch = getattr(model, 'arch', None)
    if check_arch and ckpt_arch and model_arch and ckpt_arch != model_arch:
        raise RuntimeError(
            f"架构不匹配: checkpoint 是 '{ckpt_arch}'，当前模型是 '{model_arch}'。\n"
            f"  请用 --arch {ckpt_arch} 重新运行，或改 config.MODEL_ARCH。\n"
            f"  两套架构的权重完全不通用，必须从对应的 checkpoint 继续。")

    model.load_state_dict(ckpt['model_state_dict'])
    if optimizer is not None and 'optimizer_state_dict' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    return ckpt


def load_tokenizer_for_weights(weights_path, ckpt=None):
    """
    加载与权重配套的词表

    词表就放在权重同目录（train.py 会生成 runs/v{N}/vocab.json），
    这里不再回退到任何共享路径 —— 拿错词表会让输出全错却不报错，
    所以宁可缺文件时直接失败。

    Args:
        weights_path: .pth 权重路径
        ckpt: 已加载的 checkpoint（可选）。提供时会校验 vocab_size 是否一致。
    Returns:
        LaTeXTokenizer
    """
    from tokenizer import LaTeXTokenizer

    vocab_path = os.path.join(os.path.dirname(weights_path) or '.', 'vocab.json')
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(
            f"未找到与权重配套的词表: {vocab_path}\n"
            f"  词表与权重放在同一目录（train.py 会生成 runs/v{{N}}/vocab.json）。\n"
            f"  请把训练时该版本目录下的 vocab.json 和权重一起拷贝过来。")

    tokenizer = LaTeXTokenizer.load(vocab_path)

    # 用 checkpoint 里记的 vocab_size 交叉验证，防止误配了别的版本的词表
    if ckpt is not None:
        ckpt_vocab = ckpt.get('vocab_size')
        if ckpt_vocab is not None and ckpt_vocab != tokenizer.vocab_size:
            raise RuntimeError(
                f"词表与权重不匹配: checkpoint 记录 vocab_size={ckpt_vocab}，"
                f"但 {vocab_path} 是 {tokenizer.vocab_size}。\n"
                f"  这两个文件必须来自同一次训练。")
    return tokenizer


# ==================== 版本管理 ====================

def get_version_dir(base_dir, version=None):
    """
    获取版本目录 runs/v{N}/
    version=None 时自动递增
    """
    os.makedirs(base_dir, exist_ok=True)
    if version is not None:
        v_dir = os.path.join(base_dir, f"v{version}")
        os.makedirs(v_dir, exist_ok=True)
        return v_dir
    # 自动递增
    existing = [d for d in os.listdir(base_dir)
                if d.startswith('v') and d[1:].isdigit()]
    next_v = max([int(d[1:]) for d in existing], default=0) + 1
    v_dir = os.path.join(base_dir, f"v{next_v}")
    os.makedirs(v_dir, exist_ok=True)
    return v_dir


# ==================== LaTeX 渲染 ====================
# 使用 matplotlib 自带的 mathtext 引擎，无需安装 TeX 发行版。
# mathtext 只支持 LaTeX 数学模式的一个子集，因此采用逐级降级的策略：
#   原样渲染 → 清洗排版类命令 → 剔除未知命令 → 退化为纯文本

# \begin{array}{c} / \begin{cases} 等环境（mathtext 不支持）
_BEGIN_ENV_RE = re.compile(r'\\begin\s*\{[^}]*\}(?:\s*\{[^}]*\}){0,2}')
_END_ENV_RE = re.compile(r'\\end\s*\{[^}]*\}')

# 纯排版命令，删掉不影响公式语义
_LAYOUT_CMD_RE = re.compile(
    r'\\(?:displaystyle|textstyle|scriptstyle|scriptscriptstyle'
    r'|limits|nolimits'
    r'|big|Big|bigg|Bigg|bigl|bigr|Bigl|Bigr|biggl|biggr|Biggl|Biggr'
    r'|small|tiny|footnotesize|scriptsize|normalsize'
    r'|large|Large|LARGE|huge|Huge)\b')

_CMD_RE = re.compile(r'\\[a-zA-Z]+')

# mathtext 解析失败时报 "Unknown symbol: \xxx"，据此摘掉不认识的部分
_UNKNOWN_CMD_RE = re.compile(r'Unknown symbol: (\\[a-zA-Z]+)')
# 兜底：报错信息里带出错位置，如 "Expected \text, found 'circled' (at char 5)"
# （\textcircled 被 mathtext 误当成 \text 前缀），按位置反查并摘掉该命令
_ERR_CHAR_RE = re.compile(r'at char (\d+)')

# CJK 字符（含扩展区）
_CJK_RE = re.compile(r'[⺀-鿿豈-﫿]')

# 候选中文字体，按平台可用性依次尝试
_CJK_FONT_CANDIDATES = (
    'SimHei', 'Microsoft YaHei', 'SimSun',              # Windows
    'Noto Sans CJK SC', 'Noto Sans CJK JP', 'Source Han Sans SC',
    'WenQuanYi Zen Hei', 'WenQuanYi Micro Hei',          # Linux
    'PingFang SC', 'Heiti SC', 'STHeiti',                # macOS
)
_cjk_font_cache = []


def _find_cjk_font():
    """找一个系统里可用的中文字体名；找不到返回 None"""
    if _cjk_font_cache:
        return _cjk_font_cache[0]
    from matplotlib import font_manager
    found = None
    for name in _CJK_FONT_CANDIDATES:
        try:
            font_manager.findfont(name, fallback_to_default=False)
            found = name
            break
        except Exception:
            continue
    _cjk_font_cache.append(found)
    return found


def _sanitize_latex(latex: str) -> str:
    """清洗 mathtext 不支持的排版命令，尽量保留公式内容"""
    s = _BEGIN_ENV_RE.sub(' ', latex)
    s = _END_ENV_RE.sub(' ', s)
    s = _LAYOUT_CMD_RE.sub(' ', s)
    s = s.replace('\\\\', ' ')          # 数组/表格的换行符
    # mathtext 把 % 和 # 当作注释/特殊符，会吞掉后半段，必须转义
    s = re.sub(r'(?<!\\)%', r'\\%', s)
    s = re.sub(r'(?<!\\)#', r'\\#', s)
    s = re.sub(r'\s+', ' ', s).strip()
    s = re.sub(r'\{\s*\}', '', s)       # 去掉空组，避免多余的空间
    return s.strip()


def _mono_family():
    """
    显示 LaTeX 源码用的字体族：等宽字体 + 中文回退
    注意必须作为 family 参数显式传给 text()，配 font.monospace 这个
    rcParam 实测不生效（中文仍会渲染成方框）
    """
    cjk = _find_cjk_font()
    return ['DejaVu Sans Mono', cjk] if cjk else 'monospace'


def _font_rc(latex: str, is_math: bool) -> dict:
    """
    按内容选择字体配置
    - 普通公式：用 matplotlib 默认的 mathtext 字体（数学排版最佳）
    - 含中文：切到中文字体，否则中文会渲染成方框；代价是 \\alpha 等
      希腊字母会退化成拉丁形近字母，因此模式标签会标出 'cjk' 提醒
    """
    rc = {}
    if not _CJK_RE.search(latex or ''):
        return rc

    cjk = _find_cjk_font()
    if not cjk:
        return rc

    if is_math:
        rc.update({'mathtext.fontset': 'custom', 'mathtext.rm': cjk,
                   'mathtext.it': cjk, 'mathtext.bf': cjk})
    return rc


def _typeset(body: str, is_math: bool, fontsize: int,
             dpi: int, padding: int) -> np.ndarray:
    """
    排版一段文本并栅格化为 RGB 图像（白底黑字）
    Args:
        body: 文本内容；is_math=True 时按数学模式（需自带 $ 包裹）处理
    Returns:
        np.ndarray (H, W, 3) uint8, RGB
    """
    import matplotlib
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    s = f"${body}$" if is_math else body

    with matplotlib.rc_context(_font_rc(body, is_math)):
        fig = Figure(figsize=(1, 1), dpi=dpi)
        FigureCanvasAgg(fig)                   # 绑定 Agg canvas，无需 pyplot
        fig.patch.set_facecolor('white')

        # va='bottom' 必须显式指定：默认的 'baseline' 会把 \frac 的分母、
        # 上下标等降到基线以下的部分裁掉
        kwargs = {} if is_math else {'family': _mono_family()}
        txt = fig.text(0.0, 0.0, s, fontsize=fontsize, color='black',
                       ha='left', va='bottom', **kwargs)

        # 第一次绘制：量出文本实际占用的像素尺寸
        fig.canvas.draw()
        bbox = txt.get_window_extent(fig.canvas.get_renderer())

        w = max(int(bbox.width + 2 * padding), 1)
        h = max(int(bbox.height + 2 * padding), 1)

        # 超长公式降低 dpi，避免生成上万像素的巨图
        if w > 6000:
            scale = 6000 / w
            dpi = max(int(dpi * scale), 60)
            w = max(int(w * scale), 1)
            h = max(int(h * scale), 1)

        # 按文本尺寸重建画布，并把文本摆到带 padding 的位置
        fig.set_size_inches(w / dpi, h / dpi)
        txt.set_position((padding / w, padding / h))

        # 第二次绘制：出图
        fig.canvas.draw()
        arr = np.asarray(fig.canvas.buffer_rgba())   # (H, W, 4) uint8, RGBA
        return np.ascontiguousarray(arr[..., :3]).copy()


def _strip_cmd_at(text, pos):
    """删除 text 中位于 pos 处的 \\命令（退化时删单个字符）；无法删除返回 None"""
    if pos < 0 or pos >= len(text):
        return None
    start = text.rfind('\\', 0, pos + 1)
    if start != -1:
        end = start + 1
        while end < len(text) and text[end].isalpha():
            end += 1
        if end > start + 1:                 # 确实是一段 \command
            return text[:start] + ' ' + text[end:]
    return text[:pos] + ' ' + text[pos + 1:]


def _typeset_repair(body, fontsize, dpi, padding, max_repair=16):
    """
    渲染数学模式；mathtext 解析失败时，摘掉出错的那个命令后重试。
    比维护白名单更忠实 —— 只丢弃引擎确实不认识的部分，其余原样保留
    Returns:
        (image, 实际渲染的字符串)
    """
    current = body
    last_err = None
    for _ in range(max_repair):
        try:
            return _typeset(current, True, fontsize, dpi, padding), current
        except Exception as e:
            last_err = e

            # 首选：报错直接点名了未知命令
            m = _UNKNOWN_CMD_RE.search(str(e))
            if m:
                cmd = m.group(1)
                nxt = re.sub(re.escape(cmd) + r'(?![a-zA-Z])', ' ', current)
            else:
                # 兜底：按报错位置反查命令（如 \textcircled 被当成 \text 前缀）
                m2 = _ERR_CHAR_RE.search(str(e))
                if not m2:
                    raise
                nxt = _strip_cmd_at(current, int(m2.group(1)))
                if nxt is None:
                    raise

            nxt = re.sub(r'\s+', ' ', nxt).strip()
            if nxt == current:              # 摘不掉，避免死循环
                raise
            current = nxt
    raise last_err


def render_latex_image(latex, fontsize=28, dpi=200, padding=12):
    """
    把 LaTeX 字符串渲染成图像（白底黑字）
    逐级降级: 清洗后的 LaTeX → 摘掉不支持的命令 → 纯文本
    Args:
        latex: str, LaTeX 字符串
        fontsize: 字号
        dpi: 渲染分辨率
        padding: 四周留白（像素）
    Returns:
        (image, mode)  image: np.ndarray (H, W, 3) uint8 或 None
                       mode:  'latex' | 'latex+cjk' | 'partial' | 'text'
                              | 'empty' | 'fail'
                       'partial' 表示丢弃过 mathtext 不认识的部分，
                       'text' 表示完全无法排版、退化为等宽源码
    """
    if latex is None or not str(latex).strip():
        return None, 'empty'

    latex = str(latex).strip()
    cleaned = _sanitize_latex(latex)
    has_cjk = bool(_CJK_RE.search(cleaned))

    try:
        img, used = _typeset_repair(cleaned, fontsize, dpi, padding)
    except Exception:
        # 数学模式彻底失败：退化为等宽纯文本，至少能看到内容
        try:
            return _typeset(latex, False, fontsize, dpi, padding), 'text'
        except Exception:
            return None, 'fail'

    if used != cleaned:
        mode = 'partial'
    else:
        mode = 'latex+cjk' if has_cjk else 'latex'
    return img, mode


# ==================== 可视化 ====================

def _wrap_code(code: str, width: int = 88, max_lines: int = 3) -> str:
    """
    把过长的 LaTeX 源码折行显示。

    限制行数很重要：模型未训练好时可能吐出几百个 token 的退化序列，
    不截断的话标题会占满整个画布、把下面的图像面板挤没。
    """
    import textwrap
    lines = textwrap.wrap(code, width=width) or ['']
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1][:width - 1] + '…'
    return '\n'.join(lines)


def draw_prediction(image_np, pred_latex, gt_latex=None, save_path=None,
                    render=True, fontsize=26, dpi=110):
    """
    可视化公式识别结果：左=输入图像，中=预测 LaTeX 渲染，右=真实 LaTeX 渲染（可选）
    顶部标题显示 LaTeX 源码
    Args:
        image_np: numpy array (H, W) 灰度 或 (H, W, 3)
        pred_latex: str, 预测 LaTeX
        gt_latex: str, 真实 LaTeX (可选，用于对比)
        save_path: str, 保存路径 (可选)
        render: 是否渲染公式图像（False 时只显示源码）
        fontsize: 公式渲染字号
        dpi: 输出分辨率
    Returns:
        dict: {'pred_mode': 渲染方式, 'gt_mode': 渲染方式}
    """
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    pred_latex = '' if pred_latex is None else str(pred_latex)
    has_gt = gt_latex is not None and str(gt_latex).strip() != ''
    gt_latex = str(gt_latex) if has_gt else None
    ncols = 3 if has_gt else 2

    # 渲染公式
    info = {'pred_mode': None, 'gt_mode': None}
    pred_img = pred_mode = None
    gt_img = gt_mode = None
    if render:
        pred_img, pred_mode = render_latex_image(pred_latex, fontsize=fontsize,
                                                 dpi=dpi * 2)
        info['pred_mode'] = pred_mode
        if has_gt:
            gt_img, gt_mode = render_latex_image(gt_latex, fontsize=fontsize,
                                                 dpi=dpi * 2)
            info['gt_mode'] = gt_mode

    match = has_gt and pred_latex.strip() == gt_latex.strip()

    # 显示 LaTeX 源码用等宽字体 + 中文回退（HME100K 里有少量含中文的公式）
    mono = _mono_family()

    fig = Figure(figsize=(4.2 * ncols, 3.4), dpi=dpi)
    FigureCanvasAgg(fig)
    fig.patch.set_facecolor('white')

    gs = fig.add_gridspec(1, ncols, wspace=0.08,
                          left=0.03, right=0.97, top=0.72, bottom=0.06)

    # ---- 面板 1：输入图像 ----
    ax0 = fig.add_subplot(gs[0, 0])
    if image_np is not None:
        ax0.imshow(image_np, cmap='gray' if image_np.ndim == 2 else None)
    ax0.set_title('Input', fontsize=11)
    ax0.axis('off')

    # ---- 面板 2：预测公式渲染 ----
    ax1 = fig.add_subplot(gs[0, 1])
    if pred_img is not None:
        ax1.imshow(pred_img)
        sub = '' if pred_mode == 'latex' else f'  [{pred_mode}]'
    else:
        ax1.text(0.5, 0.5, _wrap_code(pred_latex or '<empty>'), ha='center',
                 va='center', fontsize=12, family=mono, wrap=True)
        sub = '  [no render]'
    ax1.set_title(f'Prediction{sub}', fontsize=11,
                  color='#1a7f37' if match else '#24292f')
    ax1.axis('off')

    # ---- 面板 3：真实公式渲染 ----
    if has_gt:
        ax2 = fig.add_subplot(gs[0, 2])
        if gt_img is not None:
            ax2.imshow(gt_img)
            sub = '' if gt_mode == 'latex' else f'  [{gt_mode}]'
        else:
            ax2.text(0.5, 0.5, _wrap_code(gt_latex), ha='center', va='center',
                     fontsize=12, family=mono, wrap=True)
            sub = '  [no render]'
        ax2.set_title(f'Ground Truth{sub}', fontsize=11, color='#57606a')
        ax2.axis('off')

    # ---- 顶部：LaTeX 源码 + 匹配状态 ----
    if has_gt:
        head = ('[MATCH]' if match else '[MISMATCH]') + '\n'
        head_color = '#1a7f37' if match else '#cf222e'
    else:
        head, head_color = '', '#24292f'

    # LaTeX 源码原样展示；转义 $ 以免被 matplotlib 当成数学模式
    code_txt = _wrap_code(pred_latex or '<empty>')
    fig.text(0.03, 0.985, head + 'LaTeX: ' + code_txt.replace('$', r'\$'),
             ha='left', va='top', fontsize=10, family=mono,
             color=head_color, linespacing=1.5)

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        fig.savefig(save_path, dpi=dpi, facecolor='white')

    return info


def save_latex_image(latex, save_path, fontsize=28, dpi=200):
    """
    单独保存一张公式渲染图（透明背景可选项不做，统一白底）
    Returns: (bool 是否成功, str mode)
    """
    from PIL import Image

    img, mode = render_latex_image(latex, fontsize=fontsize, dpi=dpi)
    if img is None:
        return False, mode
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    Image.fromarray(img).save(save_path)
    return True, mode


# ==================== 测试 ====================

if __name__ == "__main__":
    print("=" * 50)
    print("utils 测试")
    print("=" * 50)

    # BLEU 测试
    preds = [r"\frac { a } { b }", "x + 1 = 2"]
    refs = [r"\frac { a } { b }", "x + 1 = 3"]
    bleu = compute_bleu(preds, refs)
    print(f"BLEU: {bleu:.4f}")

    # 编辑距离
    ed = compute_edit_distance(preds, refs)
    print(f"编辑距离: {ed:.4f}")

    # 精确匹配率
    er = compute_exprate(preds, refs)
    print(f"精确匹配率: {er:.4f}")

    # AverageMeter
    meter = AverageMeter('loss')
    for v in [0.5, 0.3, 0.2]:
        meter.update(v)
    print(f"AverageMeter avg: {meter.avg:.4f}")

    print("\n[OK] 测试通过")
