"""
公式识别系统 — 推理脚本
单图推理 + 批量推理 + 公式渲染 + LaTeX 导出
"""
import os
import re
import json
import argparse
import torch
import numpy as np

from config import DEVICE, IMG_HEIGHT, IMG_MAX_WIDTH, MODEL_ARCH
from model import FormulaRecognizer
from dataset import preprocess_image
from utils import (
    draw_prediction, save_latex_image, load_tokenizer_for_weights,
    check_dependencies,
)


IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp')


class FormulaPredictor:
    """
    公式识别推理类
    用法:
        predictor = FormulaPredictor('runs/v1/best.pth')
        latex = predictor.predict('test.jpg')          # 只要 LaTeX 代码
        pred, out = predictor.predict_and_visualize('test.jpg')   # 代码 + 渲染图
    """

    def __init__(self, weights_path, device=None, arch=None):
        self.device = device or DEVICE

        # 加载 checkpoint，架构优先以 checkpoint 里的标识为准
        # （两套架构的 state_dict 完全不通用，用错只会报 key 不匹配）
        # weights_only=True：只反序列化张量与基本类型，不执行 pickle 里的任意代码。
        # best.pth 的条目（张量 / 字符串 / 数字 / 列表）都在安全白名单内，实测可加载。
        # 这是「下载别人发布的权重来推理」的主路径，用安全模式最有价值。
        ckpt = torch.load(weights_path, map_location=self.device, weights_only=True)
        ckpt_arch = ckpt.get('arch')
        self.arch = arch or ckpt_arch or MODEL_ARCH
        if ckpt_arch and arch and ckpt_arch != arch:
            print(f"[WARN] checkpoint 架构是 '{ckpt_arch}'，与指定的 '{arch}' 不符，"
                  f"按 checkpoint 的架构加载")
            self.arch = ckpt_arch

        # 词表与权重同目录，并用 checkpoint 记录的 vocab_size 交叉校验
        self.tokenizer = load_tokenizer_for_weights(weights_path, ckpt)

        # 加载模型
        self.model = FormulaRecognizer(vocab_size=self.tokenizer.vocab_size,
                                       arch=self.arch)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

    # ---------- 核心推理 ----------

    def predict(self, image_path, use_beam=False):
        """
        预测单张公式图像
        Args:
            image_path: 图像路径
            use_beam: 是否使用 Beam Search
        Returns:
            str: LaTeX 字符串
        """
        # 预处理图像（preprocess_image 内部已做等比缩放 + 宽度上限，缩小时用 INTER_AREA）
        image = preprocess_image(image_path, IMG_HEIGHT, IMG_MAX_WIDTH)

        # 转 tensor
        image_tensor = torch.from_numpy(image.astype(np.float32) / 255.0)
        image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        image_tensor = image_tensor.to(self.device)

        with torch.no_grad():
            if use_beam:
                pred_ids = self.model.beam_search(image_tensor)
            else:
                pred_ids = self.model(image_tensor)

        ids = pred_ids[0].cpu().tolist()
        return self.tokenizer.decode(ids, skip_special=True)

    # ---------- 单图：识别 + 可视化 ----------

    def predict_and_visualize(self, image_path, output_path=None,
                              gt_latex=None, use_beam=False,
                              save_formula=True, render=True):
        """
        推理并可视化
        Args:
            image_path: 输入图像路径
            output_path: 对比图输出路径 (默认同目录 _result.jpg)
            gt_latex: 真实 LaTeX (可选，用于对比)
            use_beam: 是否使用 Beam Search
            save_formula: 是否额外存一张纯公式渲染图 (_formula.png)
            render: 是否渲染公式 (False 则只显示 LaTeX 源码)
        Returns:
            tuple: (pred_latex, output_path, info)
        """
        import cv2

        pred_latex = self.predict(image_path, use_beam)

        base, ext = os.path.splitext(image_path)
        if output_path is None:
            output_path = f"{base}_result.jpg"

        # 读取原图用于可视化
        buf = np.fromfile(image_path, dtype=np.uint8)
        image = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
        if image is not None and len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 对比图：输入图 | 预测公式渲染 | 真实公式渲染(可选)
        info = draw_prediction(image, pred_latex, gt_latex, output_path,
                               render=render)

        # 单独的公式渲染图（跟随对比图所在目录，不要写回输入目录）
        if save_formula and render:
            stem = os.path.splitext(os.path.basename(image_path))[0]
            formula_path = os.path.join(
                os.path.dirname(output_path) or '.', f"{stem}_formula.png")
            ok, mode = save_latex_image(pred_latex, formula_path)
            info['formula_path'] = formula_path if ok else None
            info['formula_mode'] = mode

        return pred_latex, output_path, info

    # ---------- 批量：识别 + 导出 ----------

    def run_batch(self, input_dir, output_dir=None, use_beam=False,
                  export=('json', 'tex'), render=True, save_formula=True,
                  verbose=True):
        """
        批量推理整个目录
        Args:
            input_dir: 输入图像目录
            output_dir: 输出目录 (默认 <input_dir>/results)
            use_beam: 是否使用 Beam Search
            export: 需要导出的格式，可选 'json' / 'tex' / 'txt' / 'md'
            render: 是否渲染公式图
            save_formula: 是否为每张图存独立的公式渲染图
        Returns:
            list[dict]: 每条记录 {image, latex, result_path, formula_path, mode}
        """
        from tqdm import tqdm

        if output_dir is None:
            output_dir = os.path.join(input_dir, 'results')
        os.makedirs(output_dir, exist_ok=True)

        files = sorted([f for f in os.listdir(input_dir)
                        if f.lower().endswith(IMAGE_EXTS)])

        if verbose:
            print(f"找到 {len(files)} 张图像")

        records = []
        for fname in tqdm(files, desc="推理", disable=not verbose):
            fpath = os.path.join(input_dir, fname)
            stem = os.path.splitext(fname)[0]
            out_path = os.path.join(output_dir, f"{stem}_result.jpg")
            try:
                pred, _, info = self.predict_and_visualize(
                    fpath, out_path, use_beam=use_beam, render=render,
                    save_formula=save_formula)
                records.append({
                    'image': fname,
                    'latex': pred,
                    'result_path': out_path,
                    'formula_path': info.get('formula_path'),
                    'mode': info.get('pred_mode'),
                })
            except Exception as e:
                if verbose:
                    print(f"  错误: {fname} - {e}")
                records.append({'image': fname, 'latex': None, 'error': str(e)})

        # 导出
        export = set(export or [])
        if 'json' in export:
            p = os.path.join(output_dir, 'results.json')
            export_json(records, p)
            print(f"JSON 结果: {p}")
        if 'tex' in export:
            p = os.path.join(output_dir, 'results.tex')
            export_tex(records, p)
            print(f"LaTeX 导出: {p}  (用 xelatex 编译)")
        if 'txt' in export:
            p = os.path.join(output_dir, 'results.txt')
            export_txt(records, p)
            print(f"纯文本: {p}")
        if 'md' in export:
            p = os.path.join(output_dir, 'results.md')
            export_md(records, p)
            print(f"Markdown: {p}")

        print(f"可视化结果保存在: {output_dir}")
        return records


# ==================== 导出 ====================

def export_json(records, path):
    """导出为 JSON（含图像名 → LaTeX 的映射）"""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def export_txt(records, path):
    """导出为纯文本：每行 `图像名<TAB>LaTeX`，可直接用作训练标签"""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for r in records:
            if r.get('latex'):
                f.write(f"{r['image']}\t{r['latex']}\n")


def _tex_escape_note(latex):
    """检查预测结果里是否含会破坏数学环境的字符"""
    return '$' in (latex or '')


# 需要对齐的数学环境：其中的 & 是对齐符，必须保留
# 注意 \s* —— 标签是空格分隔的（写出来是 "\begin { array } { c }"），
# 不允许空格就会漏匹配，反而把 array 里的 & 错误转义掉
_ALIGN_ENV_RE = re.compile(
    r'\\begin\s*\{\s*(?:array|matrix|aligned|align|cases|split)')
# 数学环境（用于判断 \\ 是否落在环境内部）
_MATH_ENV_RE = re.compile(r'\\begin\s*\{[^}]*\}|\\end\s*\{[^}]*\}')


def _has_top_level_newline(latex):
    """
    公式里是否存在落在环境之外的 \\\\（equation* 中会报 no line here to end）
    用嵌套深度跟踪：只有深度为 0 的片段才算环境外，
    array/cases 内部的换行是合法的，不能包 gathered
    """
    depth = 0
    pos = 0
    for m in _MATH_ENV_RE.finditer(latex):
        if depth == 0 and re.search(r'\\\\', latex[pos:m.start()]):
            return True
        if m.group(0).startswith('\\begin'):
            depth += 1
        else:
            depth = max(depth - 1, 0)
        pos = m.end()
    return depth == 0 and bool(re.search(r'\\\\', latex[pos:]))


def _tex_safe_math(latex):
    """
    转义会破坏 LaTeX 编译的字符。
    % 和 # 在 LaTeX 中是注释符 / 参数符，数学模式下同样是特殊字符：裸 % 会把
    本行剩余内容连同后面的 \\end{equation*} 一起注释掉，导致文件无法编译，
    因此必须转义。& 只有在 array/matrix 等对齐环境里才有意义，按内容决定。
    """
    s = re.sub(r'(?<!\\)%', r'\\%', latex)
    s = re.sub(r'(?<!\\)#', r'\\#', s)
    if not _ALIGN_ENV_RE.search(s):
        s = re.sub(r'(?<!\\)&', r'\\&', s)
    return s


def _tex_wrap_math(latex):
    """把公式包进数学环境；顶层换行用 gathered 兜住，避免编译报错"""
    body = _tex_safe_math(latex)
    if _has_top_level_newline(body):
        return [r'\begin{equation*}', r'\begin{gathered}', body,
                r'\end{gathered}', r'\end{equation*}']
    return [r'\begin{equation*}', body, r'\end{equation*}']


def export_tex(records, path, standalone=True):
    """
    导出为 .tex
    standalone=True 时生成可直接编译的完整文档（pdflatex/xelatex 均可）
    """
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)

    lines = []
    if standalone:
        lines += [
            '% 公式识别系统 — 自动导出',
            '% 编译: xelatex results.tex',
            '%   （文档内含中文，pdflatex 无法处理，必须用 xelatex）',
            r'\documentclass[11pt]{article}',
            r'\usepackage{amsmath,amssymb}',
            r'\usepackage[margin=2.5cm]{geometry}',
            r'\usepackage{ctex}   % 未安装 ctex 时可注释掉本行，中文将无法显示',
            r'\begin{document}',
            r'\section*{公式识别结果}',
            '',
        ]

    for i, r in enumerate(records, 1):
        name = r['image']
        latex = r.get('latex')
        lines.append(f'% ---------- [{i}] {name} ----------')
        if not latex or not latex.strip():
            lines.append(f'% (无识别结果)')
            lines.append('')
            continue
        if _tex_escape_note(latex):
            # 含 $ 时不能放进数学环境，原样输出供人工检查
            lines.append(f'% 含 $ 字符，未放入数学环境')
            lines.append(latex)
            lines.append('')
            continue
        lines += _tex_wrap_math(latex) + ['']

    if standalone:
        lines.append(r'\end{document}')

    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


def export_md(records, path):
    """导出为 Markdown：表格列出图像与 LaTeX 代码"""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    lines = ['# 公式识别结果', '',
             '| # | 图像 | LaTeX |', '|---|------|-------|']
    for i, r in enumerate(records, 1):
        latex = (r.get('latex') or '').replace('|', r'\|')
        lines.append(f"| {i} | {r['image']} | `{latex}` |")
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


# ==================== CLI ====================

def parse_args():
    parser = argparse.ArgumentParser(description='公式识别系统 — 推理')
    parser.add_argument('--weights', type=str, required=True,
                        help='模型权重路径 (.pth)')
    parser.add_argument('--input', type=str, required=True,
                        help='输入图像路径或目录')
    parser.add_argument('--arch', type=str, default=None,
                        choices=['vit_transformer', 'cnn_lstm'],
                        help=f'模型架构（默认取 checkpoint 标识，其次 config.MODEL_ARCH={MODEL_ARCH}）')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='输出目录 (默认: 输入同目录)')
    parser.add_argument('--beam', action='store_true',
                        help='使用 Beam Search')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--no_render', action='store_true',
                        help='不渲染公式图像，只输出 LaTeX 源码')
    parser.add_argument('--no_formula_png', action='store_true',
                        help='不额外保存单张公式渲染图 (_formula.png)')
    parser.add_argument('--export', type=str, default='json,tex',
                        help='批量模式的导出格式，逗号分隔: json,tex,txt,md,none '
                             '(默认: json,tex)')
    return parser.parse_args()


if __name__ == "__main__":
    check_dependencies()      # tqdm / matplotlib / PIL 都是惰性导入，先探一遍
    args = parse_args()
    predictor = FormulaPredictor(args.weights, args.device, arch=args.arch)
    print(f"架构: {predictor.arch}")

    render = not args.no_render
    save_formula = not args.no_formula_png

    input_path = args.input
    output_dir = args.output_dir

    if os.path.isfile(input_path):
        # ---------- 单图推理 ----------
        # --output_dir 指的是目录，不能直接当输出文件名用
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            stem = os.path.splitext(os.path.basename(input_path))[0]
            out_path = os.path.join(output_dir, f"{stem}_result.jpg")
        else:
            out_path = None

        pred, out_path, info = predictor.predict_and_visualize(
            input_path, out_path, use_beam=args.beam,
            render=render, save_formula=save_formula)

        print("\n" + "=" * 60)
        print("LaTeX 代码:")
        print(f"  {pred}")
        print("=" * 60)
        print(f"对比图 (输入图 | 预测渲染 | 真实渲染): {out_path}")
        if info.get('formula_path'):
            print(f"公式渲染图: {info['formula_path']}")
        if info.get('pred_mode') and info['pred_mode'] != 'latex':
            print(f"注意: 渲染降级为 '{info['pred_mode']}' 模式，"
                  f"部分 mathtext 不支持的语法已被简化")

    elif os.path.isdir(input_path):
        # ---------- 批量推理 ----------
        export = [] if args.export.strip().lower() == 'none' \
            else [e.strip() for e in args.export.split(',') if e.strip()]

        predictor.run_batch(input_path, output_dir, use_beam=args.beam,
                            export=export, render=render,
                            save_formula=save_formula)
    else:
        print(f"输入路径不存在: {input_path}")
