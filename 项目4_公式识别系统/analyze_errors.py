"""
公式识别 — 错误分析

把预测错误自动归类，用来决定下一步该优化什么，而不是凭几个样例猜。

动机：H=64 那次从"错误全是单字形混淆"成功推出"分辨率不足"，那是因为该模式在
样例里压倒性地一致。但后续观察到的"大小写 / l↔1 混淆"只来自 2 个样例——
样本量根本不足以判断它占多大比重。本脚本给出真实分布。

用法:
    python analyze_errors.py --weights runs/v2/best.pth --split test
    python analyze_errors.py --weights runs/v2/best.pth --split test --show-samples 3
"""
import os
import argparse
import collections
import difflib

import torch

from config import DEVICE, MODEL_ARCH
from model import FormulaRecognizer
from dataset import get_eval_loader, get_test_loader
from utils import load_tokenizer_for_weights, _levenshtein_distance
from tokenizer import tokenize_latex

# 形近字符归一化表：用于判断"是否只差一对形近字符"
CONFUSABLE = str.maketrans({
    'l': '1', 'I': '1', 'O': '0', 'o': '0',
    'S': '5', 's': '5', 'Z': '2', 'z': '2',
    'B': '8', 'G': '6',
})

CATEGORIES = [
    ('correct',         '完全正确'),
    ('case_only',       '仅大小写差异'),
    ('confusable_only', '仅形近字符差异'),
    ('close',           '小错（token 编辑距离 ≤ 2）'),
    ('moderate',        '中等错（≤ 5）'),
    ('far',             '大错 / 结构错'),
]

BRACES = '{}'


def classify(pred_toks, ref_toks):
    """给一条预测归类"""
    if pred_toks == ref_toks:
        return 'correct'
    if [t.lower() for t in pred_toks] == [t.lower() for t in ref_toks]:
        return 'case_only'
    if ([t.translate(CONFUSABLE) for t in pred_toks]
            == [t.translate(CONFUSABLE) for t in ref_toks]):
        return 'confusable_only'
    # _levenshtein_distance 对列表同样适用（只用到 len / 索引 / 相等比较）
    d = _levenshtein_distance(pred_toks, ref_toks)
    if d <= 2:
        return 'close'
    if d <= 5:
        return 'moderate'
    return 'far'


def edit_ops(pred_toks, ref_toks):
    """
    用 SequenceMatcher 得到**真实**的编辑操作。

    不能用逐位置 zip —— 预测与参考长度不同时，分歧点之后全部错位，
    会产生大量虚假的"替换对"。
    Returns:
        (subs, n_ins, n_del)  subs 是 (参考 token, 预测 token) 列表
    """
    sm = difflib.SequenceMatcher(a=ref_toks, b=pred_toks, autojunk=False)
    subs, n_ins, n_del = [], 0, 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == 'replace':
            for k in range(max(i2 - i1, j2 - j1)):
                r = ref_toks[i1 + k] if i1 + k < i2 else '<无>'
                p = pred_toks[j1 + k] if j1 + k < j2 else '<无>'
                subs.append((r, p))
        elif tag == 'insert':
            n_ins += j2 - j1
        elif tag == 'delete':
            n_del += i2 - i1
    return subs, n_ins, n_del


def brace_only_diff(pred_toks, ref_toks):
    """是否「只有括号方向不同」：非括号 token 完全一致，且括号数量相同"""
    if pred_toks.count('{') + pred_toks.count('}') == 0:
        return False
    if [t for t in pred_toks if t not in BRACES] != \
       [t for t in ref_toks if t not in BRACES]:
        return False
    return (pred_toks.count('{') == ref_toks.count('{')
            and pred_toks.count('}') == ref_toks.count('}'))


@torch.no_grad()
def collect(model, loader, tokenizer, limit=None):
    model.eval()
    preds, refs, names = [], [], []
    for batch in loader:
        if limit and len(preds) >= limit:
            break
        images = batch['images'].to(DEVICE)
        widths = batch.get('widths')
        out = model(images, widths=widths)
        for j in range(out.size(0)):
            preds.append(tokenizer.decode(out[j].cpu().tolist(), skip_special=True))
        refs.extend(batch['latex_list'])
        names.extend(batch.get('name_list', [''] * out.size(0)))
    return preds, refs, names


def main():
    ap = argparse.ArgumentParser(description='公式识别 — 错误分析')
    ap.add_argument('--weights', required=True)
    ap.add_argument('--split', default='test', choices=['eval', 'test'])
    ap.add_argument('--limit', type=int, default=None, help='只分析前 N 条（快速试跑）')
    ap.add_argument('--show-samples', type=int, default=0,
                    help='每个类别额外展示 N 条样例')
    args = ap.parse_args()

    # weights_only=True：只反序列化张量与基本类型，不执行 pickle 里的任意代码
    ckpt = torch.load(args.weights, map_location=DEVICE, weights_only=True)
    arch = ckpt.get('arch') or MODEL_ARCH
    tokenizer = load_tokenizer_for_weights(args.weights, ckpt)

    model = FormulaRecognizer(vocab_size=tokenizer.vocab_size, arch=arch).to(DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    print(f"权重: {args.weights}   架构: {arch}   词表: {tokenizer.vocab_size}")

    if args.split == 'eval':
        loader = get_eval_loader(batch_size=32, num_workers=4, tokenizer=tokenizer)
        split_name = 'CROHME eval'
    else:
        loader = get_test_loader(batch_size=32, num_workers=4, tokenizer=tokenizer)
        split_name = 'HME100K test'

    preds, refs, names = collect(model, loader, tokenizer, args.limit)
    total = len(preds)
    print(f"分析 {total} 条（{split_name}）\n")

    stats = collections.Counter()
    samples = collections.defaultdict(list)
    subs_counter = collections.Counter()
    len_delta = collections.Counter()
    # 后处理上限分析
    same_multiset = 0        # token 多重集相同（只是顺序/方向错）
    brace_dir_only = 0       # 只有括号方向不同
    has_brace = 0            # 预测里含括号的错误
    unbalance_pred = 0       # 预测括号不配平
    unbalance_ref = 0        # 参考括号不配平（说明数据本身如此）

    for p, r in zip(preds, refs):
        pt, rt = tokenize_latex(p), tokenize_latex(r)
        c = classify(pt, rt)
        stats[c] += 1
        if c == 'correct':
            continue
        if len(samples[c]) < args.show_samples:
            samples[c].append((p, r))

        delta = len(pt) - len(rt)
        len_delta[delta] += 1
        subs, n_ins, n_del = edit_ops(pt, rt)
        for pair in subs:
            subs_counter[pair] += 1

        if sorted(pt) == sorted(rt):
            same_multiset += 1
        if brace_only_diff(pt, rt):
            brace_dir_only += 1
        if any(t in BRACES for t in pt):
            has_brace += 1
        if pt.count('{') != pt.count('}'):
            unbalance_pred += 1
        if rt.count('{') != rt.count('}'):
            unbalance_ref += 1

    err = total - stats['correct']

    print('=' * 70)
    print(f'{"类别":32s} {"条数":>8s} {"占全体":>9s} {"占错误":>9s}')
    print('-' * 70)
    for key, label in CATEGORIES:
        n = stats[key]
        # 'correct' 不属于错误，不参与"占错误"的比例
        share = '-' if key == 'correct' else f'{n / err * 100:.2f}%' if err else '-'
        print(f'{label:32s} {n:8d} {n / total * 100:8.2f}% {share:>9s}')
    print('-' * 70)
    print(f'{"总错误数":32s} {err:8d} {err / total * 100:8.2f}%')
    print('=' * 70)

    # ---------------- 后处理天花板 ----------------
    print('\n### 后处理能挽回多少（天花板分析）\n')
    print(f'  错误总数                                 {err:6d}')
    print(f'  其中 token 多重集与参考相同              {same_multiset:6d}  '
          f'({same_multiset / err * 100:5.1f}% of errors)')
    print(f'    → 这些是"元素都对、顺序/方向错了"，'
          f'理论上可由结构修复挽回')
    print(f'  其中「只有括号方向不同」                 {brace_dir_only:6d}  '
          f'({brace_dir_only / err * 100:5.1f}% of errors)')
    print(f'    → 最明确可修复的一类（{brace_dir_only / total * 100:.2f}% of all）')
    print(f'  预测里含括号的错误                       {has_brace:6d}  '
          f'({has_brace / err * 100:5.1f}% of errors)')
    print(f'  预测括号不配平                           {unbalance_pred:6d}  '
          f'({unbalance_pred / err * 100:5.1f}% of errors)')
    print(f'  参考括号不配平（数据本身如此）           {unbalance_ref:6d}')

    # ---------------- 长度差 ----------------
    if len_delta:
        print('\n--- 错误样本的长度差（预测 token 数 - 参考 token 数）---')
        for k in sorted(len_delta, key=lambda x: -len_delta[x])[:8]:
            n = len_delta[k]
            print(f'  {k:+3d}: {n:6d} 条  ({n / err * 100:5.1f}% of errors)')

    # ---------------- 真实替换对 ----------------
    print('\n--- 最常见的 token 替换（参考 -> 预测，经 SequenceMatcher 真实对齐）---')
    for (r_tok, p_tok), n in subs_counter.most_common(20):
        print(f'  {r_tok!r:14s} -> {p_tok!r:14s}  {n:6d} 次')

    # 括号方向单独统计
    l2r = subs_counter[('{', '}')]
    r2l = subs_counter[('}', '{')]
    if l2r or r2l:
        print(f'\n  其中括号方向错:  {{ -> }}  {l2r} 次   '
              f'}} -> {{  {r2l} 次   合计 {l2r + r2l}')

    if args.show_samples:
        for key, label in CATEGORIES:
            if key == 'correct' or not samples[key]:
                continue
            print(f'\n--- {label} 样例 ---')
            for p, r in samples[key]:
                print(f'  Pred: {p[:70]}')
                print(f'  GT:   {r[:70]}')


if __name__ == '__main__':
    main()
