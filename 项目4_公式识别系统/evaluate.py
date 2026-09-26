"""
公式识别系统 — 评估脚本
在 CROHME eval / HME100K test 上计算 ExpRate、BLEU、编辑距离

**主指标是 ExpRate**（表达式完全匹配率）—— MER 领域的标准口径。
BLEU-4 保留作为辅助参考，但它对公式这种结构性序列偏弱。

HME100K test 集有官方难度划分（easy/medium/hard），评估时按难度分档报告，
比单纯按长度分档有信息量得多。
"""
import os
import json
import argparse
import torch

from config import (
    DEVICE, MODEL_ARCH, HME100K_SUBSET_DIR, BEAM_WIDTH,
)
from model import FormulaRecognizer
from dataset import get_eval_loader, get_test_loader
from utils import (
    load_tokenizer_for_weights, compute_bleu, compute_edit_distance,
    compute_exprate, check_dependencies,
)

DIFFICULTY_LEVELS = ('easy', 'medium', 'hard')


def load_difficulty_map():
    """
    读取 HME100K 官方难度划分
    Returns:
        dict: {图像文件名: 'easy'|'medium'|'hard'}；文件缺失时返回空 dict
    """
    mapping = {}
    if not os.path.isdir(HME100K_SUBSET_DIR):
        return mapping
    for level in DIFFICULTY_LEVELS:
        path = os.path.join(HME100K_SUBSET_DIR, f'{level}.json')
        if not os.path.exists(path):
            continue
        try:
            with open(path, encoding='utf-8') as f:
                stems = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        for stem in stems:
            name = stem if os.path.splitext(stem)[1] else f'{stem}.jpg'
            mapping[name] = level
    return mapping


def _decode_all(model, loader, tokenizer, use_beam=False):
    """跑完整个 loader，返回 (preds, refs, names)"""
    model.eval()
    all_preds, all_refs, all_names = [], [], []

    with torch.no_grad():
        for batch in loader:
            images = batch['images'].to(DEVICE)
            widths = batch.get('widths')
            names = batch.get('name_list', [''] * images.size(0))

            if use_beam:
                # beam_search 只支持 batch_size=1
                for i in range(images.size(0)):
                    w = None if widths is None else widths[i:i + 1]
                    p = model.beam_search(images[i:i + 1], widths=w)
                    all_preds.append(tokenizer.decode(p[0].cpu().tolist(),
                                                      skip_special=True))
            else:
                pred_ids = model(images, widths=widths)   # greedy decode
                for i in range(pred_ids.size(0)):
                    all_preds.append(tokenizer.decode(pred_ids[i].cpu().tolist(),
                                                      skip_special=True))

            all_refs.extend(batch['latex_list'])
            all_names.extend(names)

    return all_preds, all_refs, all_names


def evaluate(model, loader, tokenizer, split_name="test", use_beam=False,
             difficulty_map=None):
    """评估模型并返回指标"""
    all_preds, all_refs, all_names = _decode_all(model, loader, tokenizer, use_beam)
    total = len(all_preds)
    if total == 0:
        print(f'\n{split_name}: 没有样本')
        return {'exprate': 0.0, 'bleu': 0.0, 'edit_dist': 0.0}

    bleu = compute_bleu(all_preds, all_refs)
    exprate = compute_exprate(all_preds, all_refs)
    edit_dist = compute_edit_distance(all_preds, all_refs)

    print(f"\n{'=' * 62}")
    print(f"评估结果 — {split_name} ({total} 样本)"
          + ("  [beam search]" if use_beam else "  [greedy]"))
    print(f"{'=' * 62}")
    # 主指标在前
    print(f"  ExpRate (主):  {exprate:.4f}   ({int(round(exprate * total))}/{total})")
    print(f"  BLEU-4:        {bleu:.4f}")
    print(f"  编辑距离:       {edit_dist:.4f}")

    # ---- 按 HME100K 官方难度分档 ----
    if difficulty_map:
        buckets = {lv: [[], []] for lv in DIFFICULTY_LEVELS}
        matched = 0
        for pred, ref, name in zip(all_preds, all_refs, all_names):
            lv = difficulty_map.get(name)
            if lv:
                buckets[lv][0].append(pred)
                buckets[lv][1].append(ref)
                matched += 1
        if matched:
            print(f"\n  按难度分档 (HME100K 官方划分, 覆盖 {matched}/{total}):")
            for lv in DIFFICULTY_LEVELS:
                preds, refs = buckets[lv]
                if preds:
                    print(f"    {lv:8s} {len(preds):6d} 样本, "
                          f"ExpRate: {compute_exprate(preds, refs):.4f}")

    # ---- 按长度分档（辅助）----
    short_p, short_r, mid_p, mid_r, long_p, long_r = [], [], [], [], [], []
    for pred, ref in zip(all_preds, all_refs):
        n_tokens = len(ref.split())
        if n_tokens < 15:
            short_p.append(pred); short_r.append(ref)
        elif n_tokens < 40:
            mid_p.append(pred); mid_r.append(ref)
        else:
            long_p.append(pred); long_r.append(ref)

    print(f"\n  按长度分档:")
    for label, preds, refs in (('短 (<15)', short_p, short_r),
                               ('中 (15-40)', mid_p, mid_r),
                               ('长 (>40)', long_p, long_r)):
        if preds:
            print(f"    {label:12s} {len(preds):6d} 样本, "
                  f"ExpRate: {compute_exprate(preds, refs):.4f}")

    print("\n--- 预测样例 ---")
    for i in range(min(5, total)):
        # 用 ASCII 标记：✓/✗ 不在 GBK 字符集内，Windows 中文控制台会
        # 在打印时抛 UnicodeEncodeError，让整个评估脚本看起来是失败的
        match = "[OK]" if all_preds[i].strip() == all_refs[i].strip() else "[X] "
        print(f"  {match} Pred: {all_preds[i][:60]}")
        print(f"       GT:   {all_refs[i][:60]}")
    print()

    return {'exprate': exprate, 'bleu': bleu, 'edit_dist': edit_dist}


def parse_args():
    parser = argparse.ArgumentParser(description='公式识别系统 — 评估')
    parser.add_argument('--weights', type=str, required=True,
                        help='模型权重路径 (.pth)')
    parser.add_argument('--arch', type=str, default=None,
                        choices=['vit_transformer', 'cnn_lstm'],
                        help=f'模型架构（默认取 config.MODEL_ARCH={MODEL_ARCH}）')
    parser.add_argument('--split', type=str, default='eval',
                        choices=['eval', 'test', 'all'],
                        help='评估集: eval(CROHME), test(HME100K), all(两者)')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--beam', action='store_true',
                        help='用 beam search 解码（慢，但通常更准）')
    return parser.parse_args()


if __name__ == "__main__":
    check_dependencies()      # 缺包就立刻报错，别等跑到算指标时才炸
    args = parse_args()
    arch = args.arch or MODEL_ARCH

    # weights_only=True：只反序列化张量与基本类型，不执行 pickle 里的任意代码
    ckpt = torch.load(args.weights, map_location=DEVICE, weights_only=True)

    # 架构一致性：优先信 checkpoint 里的标识，避免用错模型加载
    ckpt_arch = ckpt.get('arch')
    if ckpt_arch and ckpt_arch != arch:
        print(f"[WARN] checkpoint 架构是 '{ckpt_arch}'，当前参数是 '{arch}'，"
              f"按 checkpoint 的架构加载")
        arch = ckpt_arch

    # 词表与权重同目录，且用 checkpoint 里的 vocab_size 交叉校验
    tokenizer = load_tokenizer_for_weights(args.weights, ckpt)

    model = FormulaRecognizer(vocab_size=tokenizer.vocab_size, arch=arch).to(DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    print(f"加载模型: {args.weights}")
    print(f"架构: {arch}")
    expr = ckpt.get('val_exprate')
    bleu_ckpt = ckpt.get('val_bleu')
    print(f"Epoch: {ckpt.get('epoch', '?')}, "
          f"Val ExpRate: {'N/A' if expr is None else f'{expr:.4f}'}, "
          f"Val BLEU: {'N/A' if bleu_ckpt is None else f'{bleu_ckpt:.4f}'}")
    print(f"词表大小: {tokenizer.vocab_size}")

    difficulty_map = load_difficulty_map()
    if difficulty_map:
        counts = {lv: sum(1 for v in difficulty_map.values() if v == lv)
                  for lv in DIFFICULTY_LEVELS}
        print(f"HME100K 难度划分: " +
              ", ".join(f"{k} {v}" for k, v in counts.items()))

    if args.split in ('eval', 'all'):
        eval_loader = get_eval_loader(batch_size=args.batch_size, tokenizer=tokenizer)
        evaluate(model, eval_loader, tokenizer, "CROHME eval", use_beam=args.beam)

    if args.split in ('test', 'all'):
        test_loader = get_test_loader(batch_size=args.batch_size, tokenizer=tokenizer)
        evaluate(model, test_loader, tokenizer, "HME100K test", use_beam=args.beam,
                 difficulty_map=difficulty_map)
