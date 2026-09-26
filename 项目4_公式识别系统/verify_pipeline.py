"""
公式识别系统 — pipeline 验证脚本

在不训练的前提下，验证整条链路的正确性。核心是**标签泄漏检查**：
Transformer 解码器训练用的是单次并行前向 + 因果掩码，张量对齐极易 off-by-one，
一旦泄漏，模型会「抄答案」——训练 loss 正常下降，推理却完全无效。
这种错误不会报错、不会 NaN，只有专项测试才能发现。

用法:
    python verify_pipeline.py              # 结构检查 + 标签泄漏（快，无需数据）
    python verify_pipeline.py --overfit     # 追加过拟合测试（需要数据集）
    python verify_pipeline.py --arch cnn_lstm
"""
import argparse
import random
import sys
import time

import numpy as np
import torch

from config import (
    SOS_TOKEN, PAD_TOKEN, EOS_TOKEN, IMG_HEIGHT, VIT_HIDDEN, VIT_PATCH_SIZE,
    MAX_SEQ_LEN, MODEL_ARCH,
)
from model import FormulaRecognizer, build_2d_sincos_pos_embed


# ==================== 1. 标签泄漏检查 ====================

def check_label_leakage(arch, vocab_size=245, seed=0):
    """
    验证「位置 k 的预测只依赖 targets[0..k]」—— 两套架构经 training_step
    对齐后约定是统一的。

    做法：把 targets[j:] 换成随机值重算。位置 k < j 只看到 targets[0..k]
    （不含被篡改的部分），输出必须逐元素不变；位置 j 会看到 targets[j]，
    必须**确实变了** —— 否则说明模型根本没在读 targets，测试会假通过。
    """
    torch.manual_seed(seed)
    random.seed(seed)
    model = FormulaRecognizer(vocab_size=vocab_size, arch=arch)
    model.eval()          # 关掉 dropout，保证可复现
    model.to('cpu')

    B, T = 3, 16
    images = torch.randn(B, 1, IMG_HEIGHT, 240)
    targets = torch.randint(4, vocab_size, (B, T))
    targets[:, 0] = SOS_TOKEN
    targets[:, -1] = EOS_TOKEN

    j = 8                  # 从第 8 列起改标签

    def run(tg):
        with torch.no_grad():
            logits, tgts = model.training_step(images, tg, teacher_forcing_ratio=1.0)
        return logits.view(B, T - 1, vocab_size)

    out1 = run(targets)

    tampered = targets.clone()
    tampered[:, j:] = torch.randint(4, vocab_size, (B, T - j))
    out2 = run(tampered)

    # 位置 k 预测 targets[k+1]，依赖 targets[0..k]
    head_same = torch.allclose(out1[:, :j], out2[:, :j], atol=1e-5)
    tail_differs = not torch.allclose(out1[:, j:], out2[:, j:], atol=1e-5)

    max_delta_head = (out1[:, :j] - out2[:, :j]).abs().max().item()
    max_delta_tail = (out1[:, j:] - out2[:, j:]).abs().max().item()

    print(f'  [{arch}] 篡改 targets[{j}:] 后:')
    print(f'      位置 0..{j - 1} 最大差异 {max_delta_head:.3e}  → '
          f'{"无泄漏 OK" if head_same else "泄漏! FAIL"}')
    print(f'      位置 {j}.. 最大差异 {max_delta_tail:.3e}  → '
          f'{"依赖被正确切断 OK" if tail_differs else "未受影响!"}')

    if not head_same:
        print('      ✗ FAIL: 早期位置的输出受到了 targets[j:] 的影响（标签泄漏）')
        return False
    if not tail_differs:
        print('      ✗ FAIL: 改标签完全没影响输出，说明模型没在读 targets，'
              '本测试无法证实无泄漏')
        return False
    return True


# ==================== 2. 位置编码顺序检查 ====================

def check_pos_embed_ordering():
    """
    2D sin-cos 位置编码必须与 patch 的 flatten 顺序一致（行优先）。

    做法：同一行内的位置应共享"行分量"（前半维），同一列内应共享"列分量"（后半维）。
    顺序搞反的话 ViT 会把行列弄混，但不会报错 —— 属于静默错误。
    """
    rows, cols = 4, 7
    pe = build_2d_sincos_pos_embed(VIT_HIDDEN, rows, cols)
    assert pe.shape == (rows * cols, VIT_HIDDEN), pe.shape
    half = VIT_HIDDEN // 2

    ok = True
    for i in range(rows * cols):
        r, c = divmod(i, cols)
        # 与同行首个位置比较：行分量应相同
        row_head = pe[r * cols, :half]
        if not torch.allclose(pe[i, :half], row_head, atol=1e-6):
            ok = False
            print(f'      ✗ 索引 {i} (行{r}列{c}) 的行分量与同行不一致')
        # 与同列首个位置比较：列分量应相同
        col_tail = pe[c, half:]
        if not torch.allclose(pe[i, half:], col_tail, atol=1e-6):
            ok = False
            print(f'      ✗ 索引 {i} (行{r}列{c}) 的列分量与同列不一致')

    print(f'  [2D 位置编码] 网格 {rows}x{cols}，行优先顺序检查 → '
          f'{"OK" if ok else "FAIL"}')
    return ok


# ==================== 3. Padding mask 检查 ====================

def check_padding_mask(arch='vit_transformer'):
    """padding 区域的 patch 必须被标为 True（忽略）"""
    if arch != 'vit_transformer':
        print(f'  [padding mask] {arch} 不使用 mask，跳过')
        return True

    model = FormulaRecognizer(vocab_size=50, arch=arch)
    model.eval()

    patch = model.encoder.patch_size
    rows = IMG_HEIGHT // patch
    max_w = 320
    widths = torch.tensor([320, 160, 17])          # 满宽 / 半宽 / 刚过一个 patch
    cols = max_w // patch
    mask = model._patch_padding_mask(widths, cols)

    assert mask.shape == (3, rows * cols), mask.shape

    # 逐样本核对：前 valid_cols 列为 False，其余为 True
    ok = True
    for b, w in enumerate(widths.tolist()):
        valid = min((w + patch - 1) // patch, cols)
        m = mask[b].reshape(rows, cols)
        if m[:, :valid].any() or not m[:, valid:].all():
            ok = False
            print(f'      ✗ 样本{b} 宽度{w}: 期望前 {valid} 列有效，实际 mask 不符')

    print(f'  [padding mask] 宽度 {widths.tolist()} → patch 列 '
          f'{[(w + patch - 1) // patch for w in widths.tolist()]}'
          f'  {"OK" if ok else "FAIL"}')
    return ok


# ==================== 4. 过拟合测试 ====================

def check_overfit(arch, n_samples=12, steps=200, lr=3e-4):
    """
    小样本过拟合：整条链路（tokenizer → dataset → model → loss → decode）
    任何一环有问题都会体现为 ExpRate 上不去。
    """
    from dataset import get_train_loader
    from config import PAD_TOKEN as _PAD, DEVICE as _DEV

    print(f'  [{arch}] 取 {n_samples} 个真实样本过拟合 {steps} 步 (设备 {_DEV}) ...')
    loader, tokenizer = get_train_loader(batch_size=n_samples, num_workers=0)
    batch = next(iter(loader))

    # 注意：模型和数据都必须搬到 _DEV，否则会静默地在 CPU 上跑（慢几十倍）
    images = batch['images'][:n_samples].to(_DEV)
    targets = batch['token_ids'][:n_samples].to(_DEV)
    widths = batch['widths'][:n_samples]
    refs = batch['latex_list'][:n_samples]

    model = FormulaRecognizer(vocab_size=tokenizer.vocab_size, arch=arch).to(_DEV)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    crit = torch.nn.CrossEntropyLoss(ignore_index=_PAD, label_smoothing=0.0)

    t0 = time.time()
    for step in range(steps):
        out, tgt = model.training_step(images, targets,
                                       teacher_forcing_ratio=1.0, widths=widths)
        loss = crit(out, tgt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (step + 1) % 50 == 0:
            per_step = (time.time() - t0) / (step + 1)
            print(f'      step {step + 1:3d}  loss {loss.item():.4f}  '
                  f'({per_step * 1000:.0f} ms/step)')

    model.eval()
    with torch.no_grad():
        preds = model(images, widths=widths)
    hit = 0
    for i in range(preds.size(0)):
        hyp = tokenizer.decode(preds[i].cpu().tolist(), skip_special=True)
        if hyp.strip() == refs[i].strip():
            hit += 1
    rate = hit / preds.size(0)
    print(f'      过拟合 ExpRate: {rate:.2%} ({hit}/{preds.size(0)})')
    print(f'      最终 loss: {loss.item():.4f}')

    ok = rate >= 0.5
    print(f'      → {"OK" if ok else "FAIL：连训练集都拟合不上，链路有问题"}')
    return ok


# ==================== 主流程 ====================

def main():
    ap = argparse.ArgumentParser(description='公式识别系统 — pipeline 验证')
    ap.add_argument('--arch', default=None, choices=['vit_transformer', 'cnn_lstm'],
                    help='只测指定架构（默认两套都测）')
    ap.add_argument('--overfit', action='store_true',
                    help='追加过拟合测试（需要数据集，较慢）')
    ap.add_argument('--steps', type=int, default=200)
    args = ap.parse_args()

    arches = [args.arch] if args.arch else ['vit_transformer', 'cnn_lstm']
    results = {}

    print('=' * 62)
    print('1. 标签泄漏检查（最关键）')
    print('=' * 62)
    for a in arches:
        results[f'leakage:{a}'] = check_label_leakage(a)

    print()
    print('=' * 62)
    print('2. 2D 位置编码顺序检查')
    print('=' * 62)
    results['pos_embed'] = check_pos_embed_ordering()

    print()
    print('=' * 62)
    print('3. Padding mask 检查')
    print('=' * 62)
    results['padding_mask'] = check_padding_mask('vit_transformer')

    if args.overfit:
        print()
        print('=' * 62)
        print('4. 过拟合测试')
        print('=' * 62)
        for a in arches:
            try:
                results[f'overfit:{a}'] = check_overfit(a, steps=args.steps)
            except Exception as e:
                print(f'  [{a}] 过拟合测试出错: {type(e).__name__}: {e}')
                results[f'overfit:{a}'] = False

    print()
    print('=' * 62)
    failed = [k for k, v in results.items() if not v]
    for k, v in results.items():
        print(f'  {"PASS" if v else "FAIL"}  {k}')
    print('=' * 62)
    if failed:
        print(f'{len(failed)} 项未通过: {failed}')
        sys.exit(1)
    print('全部通过')


if __name__ == '__main__':
    main()
