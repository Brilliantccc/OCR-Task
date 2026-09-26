"""
公式识别系统 — 训练脚本
支持: CROHME + HME100K 混合训练，双架构（vit_transformer / cnn_lstm）
特性: AMP 混合精度（含 torch 版本兼容）、ExpRate 早停、断点续训

主指标是 **表达式准确率 ExpRate**（预测与真实 LaTeX 完全一致的比例），
BLEU 作为辅助参考 —— 这是 MER 领域的标准口径，也是 best.pth 的选择依据。
"""
import os
import sys
import time
import json
import argparse
import random
from contextlib import nullcontext as _nullcontext
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from config import (
    DEVICE, BATCH_SIZE, BASE_LR, ENCODER_LR, EPOCHS, WARMUP_EPOCHS,
    GRAD_CLIP_MAX_NORM, LABEL_SMOOTHING, EARLY_STOP_PATIENCE,
    MODEL_DIR, PAD_TOKEN, WEIGHT_DECAY, get_teacher_forcing_ratio,
    PERSISTENT_WORKERS, PREFETCH_FACTOR, MODEL_ARCH, EVAL_USE_BEAM, BEAM_WIDTH,
)
from dataset import get_train_loader, get_eval_loader
from model import FormulaRecognizer
from utils import (
    AverageMeter, save_checkpoint, load_checkpoint, get_version_dir,
    compute_bleu, compute_edit_distance, compute_exprate, check_dependencies,
)

# ==================== AMP 兼容层 ====================
# torch.amp.GradScaler 是较新的统一 API。**已在 AutoDL 上实测确认**
# （torch 2.1.2+cu121）：
#     >>> from torch.amp import GradScaler
#     ImportError: cannot import name 'GradScaler' from 'torch.amp'
# 即统一命名空间的 GradScaler 在 2.1 上确实不存在，必须回退到 torch.cuda.amp。
# 本地开发机是 torch 2.13，直接 import 能过，所以这个坑只有上机才会暴露。
try:
    from torch.amp import autocast as _autocast, GradScaler as _GradScaler
    _NEW_AMP_API = True
except ImportError:                                    # torch < 2.3
    from torch.cuda.amp import autocast as _autocast, GradScaler as _GradScaler
    _NEW_AMP_API = False


def build_scaler(use_amp):
    """
    构造 GradScaler

    注意旧 API 的第一个**位置参数是 init_scale**，传 'cuda' 会直接报错，
    所以新旧的构造方式必须分开写。
    """
    if not use_amp:
        return None
    return _GradScaler('cuda') if _NEW_AMP_API else _GradScaler(enabled=True)


def autocast_ctx(use_amp):
    """返回 autocast 上下文（新旧 API 签名不同）"""
    if not use_amp:
        return _nullcontext()
    return _autocast('cuda', dtype=torch.float16) if _NEW_AMP_API else _autocast()


# ==================== Reproducibility ====================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False  # 性能优先
    torch.backends.cudnn.benchmark = True


# ==================== 学习率调度 ====================

def get_lr_scheduler(optimizer, warmup_epochs, total_epochs):
    """线性 warmup + Cosine Annealing"""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1 + np.cos(np.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ==================== 训练一个 epoch ====================

def train_one_epoch(model, loader, optimizer, scaler, criterion,
                    epoch, tf_ratio, use_amp=True, epochs=None, progress=True):
    model.train()
    loss_meter = AverageMeter('train_loss')

    desc = f"Epoch {epoch + 1}/{epochs}" if epochs else f"Epoch {epoch + 1}"
    # disable=None: 交互终端显示进度条，重定向到日志文件时自动关闭（避免刷屏）
    pbar = tqdm(loader, desc=desc, disable=None if progress else True,
                dynamic_ncols=True, leave=True)

    # 进度条被禁用时（nohup / 重定向到日志）退化为每 10% 打一行，
    # 否则一个 epoch 好几分钟完全没有中间反馈，无法判断是否卡住
    n_total = len(loader) if hasattr(loader, '__len__') else 0
    log_every = max(n_total // 10, 1)

    for step, batch in enumerate(pbar):
        images = batch['images'].to(DEVICE)
        targets = batch['token_ids'].to(DEVICE)
        widths = batch.get('widths')

        optimizer.zero_grad(set_to_none=True)

        with autocast_ctx(use_amp):
            # training_step 返回**已按架构对齐**的 (logits, targets)，
            # 两套架构的对齐方式不同（见 model.FormulaRecognizer.training_step）
            output, target = model.training_step(
                images, targets, teacher_forcing_ratio=tf_ratio, widths=widths)
            loss = criterion(output, target)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_MAX_NORM)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_MAX_NORM)
            optimizer.step()

        loss_meter.update(loss.item(), images.size(0))

        # 实时反馈：跑一个 epoch 要好几分钟，只看结尾那一行无法判断快慢或是否卡住
        pbar.set_postfix(loss=f'{loss_meter.avg:.4f}',
                         lr=f"{optimizer.param_groups[1]['lr']:.2e}",
                         refresh=False)
        if pbar.disable and n_total and (step + 1) % log_every == 0:
            print(f"  [{desc}] {step + 1}/{n_total}  loss {loss_meter.avg:.4f}",
                  flush=True)

    return loss_meter.avg


# ==================== 验证 ====================

@torch.no_grad()
def validate(model, loader, tokenizer, use_beam=None, progress=True):
    """
    返回 (exprate, bleu, edit_dist)

    注意 EVAL_USE_BEAM：默认用贪心解码（快），但这样选出的 best.pth 是
    贪心口径下最优，与推理时开 --beam 的口径不一致。要严格一致就设为 True。
    """
    use_beam = EVAL_USE_BEAM if use_beam is None else use_beam
    model.eval()
    all_preds = []
    all_refs = []

    desc = '验证(beam)' if use_beam else '验证'
    pbar = tqdm(loader, desc=desc, disable=None if progress else True,
                dynamic_ncols=True, leave=False)

    for batch in pbar:
        images = batch['images'].to(DEVICE)
        widths = batch.get('widths')

        if use_beam:
            # beam_search 只支持 batch_size=1，逐张解码
            pred_ids = [model.beam_search(images[i:i + 1],
                                          widths=None if widths is None else widths[i:i + 1])
                        for i in range(images.size(0))]
            for i, p in enumerate(pred_ids):
                all_preds.append(tokenizer.decode(p[0].cpu().tolist(), skip_special=True))
                all_refs.append(batch['latex_list'][i])
        else:
            pred_ids = model(images, widths=widths)   # greedy decode
            for i in range(pred_ids.size(0)):
                ids = pred_ids[i].cpu().tolist()
                all_preds.append(tokenizer.decode(ids, skip_special=True))
                all_refs.append(batch['latex_list'][i])

    return (compute_exprate(all_preds, all_refs),
            compute_bleu(all_preds, all_refs),
            compute_edit_distance(all_preds, all_refs))


# ==================== 训练主流程 ====================

def train(args=None):
    # 先探依赖再干活：nltk 等是惰性导入的，缺包会等到第一个 epoch
    # 跑完、开始验证时才炸，白白烧掉一整轮训练
    check_dependencies()

    set_seed()

    if args is None:
        args = parse_args()

    arch = args.arch or MODEL_ARCH
    batch_size = args.batch_size or BATCH_SIZE
    epochs = args.epochs or EPOCHS
    base_lr = args.lr or BASE_LR
    resume_path = args.resume

    version_dir = get_version_dir(MODEL_DIR, args.version)
    print("=" * 62)
    print(f"公式识别系统 — 训练")
    print(f"架构: {arch}")
    print(f"版本目录: {version_dir}")
    print(f"设备: {DEVICE}")
    if not _NEW_AMP_API:
        print(f"AMP: 使用旧版 torch.cuda.amp API（torch < 2.3）")
    print(f"Batch Size: {batch_size}")
    print(f"Epochs: {epochs}")
    print(f"LR: {base_lr} (encoder: {ENCODER_LR})")
    print(f"Warmup: {WARMUP_EPOCHS} epochs")
    if arch == 'cnn_lstm':
        print(f"Teacher Forcing: {get_teacher_forcing_ratio(0):.2f} → "
              f"{get_teacher_forcing_ratio(epochs):.2f}")
    else:
        print(f"Teacher Forcing: 不适用（Transformer 是单次并行前向 + 因果掩码）")
    print("=" * 62)

    # 数据
    train_loader, tokenizer = get_train_loader(batch_size=batch_size)
    eval_loader = get_eval_loader(batch_size=batch_size, tokenizer=tokenizer)
    print(f"训练集: {len(train_loader.dataset)} 样本")
    print(f"验证集: {len(eval_loader.dataset)} 样本")
    print(f"词表大小: {tokenizer.vocab_size}")

    tokenizer.save(os.path.join(version_dir, 'vocab.json'))

    # 模型
    model = FormulaRecognizer(vocab_size=tokenizer.vocab_size, arch=arch).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"模型参数量: {total_params:.1f}M")
    if hasattr(model.encoder, 'pretrain_summary'):
        print(f"编码器预训练: {model.encoder.pretrain_summary()}")

    # 优化器 — 分层学习率（encoder/decoder 两个参数组，顺序不可变）
    optimizer = torch.optim.AdamW([
        {'params': model.encoder.parameters(), 'lr': ENCODER_LR},
        {'params': model.decoder.parameters(), 'lr': base_lr},
    ], weight_decay=WEIGHT_DECAY)

    scheduler = get_lr_scheduler(optimizer, WARMUP_EPOCHS, epochs)
    use_amp = DEVICE == 'cuda'
    scaler = build_scaler(use_amp)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN, label_smoothing=LABEL_SMOOTHING)

    # 断点续训
    start_epoch = 0
    # best.pth 的选择依据：先比 ExpRate，ExpRate 相同时用 BLEU 决胜。
    # 初始化为 -1 而不是 0，否则 ExpRate 为 0 的早期 epoch 永远不会保存 best.pth
    # ——ExpRate 是离散指标，训练初期大量为 0，纯 > 判断会导致前若干轮
    # 完全没有 best.pth（用连续的 BLEU 时不会暴露这个问题）。
    best_exprate = -1.0
    best_bleu = -1.0
    train_history = {'train_loss': [], 'val_exprate': [], 'val_bleu': []}

    if resume_path:
        print(f"加载 checkpoint: {resume_path}")
        ckpt = load_checkpoint(resume_path, model, optimizer, map_location=DEVICE)
        start_epoch = ckpt['epoch'] + 1
        best_exprate = ckpt.get('val_exprate', -1.0)
        best_bleu = ckpt.get('val_bleu', -1.0)
        if 'train_history' in ckpt:
            train_history = ckpt['train_history']
        if 'scheduler_state_dict' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        else:
            for _ in range(max(0, start_epoch - 1)):
                scheduler.step()
        print(f"从 epoch {start_epoch} 继续，最佳 ExpRate: {best_exprate:.4f} "
              f"(BLEU {best_bleu:.4f})")

    patience_counter = 0

    # ---- 训练循环 ----
    for epoch in range(start_epoch, epochs):
        t0 = time.time()

        if epoch > 0:
            scheduler.step()

        tf_ratio = get_teacher_forcing_ratio(epoch)
        current_lr = optimizer.param_groups[1]['lr']

        train_loss = train_one_epoch(
            model, train_loader, optimizer, scaler, criterion, epoch, tf_ratio,
            use_amp=use_amp, epochs=epochs)

        val_exprate, val_bleu, val_ed = validate(model, eval_loader, tokenizer)

        train_history['train_loss'].append(train_loss)
        train_history['val_exprate'].append(val_exprate)
        train_history['val_bleu'].append(val_bleu)

        elapsed = time.time() - t0

        print(f"\nEpoch {epoch+1}/{epochs} ({elapsed:.1f}s)  "
              f"LR: {current_lr:.2e}" + (f"  TF: {tf_ratio:.2f}" if arch == 'cnn_lstm' else ""))
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val ExpRate: {val_exprate:.4f}  BLEU: {val_bleu:.4f}  "
              f"EditDist: {val_ed:.4f}")

        # latest.pth 带优化器状态，用于断点续训
        save_checkpoint(
            os.path.join(version_dir, 'latest.pth'),
            epoch, model, optimizer, val_bleu, train_loss,
            train_history, scheduler=scheduler, val_exprate=val_exprate,
            with_optimizer=True, with_history=True)

        # best.pth 以 ExpRate 为主判据，BLEU 仅在同分时决胜。
        # 早停计数也按这个复合判据重置 —— 不能只盯 ExpRate：
        # 它是离散指标，训练早期可能连续多轮都是 0（模型还在爬坡），
        # 严格按它计数会在第 10 轮就误触发早停，把还没起步的训练掐死。
        if (val_exprate, val_bleu) > (best_exprate, best_bleu):
            is_exprate_gain = val_exprate > best_exprate
            best_exprate, best_bleu = val_exprate, val_bleu
            patience_counter = 0
            # best.pth 只留模型权重 + 元信息：这是给推理/部署和下载到本地的，
            # 优化器状态（AdamW 约 826MB）对推理毫无用处，留着只会让文件大 3 倍。
            # 要续训请用 latest.pth。
            save_checkpoint(
                os.path.join(version_dir, 'best.pth'),
                epoch, model, optimizer, val_bleu, train_loss,
                train_history, scheduler=scheduler, val_exprate=val_exprate,
                with_optimizer=False, with_history=True)
            gain = "ExpRate" if is_exprate_gain else "BLEU 决胜"
            print(f"  ★ 新最佳模型({gain})! ExpRate: {best_exprate:.4f} "
                  f"(BLEU {val_bleu:.4f})")
        else:
            patience_counter += 1
            print(f"  早停计数: {patience_counter}/{EARLY_STOP_PATIENCE}")

        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"\n早停触发! 最佳 ExpRate: {best_exprate:.4f}")
            break

    history_path = os.path.join(version_dir, 'train_history.json')
    with open(history_path, 'w') as f:
        json.dump(train_history, f, indent=2)

    print(f"\n训练完成! 最佳 ExpRate: {best_exprate:.4f} (BLEU {best_bleu:.4f})")
    print(f"模型保存在: {version_dir}")

    return best_exprate


# ==================== Argparse ====================

def parse_args():
    parser = argparse.ArgumentParser(description='公式识别系统 — 训练')
    parser.add_argument('--arch', type=str, default=None,
                        choices=['vit_transformer', 'cnn_lstm'],
                        help=f'模型架构（默认取 config.MODEL_ARCH={MODEL_ARCH}）')
    parser.add_argument('--lr', type=float, default=None, help='学习率')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size')
    parser.add_argument('--epochs', type=int, default=None, help='训练轮数')
    parser.add_argument('--resume', type=str, default=None,
                        help='断点续训 checkpoint 路径')
    parser.add_argument('--version', type=int, default=None,
                        help='版本号 (默认自动递增)')
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
