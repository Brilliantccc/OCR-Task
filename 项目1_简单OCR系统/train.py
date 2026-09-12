"""
训练脚本 - tqdm进度条展示
支持：正常训练、断点续训、微调模式
"""
import os
import argparse
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

from config import (
    DEVICE, NUM_EPOCHS, LEARNING_RATE, LR_WARMUP_EPOCHS, LR_MIN,
    EARLY_STOP_PATIENCE, SAVE_INTERVAL, VERSION, VERSION_DIR, init_dirs
)
from dataset import get_data_loaders
from model import CRNN, weights_init
from utils import (
    decode_output, compute_accuracy, compute_edit_distance,
    save_checkpoint, load_checkpoint, AverageMeter, IDX_TO_CHAR
)


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, total_epochs):
    """训练一个epoch"""
    model.train()

    loss_meter = AverageMeter()
    char_acc_meter = AverageMeter()

    pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{total_epochs} [Train]')

    for batch_idx, (images, labels, label_lengths) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device)
        label_lengths = label_lengths.to(device)

        # 前向传播
        outputs = model(images)

        # 准备CTC Loss输入
        seq_lengths = torch.full((outputs.size(0),), outputs.size(1), dtype=torch.long).to(device)

        # CTC Loss
        loss = criterion(outputs.permute(1, 0, 2), labels, seq_lengths, label_lengths)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        # 计算准确率
        pred_texts = decode_output(outputs)
        gt_texts = []
        for i in range(labels.size(0)):
            gt_text = ''.join([IDX_TO_CHAR.get(c, '') for c in labels[i, :label_lengths[i]].cpu().numpy()])
            gt_texts.append(gt_text)

        char_acc, _ = compute_accuracy(pred_texts, gt_texts)

        # 更新统计
        loss_meter.update(loss.item(), images.size(0))
        char_acc_meter.update(char_acc, images.size(0))

        # 更新进度条
        pbar.set_postfix({
            'loss': f'{loss_meter.avg:.4f}',
            'acc': f'{char_acc_meter.avg:.4f}'
        })

    return {
        'loss': loss_meter.avg,
        'char_acc': char_acc_meter.avg
    }


def validate(model, val_loader, criterion, device, epoch, total_epochs):
    """验证"""
    model.eval()

    loss_meter = AverageMeter()
    char_acc_meter = AverageMeter()
    word_acc_meter = AverageMeter()
    edit_dist_meter = AverageMeter()

    pbar = tqdm(val_loader, desc=f'Epoch {epoch}/{total_epochs} [Val]')

    with torch.no_grad():
        for images, labels, label_lengths in pbar:
            images = images.to(device)
            labels = labels.to(device)
            label_lengths = label_lengths.to(device)

            # 前向传播
            outputs = model(images)

            # CTC Loss
            seq_lengths = torch.full((outputs.size(0),), outputs.size(1), dtype=torch.long).to(device)
            loss = criterion(outputs.permute(1, 0, 2), labels, seq_lengths, label_lengths)

            # 解码
            pred_texts = decode_output(outputs)
            gt_texts = []
            for i in range(labels.size(0)):
                gt_text = ''.join([IDX_TO_CHAR.get(c, '') for c in labels[i, :label_lengths[i]].cpu().numpy()])
                gt_texts.append(gt_text)

            # 计算指标
            char_acc, word_acc = compute_accuracy(pred_texts, gt_texts)
            edit_dist = compute_edit_distance(pred_texts, gt_texts)

            # 更新统计
            loss_meter.update(loss.item(), images.size(0))
            char_acc_meter.update(char_acc, images.size(0))
            word_acc_meter.update(word_acc, images.size(0))
            edit_dist_meter.update(edit_dist, images.size(0))

            # 更新进度条
            pbar.set_postfix({
                'loss': f'{loss_meter.avg:.4f}',
                'acc': f'{char_acc_meter.avg:.4f}'
            })

    return {
        'loss': loss_meter.avg,
        'char_acc': char_acc_meter.avg,
        'word_acc': word_acc_meter.avg,
        'edit_dist': edit_dist_meter.avg
    }


def main():
    """主训练函数"""
    parser = argparse.ArgumentParser(description='OCR Training')
    parser.add_argument('--resume', action='store_true', help='断点续训：从上次停止的地方继续训练')
    parser.add_argument('--finetune', type=str, default=None, help='微调模式：指定预训练模型路径')
    parser.add_argument('--finetune_lr', type=float, default=LEARNING_RATE * 0.1, help='微调学习率（默认为原学习率的0.1倍）')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS, help='训练轮数')
    args = parser.parse_args()

    # 初始化目录
    init_dirs()

    # 打印头部
    print("\n" + "=" * 60)
    print(f"  OCR Training - CRNN + CTC Loss (Version {VERSION})")
    print("=" * 60)

    # 设置设备
    device = torch.device(DEVICE)
    print(f"\n  Device: {device}")

    if device.type == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        print(f"  GPU: {gpu_name} ({gpu_mem:.1f} GB)")

    # 加载数据
    print("\n  Loading data...")
    train_loader, val_loader, test_loader = get_data_loaders()
    print(f"  Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)} | Test: {len(test_loader.dataset)}")

    # 创建模型
    model = CRNN().to(device)
    model.apply(weights_init)

    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model: {total_params:,} params ({trainable_params:,} trainable)")

    # 损失函数和优化器
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    start_epoch = 0
    best_acc = 0
    mode = "Normal"

    # 根据模式加载模型
    if args.finetune:
        # 微调模式：加载预训练模型，使用较小学习率
        mode = "Finetune"
        print(f"\n  Finetune mode: loading {args.finetune}")
        load_checkpoint(model, None, args.finetune)
        optimizer = AdamW(model.parameters(), lr=args.finetune_lr, weight_decay=1e-4)
        print(f"  Finetune LR: {args.finetune_lr}")
    elif args.resume:
        # 断点续训：加载checkpoint继续训练
        mode = "Resume"
        checkpoint_path = os.path.join(VERSION_DIR, "best_model.pth")
        if os.path.exists(checkpoint_path):
            start_epoch, best_acc = load_checkpoint(model, None, checkpoint_path)
            start_epoch += 1
            print(f"\n  Resuming from epoch {start_epoch}, best acc: {best_acc:.4f}")
        else:
            print(f"\n  No checkpoint found, starting from scratch")
        optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    else:
        # 正常训练
        optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    # 学习率调度：预热 + 余弦退火
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=LR_WARMUP_EPOCHS
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=num_epochs - LR_WARMUP_EPOCHS,
        eta_min=LR_MIN
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[LR_WARMUP_EPOCHS]
    )

    # 打印训练模式
    print(f"\n  Mode: {mode}")

    # 训练循环
    print("\n" + "=" * 60)
    print("  Starting training...")
    print("=" * 60)

    early_stop_counter = 0
    num_epochs = args.epochs
    best_val_acc = 0

    for epoch in range(start_epoch, num_epochs):
        # 训练
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch + 1, num_epochs)

        # 验证
        val_metrics = validate(model, val_loader, criterion, device, epoch + 1, num_epochs)

        # 更新学习率
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()

        # 打印epoch总结
        print(f"\n  Epoch {epoch+1}/{num_epochs} Summary:")
        print(f"  Train - Loss: {train_metrics['loss']:.4f} | Char Acc: {train_metrics['char_acc']:.4f}")
        print(f"  Val   - Loss: {val_metrics['loss']:.4f} | Char Acc: {val_metrics['char_acc']:.4f} | "
              f"Word Acc: {val_metrics['word_acc']:.4f}")
        print(f"  LR: {current_lr:.6f}")

        # 保存最佳模型
        if val_metrics['char_acc'] > best_val_acc:
            best_val_acc = val_metrics['char_acc']
            early_stop_counter = 0
            save_checkpoint(model, optimizer, epoch, best_val_acc,
                            os.path.join(VERSION_DIR, "best_model.pth"))
            print(f"  ★ New best model saved! Char Acc: {best_val_acc:.4f}")
        else:
            early_stop_counter += 1

        # 保存最后一个epoch
        save_checkpoint(model, optimizer, epoch, best_val_acc,
                        os.path.join(VERSION_DIR, "last.pth"))

        # 早停
        if early_stop_counter >= EARLY_STOP_PATIENCE:
            print(f"\n  ⚠ Early stopping! No improvement for {EARLY_STOP_PATIENCE} epochs")
            break

        print("-" * 60)

    # 最终测试
    print("\n" + "=" * 60)
    print("  Training Complete! Running final test...")
    print("=" * 60)

    load_checkpoint(model, optimizer, os.path.join(VERSION_DIR, "best_model.pth"))
    test_metrics = validate(model, test_loader, criterion, device, num_epochs, num_epochs)

    print(f"\n  Final Test Results:")
    print(f"  Char Acc:  {test_metrics['char_acc']:.4f}")
    print(f"  Word Acc:  {test_metrics['word_acc']:.4f}")
    print(f"  Edit Dist: {test_metrics['edit_dist']:.2f}")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
