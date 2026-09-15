"""
合并版训练脚本（兼容旧用法）
推荐使用独立的 train_detection.py / train_structure.py
用法: python train_combined.py --task detection [--resume] [--version N] [--backbone PATH] [--epochs N]
"""
import os
import time
import argparse
import torch

from config import (
    DEVICE, DETECTION_EPOCHS, DETECTION_LR, STRUCTURE_EPOCHS, STRUCTURE_LR,
    LR_WARMUP_EPOCHS, LR_MIN, EARLY_STOP_PATIENCE,
    DETECTION_NUM_CLASSES, STRUCTURE_NUM_CLASSES, STRUCTURE_NMS_THRESH,
    init_dirs
)
import config
from dataset import get_detection_loaders, get_structure_loaders
from model import TableDetector
from utils import save_checkpoint, load_checkpoint, save_history
from train_utils import get_lr, train_one_epoch, validate


def main():
    parser = argparse.ArgumentParser(description='Table Recognition Training')
    parser.add_argument('--task', type=str, default='detection',
                        choices=['detection', 'structure'])
    parser.add_argument('--resume', action='store_true', help='断点续训')
    parser.add_argument('--version', type=int, default=None, help='续训指定版本号')
    parser.add_argument('--finetune', type=str, default=None, help='微调模型路径')
    parser.add_argument('--backbone', type=str, default=None, help='结构识别: 迁移 backbone')
    parser.add_argument('--epochs', type=int, default=None, help='训练轮数')
    args = parser.parse_args()

    task = args.task
    is_detection = (task == 'detection')

    # 版本管理
    if args.resume and args.version:
        config.VERSION = args.version
        config.VERSION_DIR = os.path.join(config.MODEL_DIR, task, f"v{args.version}")
    elif args.resume:
        task_dir = os.path.join(config.MODEL_DIR, task)
        if os.path.exists(task_dir):
            versions = sorted([int(d[1:]) for d in os.listdir(task_dir) if d.startswith('v') and d[1:].isdigit()])
            if versions:
                config.VERSION = versions[-1]
                config.VERSION_DIR = os.path.join(task_dir, f"v{config.VERSION}")
            else:
                print("  [ERROR] No versions found, nothing to resume")
                return
        else:
            print("  [ERROR] Task directory not found, nothing to resume")
            return
    else:
        from config import init_version
        init_version(task=task)
    init_dirs()

    num_epochs = args.epochs or (DETECTION_EPOCHS if is_detection else STRUCTURE_EPOCHS)
    base_lr = DETECTION_LR if is_detection else STRUCTURE_LR
    task_name = "Table Detection" if is_detection else "Structure Recognition"
    num_classes = DETECTION_NUM_CLASSES if is_detection else STRUCTURE_NUM_CLASSES
    val_iou = 0.5 if is_detection else 0.3
    val_dynamic = not is_detection

    print("\n" + "=" * 60)
    print(f"  {task_name} (Faster R-CNN)")
    print(f"  Classes: {num_classes}")
    print(f"  Version: {config.VERSION}")
    print("=" * 60)

    device = torch.device(DEVICE)
    print(f"\n  Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB)")

    print("\n  Loading data...")
    if is_detection:
        train_loader, val_loader, test_loader = get_detection_loaders()
    else:
        train_loader, val_loader, test_loader = get_structure_loaders()
    print(f"  Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)} | Test: {len(test_loader.dataset)}")

    if is_detection:
        model = TableDetector(num_classes=num_classes, pretrained=True).to(device)
    else:
        model = TableDetector(num_classes=num_classes, pretrained=True,
                              detections_per_img=2000, nms_thresh=STRUCTURE_NMS_THRESH).to(device)
        if args.backbone:
            print(f"\n  Loading backbone from: {args.backbone}")
            model.load_backbone_from(args.backbone)

    print(f"  Model: {sum(p.numel() for p in model.parameters()):,} params")

    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    start_epoch = 0
    best_metric = 0

    if args.finetune:
        load_checkpoint(model, None, args.finetune)
        optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr * 0.1, weight_decay=1e-4)
        print(f"\n  Finetune from: {args.finetune}")
    elif args.resume:
        ckpt_path = os.path.join(config.VERSION_DIR, "best_model.pth")
        print(f"\n  Resuming: v{config.VERSION}")
        if os.path.exists(ckpt_path):
            start_epoch, best_metric = load_checkpoint(model, optimizer, ckpt_path)
            start_epoch += 1
            print(f"  From epoch {start_epoch}, best: {best_metric:.4f}")
        else:
            print("  [WARN] Checkpoint not found, starting from scratch")

    print(f"\n  Epochs: {num_epochs} | LR: {base_lr}")
    print("\n" + "=" * 60)

    early_stop_counter = 0
    history = []

    for epoch in range(start_epoch, num_epochs):
        epoch_start = time.time()
        lr = get_lr(epoch, base_lr, LR_WARMUP_EPOCHS, num_epochs, LR_MIN)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        train_metrics = train_one_epoch(model, train_loader, optimizer, device,
                                        epoch + 1, num_epochs, scaler=scaler, desc=f'{task[:3]} Train')
        val_metrics = validate(model, val_loader, device, epoch + 1, num_epochs,
                               iou_threshold=val_iou, use_dynamic=val_dynamic, desc=f'{task[:3]} Val')
        current_metric = val_metrics['mAP']
        epoch_time = time.time() - epoch_start

        print(f"\n  Epoch {epoch+1}/{num_epochs} Summary ({epoch_time:.0f}s):")
        print(f"  Train - Loss: {train_metrics['loss']:.4f} | Cls: {train_metrics['cls_loss']:.4f} | Reg: {train_metrics['reg_loss']:.4f}")
        print(f"  Val   - Loss: {val_metrics['loss']:.4f} | mAP: {val_metrics['mAP']:.4f} | P: {val_metrics['precision']:.4f} | R: {val_metrics['recall']:.4f} | F1: {val_metrics['f1']:.4f}")
        print(f"  LR: {lr:.6f}")

        history.append({
            'epoch': epoch + 1, 'train_loss': train_metrics['loss'],
            'val_loss': val_metrics['loss'], 'val_mAP': val_metrics['mAP'],
            'val_precision': val_metrics['precision'], 'val_recall': val_metrics['recall'],
            'val_f1': val_metrics['f1'], 'lr': lr,
        })

        is_best = current_metric > best_metric
        if is_best:
            best_metric = current_metric
            early_stop_counter = 0
            save_checkpoint(model, optimizer, epoch, best_metric, os.path.join(config.VERSION_DIR, "best_model.pth"))
            print(f"  * New best model saved! mAP: {best_metric:.4f}")
        else:
            early_stop_counter += 1

        save_checkpoint(model, optimizer, epoch, best_metric, os.path.join(config.VERSION_DIR, "last.pth"))

        if early_stop_counter >= EARLY_STOP_PATIENCE:
            print(f"\n  Early stopping! No improvement for {EARLY_STOP_PATIENCE} epochs")
            break
        print("-" * 60)

    # 最终测试
    print("\n" + "=" * 60)
    print("  Final test...")
    print("=" * 60)
    load_checkpoint(model, optimizer, os.path.join(config.VERSION_DIR, "best_model.pth"))
    test_metrics = validate(model, test_loader, device, num_epochs, num_epochs,
                            iou_threshold=val_iou, use_dynamic=val_dynamic, desc='Test')
    print(f"\n  Test: Loss={test_metrics['loss']:.4f} | mAP={test_metrics['mAP']:.4f} | P={test_metrics['precision']:.4f} | R={test_metrics['recall']:.4f} | F1={test_metrics['f1']:.4f}")

    save_history(history, config.VERSION_DIR)
    print(f"\n  Model saved: {config.VERSION_DIR}/")


if __name__ == "__main__":
    main()
