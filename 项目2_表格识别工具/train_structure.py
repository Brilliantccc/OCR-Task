"""
结构识别训练脚本 (TrackB)
基于 Faster R-CNN 的表格 cell/row/column 检测
用法: python train_structure.py [--resume] [--finetune PATH] [--epochs N]
"""
import os
import math
import time
import argparse
import numpy as np
import torch
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from config import (
    DEVICE, STRUCTURE_EPOCHS, STRUCTURE_LR,
    LR_WARMUP_EPOCHS, LR_MIN, EARLY_STOP_PATIENCE,
    STRUCTURE_NUM_CLASSES, STRUCTURE_NMS_THRESH,
    init_dirs, init_version
)
import config
from dataset import get_structure_loaders
from model import TableDetector
from utils import (
    AverageMeter, save_checkpoint, load_checkpoint, save_history, compute_iou
)


def get_lr(epoch, base_lr, warmup_epochs, total_epochs, min_lr):
    """预热 + 余弦退火学习率"""
    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))


# ==================== 动态阈值 ====================

def get_dynamic_iou(area):
    """根据 cell 面积动态调整 IoU 阈值"""
    if area < 2000:
        return 0.15
    elif area < 8000:
        return 0.2
    elif area < 20000:
        return 0.3
    elif area < 50000:
        return 0.4
    else:
        return 0.5


def get_dynamic_score_threshold(area):
    """根据 cell 面积动态调整置信度阈值"""
    if area < 2000:
        return 0.15
    elif area < 8000:
        return 0.2
    elif area < 20000:
        return 0.3
    elif area < 50000:
        return 0.4
    else:
        return 0.5


def compute_fast_map(predictions, ground_truths, iou_threshold=0.5, use_dynamic=True):
    """快速计算 mAP（支持动态 IoU 和动态置信度）"""
    all_classes = set()
    for gt in ground_truths:
        if len(gt['labels']) > 0:
            all_classes.update(gt['labels'].cpu().tolist())
    all_classes.discard(0)

    aps = []
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for cls in all_classes:
        cls_pred_by_img = {}
        cls_gt_by_img = {}

        for img_idx, (pred, gt) in enumerate(zip(predictions, ground_truths)):
            mask = pred['labels'].cpu() == cls
            if mask.any():
                cls_pred_by_img[img_idx] = {
                    'boxes': pred['boxes'].cpu()[mask],
                    'scores': pred['scores'].cpu()[mask],
                }
            gt_mask = gt['labels'].cpu() == cls
            if gt_mask.any():
                cls_gt_by_img[img_idx] = {
                    'boxes': gt['boxes'].cpu()[gt_mask],
                }

        if not cls_gt_by_img:
            continue

        # 预计算每张图的 IoU 矩阵（一次性批量计算）
        iou_cache = {}
        for img_idx in cls_pred_by_img:
            if img_idx in cls_gt_by_img:
                pb = cls_pred_by_img[img_idx]['boxes']
                gb = cls_gt_by_img[img_idx]['boxes']
                if len(pb) > 0 and len(gb) > 0:
                    iou_cache[img_idx] = compute_iou(pb, gb)  # [P, G]

        # 收集所有预测并按 score 排序
        all_cls_preds = []
        for img_idx, p in cls_pred_by_img.items():
            scores = p['scores']
            boxes = p['boxes']
            for j in range(len(boxes)):
                if use_dynamic:
                    area = (boxes[j][2] - boxes[j][0]) * (boxes[j][3] - boxes[j][1])
                    score_thresh = get_dynamic_score_threshold(area.item())
                else:
                    score_thresh = 0.0
                if scores[j].item() >= score_thresh:
                    all_cls_preds.append((img_idx, j, scores[j].item()))

        if not all_cls_preds:
            total_fn += sum(len(g['boxes']) for g in cls_gt_by_img.values())
            continue

        all_cls_preds.sort(key=lambda x: x[2], reverse=True)

        # 匹配（使用预计算 IoU + bool tensor 替代 Python set）
        tp_list = []
        fp_list = []
        # 用 bool tensor 代替 set，向量化查找
        matched_mask = {idx: torch.zeros(len(cls_gt_by_img[idx]['boxes']), dtype=torch.bool)
                        for idx in cls_gt_by_img}

        for img_idx, pred_idx, score in all_cls_preds:
            if img_idx not in cls_gt_by_img or img_idx not in iou_cache:
                fp_list.append(1)
                tp_list.append(0)
                continue

            ious_row = iou_cache[img_idx][pred_idx]
            gt_boxes = cls_gt_by_img[img_idx]['boxes']
            mask = matched_mask[img_idx]

            # 向量化：已匹配的 GT 设为 -1，找最大 IoU
            masked_ious = ious_row.clone()
            masked_ious[mask] = -1.0
            best_iou, best_gt_idx = masked_ious.max(0)
            best_iou = best_iou.item()
            best_gt_idx = best_gt_idx.item()

            if use_dynamic and best_gt_idx >= 0 and best_iou > 0:
                gt_box = gt_boxes[best_gt_idx]
                gt_area = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
                dynamic_iou = get_dynamic_iou(gt_area.item())
            else:
                dynamic_iou = iou_threshold

            if best_iou >= dynamic_iou and best_gt_idx >= 0:
                tp_list.append(1)
                fp_list.append(0)
                mask[best_gt_idx] = True
            else:
                tp_list.append(0)
                fp_list.append(1)

        tp_cum = np.cumsum(tp_list)
        fp_cum = np.cumsum(fp_list)
        precision_arr = tp_cum / (tp_cum + fp_cum + 1e-6)
        recall_arr = tp_cum / (sum(len(g['boxes']) for g in cls_gt_by_img.values()) + 1e-6)

        # 全点插值 AP（比11点更准确）
        for i in range(len(precision_arr) - 2, -1, -1):
            precision_arr[i] = max(precision_arr[i], precision_arr[i + 1])
        r = np.concatenate(([0.0], recall_arr, [1.0]))
        p = np.concatenate(([1.0], precision_arr, [0.0]))
        ap = np.sum((r[1:] - r[:-1]) * p[1:])
        aps.append(ap)

        total_tp += tp_cum[-1]
        total_fp += fp_cum[-1]
        total_fn += sum(len(g['boxes']) for g in cls_gt_by_img.values()) - tp_cum[-1]

    mAP = np.mean(aps) if aps else 0.0
    precision = total_tp / (total_tp + total_fp + 1e-6)
    recall = total_tp / (total_tp + total_fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    return {
        'mAP': mAP,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }


# ==================== 训练 & 验证 ====================

def train_one_epoch(model, train_loader, optimizer, device, epoch, total_epochs, scaler=None):
    """训练一个 epoch"""
    model.train()
    loss_meter = AverageMeter()
    cls_loss_meter = AverageMeter()
    reg_loss_meter = AverageMeter()

    pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{total_epochs} [Str Train]',
                ncols=120, bar_format='{l_bar}{bar:20}{r_bar}')

    for batch_idx, (images, targets) in enumerate(pbar):
        images = [img.to(device, non_blocking=True) for img in images]
        targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]

        with autocast():
            loss_dict = model(images, targets)
            total_loss = sum(v for v in loss_dict.values() if isinstance(v, torch.Tensor))

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        loss_meter.update(total_loss.item(), len(images))
        cls_loss = loss_dict.get('loss_classifier', torch.tensor(0.0))
        reg_loss = loss_dict.get('loss_box_reg', torch.tensor(0.0))
        cls_loss_meter.update(cls_loss.item() if isinstance(cls_loss, torch.Tensor) else cls_loss, len(images))
        reg_loss_meter.update(reg_loss.item() if isinstance(reg_loss, torch.Tensor) else reg_loss, len(images))

        pbar.set_postfix({
            'loss': f'{loss_meter.avg:.4f}',
            'cls': f'{cls_loss_meter.avg:.4f}',
            'reg': f'{reg_loss_meter.avg:.4f}'
        })

    return {
        'loss': loss_meter.avg,
        'cls_loss': cls_loss_meter.avg,
        'reg_loss': reg_loss_meter.avg,
    }


def validate(model, val_loader, device, epoch, total_epochs):
    """验证模型（单次遍历：loss + metrics）"""
    loss_meter = AverageMeter()
    all_preds = []
    all_gts = []

    pbar = tqdm(val_loader, desc=f'Epoch {epoch}/{total_epochs} [Str Val]',
                ncols=120, bar_format='{l_bar}{bar:20}{r_bar}')

    was_training = model.training
    with torch.no_grad():
        for images, targets in pbar:
            images = [img.to(device, non_blocking=True) for img in images]
            targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]

            # loss（需 train 模式）
            model.train()
            with autocast():
                loss_dict = model(images, targets)
                total_loss = sum(v for v in loss_dict.values() if isinstance(v, torch.Tensor))
            loss_meter.update(total_loss.item(), len(images))

            # 推理（eval 模式）
            model.eval()
            with autocast():
                outputs = model(images)
            all_preds.extend([{k: v.cpu() for k, v in o.items()} for o in outputs])
            all_gts.extend([{k: v.cpu() for k, v in t.items()} for t in targets])

            pbar.set_postfix({'loss': f'{loss_meter.avg:.4f}', 'preds': len(all_preds)})

    # 计算指标
    tqdm.write('  Computing metrics...')
    metrics = compute_fast_map(all_preds, all_gts, iou_threshold=0.3, use_dynamic=True)

    model.train() if was_training else model.eval()

    return {
        'loss': loss_meter.avg,
        **metrics,
    }


# ==================== Main ====================

def main():
    parser = argparse.ArgumentParser(description='Structure Recognition Training (TrackB)')
    parser.add_argument('--resume', action='store_true', help='断点续训（默认续训最新版本）')
    parser.add_argument('--version', type=int, default=None, help='续训指定版本号，如 --version 1 续训 v1')
    parser.add_argument('--finetune', type=str, default=None, help='微调模型路径')
    parser.add_argument('--backbone', type=str, default=None, help='从检测模型迁移 backbone')
    parser.add_argument('--epochs', type=int, default=None, help='训练轮数')
    args = parser.parse_args()

    task = 'structure'

    if args.resume and args.version:
        # 续训指定版本：直接用该版本目录
        config.VERSION = args.version
        config.VERSION_DIR = os.path.join(config.MODEL_DIR, task, f"v{args.version}")
    elif args.resume:
        # 续训最新版本：扫描目录找最大版本号
        task_dir = os.path.join(config.MODEL_DIR, task)
        if os.path.exists(task_dir):
            versions = sorted([int(d[1:]) for d in os.listdir(task_dir) if d.startswith('v') and d[1:].isdigit()])
            if versions:
                config.VERSION = versions[-1]
                config.VERSION_DIR = os.path.join(task_dir, f"v{config.VERSION}")
            else:
                print(f"  [ERROR] No versions found in {task_dir}, nothing to resume")
                print(f"  Use --version N to specify a version, or remove --resume to start new training")
                return
        else:
            print(f"  [ERROR] {task_dir} does not exist, nothing to resume")
            return
    else:
        # 新训练：创建新版本（自动 max+1）
        init_version(task=task)
    init_dirs()

    num_epochs = args.epochs or STRUCTURE_EPOCHS
    base_lr = STRUCTURE_LR

    print("\n" + "=" * 60)
    print("  Structure Recognition (Faster R-CNN)")
    print(f"  Classes: {STRUCTURE_NUM_CLASSES} (background + table + cell)")
    print(f"  Version: {config.VERSION}")
    print("=" * 60)

    device = torch.device(DEVICE)
    print(f"\n  Device: {device}")
    if device.type == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        print(f"  GPU: {gpu_name} ({gpu_mem:.1f} GB)")

    # 加载数据
    print("\n  Loading data...")
    train_loader, val_loader, test_loader = get_structure_loaders()
    print(f"  Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)} | Test: {len(test_loader.dataset)}")

    # 创建模型（结构识别用 Faster R-CNN, 3类, 低内部 NMS 阈值）
    model = TableDetector(num_classes=STRUCTURE_NUM_CLASSES, pretrained=True,
                          detections_per_img=2000,
                          nms_thresh=STRUCTURE_NMS_THRESH).to(device)
    if args.backbone:
        print(f"\n  Loading backbone from: {args.backbone}")
        model.load_backbone_from(args.backbone)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model: {total_params:,} params ({trainable_params:,} trainable)")

    # 优化器 + 混合精度
    optimizer = AdamW(model.parameters(), lr=base_lr, weight_decay=1e-4)
    scaler = GradScaler() if device.type == 'cuda' else None
    start_epoch = 0
    best_metric = 0
    mode = "Normal"
    print(f"  AMP: {'Enabled' if scaler else 'Disabled'}")

    if args.finetune:
        mode = "Finetune"
        print(f"\n  Finetune mode: loading {args.finetune}")
        load_checkpoint(model, None, args.finetune)
        optimizer = AdamW(model.parameters(), lr=base_lr * 0.1, weight_decay=1e-4)
    elif args.resume:
        mode = "Resume"
        ckpt_path = os.path.join(config.VERSION_DIR, "best_model.pth")
        print(f"\n  Resuming version: v{config.VERSION}")
        print(f"  Checkpoint: {ckpt_path}")
        if os.path.exists(ckpt_path):
            start_epoch, best_metric = load_checkpoint(model, optimizer, ckpt_path)
            start_epoch += 1
            print(f"  From epoch {start_epoch}, best: {best_metric:.4f}")
        else:
            print(f"  [WARN] Checkpoint not found, starting from scratch")

    print(f"\n  Mode: {mode}")
    print(f"  Epochs: {num_epochs} | LR: {base_lr}")

    # 训练循环
    print("\n" + "=" * 60)
    print("  Starting training...")
    print("=" * 60)

    early_stop_counter = 0
    history = []

    for epoch in range(start_epoch, num_epochs):
        epoch_start = time.time()
        lr = get_lr(epoch, base_lr, LR_WARMUP_EPOCHS, num_epochs, LR_MIN)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        train_metrics = train_one_epoch(
            model, train_loader, optimizer, device, epoch + 1, num_epochs, scaler=scaler
        )
        val_metrics = validate(model, val_loader, device, epoch + 1, num_epochs)
        current_metric = val_metrics['mAP']
        epoch_time = time.time() - epoch_start

        # 打印 epoch 总结
        print(f"\n  Epoch {epoch+1}/{num_epochs} Summary ({epoch_time:.0f}s):")
        print(f"  Train - Loss: {train_metrics['loss']:.4f} | Cls: {train_metrics['cls_loss']:.4f} | Reg: {train_metrics['reg_loss']:.4f}")
        print(f"  Val   - Loss: {val_metrics['loss']:.4f} | mAP: {val_metrics['mAP']:.4f} | P: {val_metrics['precision']:.4f} | R: {val_metrics['recall']:.4f} | F1: {val_metrics['f1']:.4f}")
        print(f"  LR: {lr:.6f}")

        history.append({
            'epoch': epoch + 1,
            'train_loss': train_metrics['loss'],
            'val_loss': val_metrics['loss'],
            'val_mAP': val_metrics['mAP'],
            'val_precision': val_metrics['precision'],
            'val_recall': val_metrics['recall'],
            'val_f1': val_metrics['f1'],
            'lr': lr,
        })

        # 保存最佳模型
        is_best = False
        if current_metric > best_metric:
            best_metric = current_metric
            is_best = True

        if is_best:
            early_stop_counter = 0
            save_checkpoint(model, optimizer, epoch, best_metric,
                            os.path.join(config.VERSION_DIR, "best_model.pth"))
            print(f"  * New best model saved! mAP: {best_metric:.4f}")
        else:
            early_stop_counter += 1

        save_checkpoint(model, optimizer, epoch, best_metric,
                        os.path.join(config.VERSION_DIR, "last.pth"))

        if early_stop_counter >= EARLY_STOP_PATIENCE:
            print(f"\n  [WARN] Early stopping! No improvement for {EARLY_STOP_PATIENCE} epochs")
            break

        print("-" * 60)

    # 最终测试
    print("\n" + "=" * 60)
    print("  Training Complete! Running final test...")
    print("=" * 60)

    load_checkpoint(model, optimizer, os.path.join(config.VERSION_DIR, "best_model.pth"))
    test_metrics = validate(model, test_loader, device, num_epochs, num_epochs)
    print(f"\n  Final Test Results:")
    print(f"  Loss: {test_metrics['loss']:.4f} | mAP: {test_metrics['mAP']:.4f} | P: {test_metrics['precision']:.4f} | R: {test_metrics['recall']:.4f} | F1: {test_metrics['f1']:.4f}")

    print("\n" + "=" * 60)

    save_history(history, config.VERSION_DIR)
    print(f"\n  Model saved: {config.VERSION_DIR}/")


if __name__ == "__main__":
    main()
