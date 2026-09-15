"""
训练脚本（GPU加速版）
支持：表格检测训练 + 结构识别训练 + 断点续训 + 微调 + 混合精度训练
"""
import os
import math
import argparse
import numpy as np
import torch
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from config import (
    DEVICE, DETECTION_EPOCHS, DETECTION_LR, STRUCTURE_EPOCHS, STRUCTURE_LR,
    LR_WARMUP_EPOCHS, LR_MIN, EARLY_STOP_PATIENCE,
    DETECTION_BATCH_SIZE, STRUCTURE_BATCH_SIZE,
    init_dirs, init_version
)
import config
from dataset import get_detection_loaders, get_structure_loaders
from model import TableDetector
from utils import (
    AverageMeter, save_checkpoint, load_checkpoint, save_history, compute_iou
)


def get_lr(epoch, base_lr, warmup_epochs, total_epochs, min_lr):
    """手动计算预热+余弦退火学习率"""
    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))


def train_one_epoch_detection(model, train_loader, optimizer, device, epoch, total_epochs, scaler=None):
    """训练检测模型一个epoch（支持混合精度训练）"""
    model.train()
    loss_meter = AverageMeter()
    cls_loss_meter = AverageMeter()
    reg_loss_meter = AverageMeter()

    pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{total_epochs} [Det Train]')

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


def get_dynamic_iou(area):
    """
    根据 cell 面积动态调整 IoU 阈值
    基于数据分布：中位数 13950，10%分位 3072
    """
    if area < 2000:      # 极小 cell (1%)
        return 0.15      # 非常宽松
    elif area < 8000:    # 小 cell (10%)
        return 0.2       # 宽松
    elif area < 20000:   # 中 cell (50%)
        return 0.3       # 标准
    elif area < 50000:   # 大 cell (75%)
        return 0.4       # 较严格
    else:                # 极大 cell (95%)
        return 0.5       # 严格


def get_dynamic_score_threshold(area):
    """
    根据 cell 面积动态调整置信度阈值
    小物体置信度通常较低，需要更低阈值
    """
    if area < 2000:      # 极小 cell
        return 0.15      # 低阈值
    elif area < 8000:    # 小 cell
        return 0.2
    elif area < 20000:   # 中 cell
        return 0.3
    elif area < 50000:   # 大 cell
        return 0.4
    else:                # 极大 cell
        return 0.5       # 高阈值


def compute_fast_map(predictions, ground_truths, iou_threshold=0.5, use_dynamic=True):
    """
    快速计算 mAP（支持动态 IoU 和动态置信度）
    """
    # 按类别统计
    all_classes = set()
    for gt in ground_truths:
        if len(gt['labels']) > 0:
            all_classes.update(gt['labels'].cpu().tolist())
    all_classes.discard(0)  # 去掉背景

    aps = []
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for cls in all_classes:
        # 收集该类别的所有 pred 和 gt（按 image_id 分组）
        cls_pred_by_img = {}
        cls_gt_by_img = {}

        for img_idx, (pred, gt) in enumerate(zip(predictions, ground_truths)):
            # 该类别的预测
            mask = pred['labels'].cpu() == cls
            if mask.any():
                cls_pred_by_img[img_idx] = {
                    'boxes': pred['boxes'].cpu()[mask],
                    'scores': pred['scores'].cpu()[mask],
                }
            # 该类别的标注
            gt_mask = gt['labels'].cpu() == cls
            if gt_mask.any():
                cls_gt_by_img[img_idx] = {
                    'boxes': gt['boxes'].cpu()[gt_mask],
                }

        if not cls_gt_by_img:
            continue

        # 预计算每张图的 IoU 矩阵
        iou_cache = {}
        for img_idx in cls_pred_by_img:
            if img_idx in cls_gt_by_img:
                pb = cls_pred_by_img[img_idx]['boxes']
                gb = cls_gt_by_img[img_idx]['boxes']
                if len(pb) > 0 and len(gb) > 0:
                    iou_cache[img_idx] = compute_iou(pb, gb)

        # 收集所有预测并按 score 排序
        all_cls_preds = []
        for img_idx, p in cls_pred_by_img.items():
            for j in range(len(p['boxes'])):
                if use_dynamic:
                    box_area = (p['boxes'][j][2] - p['boxes'][j][0]) * (p['boxes'][j][3] - p['boxes'][j][1])
                    score_thresh = get_dynamic_score_threshold(box_area.item())
                else:
                    score_thresh = 0.0
                if p['scores'][j].item() >= score_thresh:
                    all_cls_preds.append((img_idx, j, p['scores'][j].item()))

        if not all_cls_preds:
            total_fn += sum(len(g['boxes']) for g in cls_gt_by_img.values())
            continue

        all_cls_preds.sort(key=lambda x: x[2], reverse=True)

        tp_list = []
        fp_list = []
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
                matched_per_img[img_idx].add(best_gt_idx)
            else:
                tp_list.append(0)
                fp_list.append(1)

        # AP
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

        # P/R/F1
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


def validate_detection(model, val_loader, device, epoch, total_epochs):
    """验证检测模型（单次遍历：loss + metrics）"""
    loss_meter = AverageMeter()
    cls_loss_meter = AverageMeter()
    reg_loss_meter = AverageMeter()
    all_preds = []
    all_gts = []

    pbar = tqdm(val_loader, desc=f'Epoch {epoch}/{total_epochs} [Det Val]')

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
            cls_loss = loss_dict.get('loss_classifier', torch.tensor(0.0))
            reg_loss = loss_dict.get('loss_box_reg', torch.tensor(0.0))
            cls_loss_meter.update(cls_loss.item() if isinstance(cls_loss, torch.Tensor) else cls_loss, len(images))
            reg_loss_meter.update(reg_loss.item() if isinstance(reg_loss, torch.Tensor) else reg_loss, len(images))

            # 推理（eval 模式）
            model.eval()
            with autocast():
                outputs = model(images)
            all_preds.extend([{k: v.cpu() for k, v in o.items()} for o in outputs])
            all_gts.extend([{k: v.cpu() for k, v in t.items()} for t in targets])

            pbar.set_postfix({'loss': f'{loss_meter.avg:.4f}'})

    metrics = compute_fast_map(all_preds, all_gts, iou_threshold=0.3, use_dynamic=True)

    model.train() if was_training else model.eval()

    return {
        'loss': loss_meter.avg,
        'cls_loss': cls_loss_meter.avg,
        'reg_loss': reg_loss_meter.avg,
        **metrics,
    }


def train_one_epoch_structure(model, train_loader, optimizer, device, epoch, total_epochs, scaler=None):
    """训练结构识别模型一个epoch（Faster R-CNN）"""
    model.train()
    loss_meter = AverageMeter()
    cls_loss_meter = AverageMeter()
    reg_loss_meter = AverageMeter()

    pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{total_epochs} [Str Train]')

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


def validate_structure(model, val_loader, device, epoch, total_epochs):
    """验证结构识别模型（Faster R-CNN，和检测一样）"""
    loss_meter = AverageMeter()
    cls_loss_meter = AverageMeter()
    reg_loss_meter = AverageMeter()
    all_preds = []
    all_gts = []

    pbar = tqdm(val_loader, desc=f'Epoch {epoch}/{total_epochs} [Str Val]')

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
            cls_loss = loss_dict.get('loss_classifier', torch.tensor(0.0))
            reg_loss = loss_dict.get('loss_box_reg', torch.tensor(0.0))
            cls_loss_meter.update(cls_loss.item() if isinstance(cls_loss, torch.Tensor) else cls_loss, len(images))
            reg_loss_meter.update(reg_loss.item() if isinstance(reg_loss, torch.Tensor) else reg_loss, len(images))

            # 推理（eval 模式）
            model.eval()
            with autocast():
                outputs = model(images)
            all_preds.extend([{k: v.cpu() for k, v in o.items()} for o in outputs])
            all_gts.extend([{k: v.cpu() for k, v in t.items()} for t in targets])

            pbar.set_postfix({'loss': f'{loss_meter.avg:.4f}'})

    metrics = compute_fast_map(all_preds, all_gts, iou_threshold=0.3, use_dynamic=True)

    model.train() if was_training else model.eval()

    return {
        'loss': loss_meter.avg,
        'cls_loss': cls_loss_meter.avg,
        'reg_loss': reg_loss_meter.avg,
        **metrics,
    }


def main():
    parser = argparse.ArgumentParser(description='Table Recognition Training')
    parser.add_argument('--task', type=str, default='detection',
                        choices=['detection', 'structure'],
                        help='训练任务: detection(表格检测) 或 structure(结构识别)')
    parser.add_argument('--resume', action='store_true', help='断点续训（默认续训最新版本）')
    parser.add_argument('--version', type=int, default=None, help='续训指定版本号，如 --version 1 续训 v1')
    parser.add_argument('--finetune', type=str, default=None, help='微调模型路径')
    parser.add_argument('--backbone', type=str, default=None,
                        help='结构识别: 从检测模型迁移 backbone 权重')
    parser.add_argument('--epochs', type=int, default=None, help='训练轮数')
    args = parser.parse_args()

    # 初始化版本（续训时复用已有版本，新训练时创建新版本）
    task = args.task
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
                print(f"  [ERROR] No versions found in {task_dir}, nothing to resume")
                print(f"  Use --version N to specify a version, or remove --resume to start new training")
                return
        else:
            print(f"  [ERROR] {task_dir} does not exist, nothing to resume")
            return
    else:
        init_version(task=task)
    init_dirs()

    # 任务配置
    is_detection = (args.task == 'detection')
    num_epochs = args.epochs or (DETECTION_EPOCHS if is_detection else STRUCTURE_EPOCHS)
    base_lr = DETECTION_LR if is_detection else STRUCTURE_LR

    task_name = "Table Detection (Faster R-CNN)" if is_detection else "Structure Recognition (Faster R-CNN)"

    print("\n" + "=" * 60)
    print(f"  {task_name}")
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
    if is_detection:
        train_loader, val_loader, test_loader = get_detection_loaders()
    else:
        train_loader, val_loader, test_loader = get_structure_loaders()
    print(f"  Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)} | Test: {len(test_loader.dataset)}")

    # 创建模型
    if is_detection:
        model = TableDetector(num_classes=2, pretrained=True).to(device)
    else:
        # 结构识别：用 Faster R-CNN，3类
        model = TableDetector(num_classes=3, pretrained=True, detections_per_img=2000).to(device)
        if args.backbone:
            print(f"\n  Loading backbone from: {args.backbone}")
            model.load_backbone_from(args.backbone)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model: {total_params:,} params ({trainable_params:,} trainable)")

    # 优化器 + 混合精度训练
    optimizer = AdamW(model.parameters(), lr=base_lr, weight_decay=1e-4)
    scaler = GradScaler() if device.type == 'cuda' else None
    start_epoch = 0
    best_metric = 0
    mode = "Normal"
    print(f"  AMP: {'Enabled' if scaler else 'Disabled'}")

    # 加载模型
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
        lr = get_lr(epoch, base_lr, LR_WARMUP_EPOCHS, num_epochs, LR_MIN)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # 训练
        if is_detection:
            train_metrics = train_one_epoch_detection(
                model, train_loader, optimizer, device, epoch + 1, num_epochs, scaler=scaler
            )
            val_metrics = validate_detection(model, val_loader, device, epoch + 1, num_epochs)
            current_metric = val_metrics['mAP']
            metric_name = 'mAP'
        else:
            train_metrics = train_one_epoch_structure(
                model, train_loader, optimizer, device, epoch + 1, num_epochs, scaler=scaler
            )
            val_metrics = validate_structure(model, val_loader, device, epoch + 1, num_epochs)
            current_metric = val_metrics['mAP']
            metric_name = 'mAP'

        # 打印 epoch 总结
        print(f"\n  Epoch {epoch+1}/{num_epochs} Summary:")
        print(f"  Train - Loss: {train_metrics['loss']:.4f} | Cls: {train_metrics.get('cls_loss', 0):.4f} | Reg: {train_metrics.get('reg_loss', 0):.4f}")
        print(f"  Val   - Loss: {val_metrics['loss']:.4f} | mAP: {val_metrics['mAP']:.4f} | P: {val_metrics['precision']:.4f} | R: {val_metrics['recall']:.4f} | F1: {val_metrics['f1']:.4f}")
        print(f"  LR: {lr:.6f}")

        # 记录历史
        history_record = {
            'epoch': epoch + 1,
            'train_loss': train_metrics['loss'],
            'val_loss': val_metrics['loss'],
            'lr': lr,
        }
        if is_detection:
            history_record.update({
                'val_mAP': val_metrics['mAP'],
                'val_precision': val_metrics['precision'],
                'val_recall': val_metrics['recall'],
                'val_f1': val_metrics['f1'],
            })
        else:
            history_record.update({
                'val_mAP': val_metrics['mAP'],
                'val_precision': val_metrics['precision'],
                'val_recall': val_metrics['recall'],
                'val_f1': val_metrics['f1'],
            })
        history.append(history_record)

        # 保存最佳模型（mAP 越高越好）
        is_best = False
        if current_metric > best_metric:
            best_metric = current_metric
            is_best = True

        if is_best:
            early_stop_counter = 0
            save_checkpoint(model, optimizer, epoch, best_metric,
                           os.path.join(config.VERSION_DIR, "best_model.pth"))
            print(f"  * New best model saved! {metric_name}: {best_metric:.4f}")
        else:
            early_stop_counter += 1

        # 保存最后一个 epoch
        save_checkpoint(model, optimizer, epoch, best_metric,
                       os.path.join(config.VERSION_DIR, "last.pth"))

        # 早停
        if early_stop_counter >= EARLY_STOP_PATIENCE:
            print(f"\n  [WARN] Early stopping! No improvement for {EARLY_STOP_PATIENCE} epochs")
            break

        print("-" * 60)

    # 最终测试
    print("\n" + "=" * 60)
    print("  Training Complete! Running final test...")
    print("=" * 60)

    load_checkpoint(model, optimizer, os.path.join(config.VERSION_DIR, "best_model.pth"))
    if is_detection:
        test_metrics = validate_detection(model, test_loader, device, num_epochs, num_epochs)
        print(f"\n  Final Test Results:")
        print(f"  Loss: {test_metrics['loss']:.4f} | mAP: {test_metrics['mAP']:.4f}")
        target_map = 0.85
        if test_metrics['mAP'] >= target_map:
            print(f"  [OK] Target mAP >= {target_map} achieved!")
        else:
            print(f"  [WARN] mAP {test_metrics['mAP']:.4f} < {target_map}, needs improvement")
    else:
        test_metrics = validate_structure(model, test_loader, device, num_epochs, num_epochs)
        print(f"\n  Final Test Results:")
        print(f"  Loss: {test_metrics['loss']:.4f} | mAP: {test_metrics['mAP']:.4f} | P: {test_metrics['precision']:.4f} | R: {test_metrics['recall']:.4f} | F1: {test_metrics['f1']:.4f}")

    print("\n" + "=" * 60)

    # 保存训练历史
    save_history(history, config.VERSION_DIR)
    print(f"\n  Model saved: {config.VERSION_DIR}/")


if __name__ == "__main__":
    main()
