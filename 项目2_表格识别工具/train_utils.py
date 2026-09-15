"""
训练公共函数
供 train_detection.py / train_structure.py / train_combined.py 共用
"""
import math
import numpy as np
import torch
from torch.cuda.amp import autocast
from tqdm import tqdm

from utils import AverageMeter, compute_iou


# ==================== 学习率调度 ====================

def get_lr(epoch, base_lr, warmup_epochs, total_epochs, min_lr):
    """预热 + 余弦退火学习率"""
    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))


# ==================== 动态阈值 ====================

def get_dynamic_iou(area):
    """根据目标面积动态调整 IoU 阈值"""
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
    """根据目标面积动态调整置信度阈值"""
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


# ==================== mAP 计算 ====================

def compute_fast_map(predictions, ground_truths, iou_threshold=0.5, use_dynamic=True):
    """
    快速计算 mAP（批量 IoU + numpy 匹配）
    use_dynamic=True: 结构识别用动态阈值
    use_dynamic=False: 检测用固定阈值
    """
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

        # numpy 匹配（bool tensor + argmax）
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
                mask[best_gt_idx] = True
            else:
                tp_list.append(0)
                fp_list.append(1)

        # AP（全点插值）
        tp_cum = np.cumsum(tp_list)
        fp_cum = np.cumsum(fp_list)
        precision_arr = tp_cum / (tp_cum + fp_cum + 1e-6)
        recall_arr = tp_cum / (sum(len(g['boxes']) for g in cls_gt_by_img.values()) + 1e-6)

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

def train_one_epoch(model, train_loader, optimizer, device, epoch, total_epochs,
                    scaler=None, desc='Train'):
    """训练一个 epoch"""
    model.train()
    loss_meter = AverageMeter()
    cls_loss_meter = AverageMeter()
    reg_loss_meter = AverageMeter()

    pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{total_epochs} [{desc}]',
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


def validate(model, val_loader, device, epoch, total_epochs,
             iou_threshold=0.5, use_dynamic=False, desc='Val'):
    """验证模型（单次遍历：loss + metrics）"""
    loss_meter = AverageMeter()
    all_preds = []
    all_gts = []

    pbar = tqdm(val_loader, desc=f'Epoch {epoch}/{total_epochs} [{desc}]',
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
    metrics = compute_fast_map(all_preds, all_gts, iou_threshold=iou_threshold, use_dynamic=use_dynamic)

    model.train() if was_training else model.eval()

    return {
        'loss': loss_meter.avg,
        **metrics,
    }
