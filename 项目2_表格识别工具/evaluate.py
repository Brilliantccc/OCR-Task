"""
评估脚本
计算 mAP、Precision、Recall、F1
可视化检测结果
"""
import os
import argparse
import torch
import torchvision.ops as ops
import numpy as np
import torchvision.transforms as T
from tqdm import tqdm

from config import (
    DEVICE, DETECTION_CLASSES, STRUCTURE_CLASSES,
    NMS_IOU_THRESHOLD, DETECTION_SCORE_THRESHOLD, MAX_DETECTIONS, MODEL_DIR,
    NMS_METHOD, SOFT_NMS_SIGMA, SOFT_NMS_SCORE_THRESH, STRUCTURE_NMS_THRESH
)
from model import TableDetector
from dataset import get_detection_loaders, get_structure_loaders
from utils import load_checkpoint, compute_iou, draw_boxes, soft_nms


def evaluate_detection(model, test_loader, device, score_threshold=0.5):
    """
    评估表格检测模型（COCO 风格 + 传统指标）
    返回: mAP, AP@50, AP@75, AP@50:95, precision, recall, f1
    """
    model.eval()
    all_predictions = []
    all_ground_truths = []

    # 收集所有预测和标注
    for images, targets in tqdm(test_loader, desc="Evaluating"):
        images = [img.to(device) for img in images]

        with torch.no_grad():
            outputs = model(images)

        for i, output in enumerate(outputs):
            pred_boxes = output['boxes'].cpu()
            pred_scores = output['scores'].cpu()
            pred_labels = output['labels'].cpu()

            # 过滤低置信度
            keep = pred_scores > score_threshold
            pred_boxes = pred_boxes[keep]
            pred_scores = pred_scores[keep]
            pred_labels = pred_labels[keep]

            # NMS（支持 Soft-NMS 保留密集目标）
            if len(pred_boxes) > 0:
                if NMS_METHOD == 'soft':
                    nms_keep = soft_nms(pred_boxes, pred_scores,
                                        sigma=SOFT_NMS_SIGMA,
                                        score_threshold=SOFT_NMS_SCORE_THRESH)
                else:
                    nms_keep = ops.nms(pred_boxes, pred_scores, NMS_IOU_THRESHOLD)
                pred_boxes = pred_boxes[nms_keep]
                pred_scores = pred_scores[nms_keep]
                pred_labels = pred_labels[nms_keep]

            # 限制最大检测数
            if len(pred_boxes) > MAX_DETECTIONS:
                top_k = pred_scores.topk(MAX_DETECTIONS)
                pred_boxes = pred_boxes[top_k.indices]
                pred_scores = pred_scores[top_k.indices]
                pred_labels = pred_labels[top_k.indices]

            # 记录预测
            image_id = i
            for j in range(len(pred_boxes)):
                all_predictions.append({
                    'image_id': image_id,
                    'bbox': pred_boxes[j].tolist(),
                    'score': pred_scores[j].item(),
                    'label': pred_labels[j].item(),
                })

            # 记录标注
            target = targets[i]
            gt_boxes = target['boxes']
            gt_labels = target['labels']
            for j in range(len(gt_boxes)):
                all_ground_truths.append({
                    'image_id': image_id,
                    'bbox': gt_boxes[j].tolist(),
                    'label': gt_labels[j].item(),
                })

    # ---- COCO 风格 mAP（多 IoU 阈值）----
    iou_thresholds = np.arange(0.5, 1.0, 0.05)  # [0.50, 0.55, ..., 0.95]
    aps_all_iou = [[] for _ in iou_thresholds]

    id_to_name = {v: k for k, v in DETECTION_CLASSES.items()}

    for cls_id, cls_name in id_to_name.items():
        if cls_id == 0:
            continue

        cls_preds = [p for p in all_predictions if p['label'] == cls_id]
        cls_gts = [g for g in all_ground_truths if g['label'] == cls_id]

        if not cls_gts:
            continue

        cls_preds = sorted(cls_preds, key=lambda x: x['score'], reverse=True)

        for iou_idx, iou_thresh in enumerate(iou_thresholds):
            tp = np.zeros(len(cls_preds))
            fp = np.zeros(len(cls_preds))
            matched_gt = set()

            for pred_idx, pred in enumerate(cls_preds):
                best_iou = 0
                best_gt_idx = -1

                for gt_idx, gt in enumerate(cls_gts):
                    if gt['image_id'] != pred['image_id']:
                        continue
                    gt_key = (gt['image_id'], gt_idx)
                    if gt_key in matched_gt:
                        continue

                    iou = compute_iou(
                        torch.tensor(pred['bbox']).unsqueeze(0),
                        torch.tensor(gt['bbox']).unsqueeze(0)
                    ).item()

                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx

                if best_iou >= iou_thresh and best_gt_idx >= 0:
                    tp[pred_idx] = 1
                    matched_gt.add((pred['image_id'], best_gt_idx))
                else:
                    fp[pred_idx] = 1

            # 全点插值 AP（比11点更准确）
            tp_cumsum = np.cumsum(tp)
            fp_cumsum = np.cumsum(fp)
            prec = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
            rec = tp_cumsum / (len(cls_gts) + 1e-6)
            # 使 precision 单调递减
            for i in range(len(prec) - 2, -1, -1):
                prec[i] = max(prec[i], prec[i + 1])
            r = np.concatenate(([0.0], rec, [1.0]))
            p = np.concatenate(([1.0], prec, [0.0]))
            ap = np.sum((r[1:] - r[:-1]) * p[1:])

            aps_all_iou[iou_idx].append(ap)

    # 汇总 COCO 指标
    ap50 = np.mean(aps_all_iou[0]) if aps_all_iou[0] else 0       # index 0 = IoU 0.50
    ap75 = np.mean(aps_all_iou[5]) if aps_all_iou[5] else 0       # index 5 = IoU 0.75
    aps_values = [np.mean(v) for v in aps_all_iou if v]
    map_50_95 = np.mean(aps_values) if aps_values else 0

    # ---- 传统 Precision / Recall / F1（IoU=0.5）----
    total_tp = 0
    total_fp = 0
    total_fn = 0

    image_ids = set(p['image_id'] for p in all_predictions) | set(g['image_id'] for g in all_ground_truths)
    for img_id in image_ids:
        preds = [p for p in all_predictions if p['image_id'] == img_id and p['label'] > 0]
        gts = [g for g in all_ground_truths if g['image_id'] == img_id]

        matched = set()
        for pred in sorted(preds, key=lambda x: x['score'], reverse=True):
            best_iou = 0
            best_gt = -1
            for gt_idx, gt in enumerate(gts):
                if gt_idx in matched:
                    continue
                iou = compute_iou(
                    torch.tensor(pred['bbox']).unsqueeze(0),
                    torch.tensor(gt['bbox']).unsqueeze(0)
                ).item()
                if iou > best_iou:
                    best_iou = iou
                    best_gt = gt_idx

            if best_iou >= 0.5:
                total_tp += 1
                matched.add(best_gt)
            else:
                total_fp += 1

        total_fn += len(gts) - len(matched)

    precision = total_tp / (total_tp + total_fp + 1e-6)
    recall = total_tp / (total_tp + total_fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    # ---- 统计信息 ----
    total_gt = len(all_ground_truths)
    total_pred = len(all_predictions)
    images_with_gt = len(set(g['image_id'] for g in all_ground_truths))
    images_with_pred = len(set(p['image_id'] for p in all_predictions))

    return {
        # COCO 风格
        'mAP': ap50,             # mAP@0.5（和之前兼容）
        'AP50': ap50,            # AP@IoU=0.50
        'AP75': ap75,            # AP@IoU=0.75
        'map_50_95': map_50_95,  # mAP@IoU=0.50:0.95
        # 传统指标
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'total_tp': total_tp,
        'total_fp': total_fp,
        'total_fn': total_fn,
        # 统计
        'total_gt': total_gt,
        'total_pred': total_pred,
        'images_with_gt': images_with_gt,
        'images_with_pred': images_with_pred,
    }


def visualize_predictions(model, test_loader, device, save_dir, num_samples=10, class_dict=None):
    """可视化检测结果"""
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    # id -> name 映射（根据任务使用对应类别字典）
    id_to_name = {v: k for k, v in (class_dict or DETECTION_CLASSES).items()}

    count = 0
    for images, targets in test_loader:
        for i in range(len(images)):
            if count >= num_samples:
                return

            image_tensor = images[i]
            target = targets[i]

            # 逆归一化
            img = image_tensor.clone()
            img = img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + \
                  torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            img = T.ToPILImage()(img.clamp(0, 1))

            # 预测
            with torch.no_grad():
                output = model([image_tensor.to(device)])[0]

            pred_boxes = output['boxes'].cpu()
            pred_scores = output['scores'].cpu()
            pred_labels = output['labels'].cpu()

            # GT
            gt_boxes = target['boxes']
            gt_labels = target['labels']

            # 绘制
            save_path = os.path.join(save_dir, f"sample_{count:03d}.jpg")
            draw_boxes(img, gt_boxes.numpy(), labels=[f"GT:{id_to_name.get(l.item(), l.item())}" for l in gt_labels],
                      color='green', width=2)
            draw_boxes(img, pred_boxes.numpy(), labels=[f"P:{id_to_name.get(l.item(), l.item())}:{s:.2f}" for l, s in zip(pred_labels, pred_scores)],
                      scores=pred_scores.numpy(), color='red', width=2)
            img.save(save_path)

            count += 1
            print(f"  Saved: {save_path}")

        if count >= num_samples:
            break


def main():
    parser = argparse.ArgumentParser(description='Table Detection Evaluation')
    parser.add_argument('--task', type=str, default='detection',
                        choices=['detection', 'structure'])
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='模型路径，默认使用 best_model.pth')
    parser.add_argument('--visualize', action='store_true', help='可视化预测结果')
    parser.add_argument('--num_vis', type=int, default=10, help='可视化样本数')
    args = parser.parse_args()

    device = torch.device(DEVICE)
    print(f"\n  Device: {device}")

    # 加载数据
    print("\n  Loading test data...")
    _, _, test_loader = get_detection_loaders() if args.task == 'detection' else get_structure_loaders()
    print(f"  Test: {len(test_loader.dataset)} samples")

    # 加载模型（detection 和 structure 都用 Faster R-CNN）
    print("\n  Loading model...")
    if args.task == 'detection':
        model = TableDetector(num_classes=2, pretrained=False).to(device)
    else:
        model = TableDetector(num_classes=3, pretrained=False, detections_per_img=2000,
                              nms_thresh=STRUCTURE_NMS_THRESH).to(device)

    # 默认路径: 自动查找最新版本
    task_dir = os.path.join(MODEL_DIR, args.task)
    if os.path.exists(task_dir):
        versions = sorted([d for d in os.listdir(task_dir) if d.startswith('v')],
                          key=lambda x: int(x[1:]), reverse=True)
        latest = versions[0] if versions else 'v1'
    else:
        latest = 'v1'
    default_ckpt = os.path.join(task_dir, latest, "best_model.pth")
    ckpt_path = args.checkpoint or default_ckpt

    if os.path.exists(ckpt_path):
        load_checkpoint(model, None, ckpt_path)
        print(f"  Loaded: {ckpt_path}")
    else:
        print(f"  [WARN] Checkpoint not found: {ckpt_path}")
        return

    # 评估
    print("\n" + "=" * 60)
    print("  Evaluating...")
    print("=" * 60)

    # 结构识别使用更低的 score 阈值（cell 更小、更密集）
    eval_score_threshold = 0.5 if args.task == 'detection' else 0.3
    results = evaluate_detection(model, test_loader, device, score_threshold=eval_score_threshold)

    print(f"\n  Results ({args.task}):")
    print(f"  {'='*50}")
    print(f"  COCO Metrics:")
    print(f"    AP@0.50:         {results['AP50']:.4f}")
    print(f"    AP@0.75:         {results['AP75']:.4f}")
    print(f"    AP@0.50:0.95:    {results['map_50_95']:.4f}")
    print(f"  {'='*50}")
    print(f"  Traditional Metrics (IoU=0.5):")
    print(f"    Precision:       {results['precision']:.4f}")
    print(f"    Recall:          {results['recall']:.4f}")
    print(f"    F1-Score:        {results['f1']:.4f}")
    print(f"  {'='*50}")
    print(f"  TP: {results['total_tp']} | FP: {results['total_fp']} | FN: {results['total_fn']}")
    print(f"  GT: {results['total_gt']} | Pred: {results['total_pred']}")
    print(f"  Images: {results['images_with_gt']} with GT, {results['images_with_pred']} with Pred")

    target_map = 0.85
    if results['AP50'] >= target_map:
        print(f"\n  [OK] Target AP@0.50 >= {target_map} achieved!")
    else:
        print(f"\n  [WARN] AP@0.50 {results['AP50']:.4f} < {target_map}, needs improvement")

    # 可视化
    if args.visualize:
        print(f"\n  Visualizing {args.num_vis} samples...")
        vis_dir = os.path.join(MODEL_DIR, args.task, "vis")
        class_dict = DETECTION_CLASSES if args.task == 'detection' else STRUCTURE_CLASSES
        visualize_predictions(model, test_loader, device, vis_dir, args.num_vis, class_dict=class_dict)


if __name__ == "__main__":
    main()
