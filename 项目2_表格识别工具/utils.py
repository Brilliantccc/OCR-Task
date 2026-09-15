"""
工具函数：VOC XML解析、NMS、可视化、COCO格式转换
"""
import os
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw, ImageFont
import torch
import torchvision.ops as ops
import numpy as np
import json
import csv


# ==================== XML 解析 ====================

def parse_voc_xml(xml_path):
    """
    解析 cTDaR 格式 XML 标注文件
    返回: {'filename': str, 'width': int, 'height': int,
           'objects': [{'name': str, 'bbox': [x1,y1,x2,y2], ...}]}
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # 获取文件名
    filename = root.get('filename')
    if not filename:
        filename = os.path.splitext(os.path.basename(xml_path))[0] + '.jpg'

    # 获取图片尺寸（需要从图片文件读取）
    # cTDaR XML 不包含尺寸信息，先用默认值
    width = 0
    height = 0

    # 解析标注物体
    objects = []

    # 尝试解析 <table> 标签（TrackA 格式）
    for table in root.findall('.//table'):
        coords = table.find('Coords')
        if coords is not None:
            points_str = coords.get('points', '')
            # 解析 "x1,y1 x2,y2 x3,y3 x4,y4" 格式
            points = []
            for point in points_str.split():
                x, y = point.split(',')
                points.append((float(x), float(y)))

            if len(points) >= 4:
                # 转换为 [x1, y1, x2, y2] 格式（左上角和右下角）
                xs = [p[0] for p in points]
                ys = [p[1] for p in points]
                bbox = [min(xs), min(ys), max(xs), max(ys)]
                objects.append({
                    'name': 'table',
                    'bbox': bbox,
                    'difficult': 0,
                    'truncated': 0,
                })

    # 尝试解析 <cell> 标签（TrackB 格式）
    for cell in root.findall('.//cell'):
        coords = cell.find('Coords')
        if coords is not None:
            points_str = coords.get('points', '')
            points = []
            for point in points_str.split():
                x, y = point.split(',')
                points.append((float(x), float(y)))

            if len(points) >= 4:
                xs = [p[0] for p in points]
                ys = [p[1] for p in points]
                bbox = [min(xs), min(ys), max(xs), max(ys)]
                objects.append({
                    'name': 'cell',
                    'bbox': bbox,
                    'difficult': 0,
                    'truncated': 0,
                })

    # 尝试解析 <row> 和 <column> 标签
    for row in root.findall('.//row'):
        coords = row.find('Coords')
        if coords is not None:
            points_str = coords.get('points', '')
            points = []
            for point in points_str.split():
                x, y = point.split(',')
                points.append((float(x), float(y)))
            if len(points) >= 4:
                xs = [p[0] for p in points]
                ys = [p[1] for p in points]
                bbox = [min(xs), min(ys), max(xs), max(ys)]
                objects.append({
                    'name': 'row',
                    'bbox': bbox,
                    'difficult': 0,
                    'truncated': 0,
                })

    for col in root.findall('.//column'):
        coords = col.find('Coords')
        if coords is not None:
            points_str = coords.get('points', '')
            points = []
            for point in points_str.split():
                x, y = point.split(',')
                points.append((float(x), float(y)))
            if len(points) >= 4:
                xs = [p[0] for p in points]
                ys = [p[1] for p in points]
                bbox = [min(xs), min(ys), max(xs), max(ys)]
                objects.append({
                    'name': 'column',
                    'bbox': bbox,
                    'difficult': 0,
                    'truncated': 0,
                })

    return {
        'filename': filename,
        'width': width,
        'height': height,
        'objects': objects
    }


def voc_to_coco(xml_dir, img_dir, class_to_id):
    """
    将 VOC XML 目录转为 COCO 格式字典
    用于 DETR 等模型训练
    """
    coco = {
        'images': [],
        'annotations': [],
        'categories': [{'id': k, 'name': v} for k, v in class_to_id.items()]
    }

    ann_id = 0
    xml_files = [f for f in os.listdir(xml_dir) if f.endswith('.xml')]

    for img_id, xml_file in enumerate(sorted(xml_files)):
        xml_path = os.path.join(xml_dir, xml_file)
        ann = parse_voc_xml(xml_path)

        # 添加图片信息
        img_path = os.path.join(img_dir, ann['filename'])
        coco['images'].append({
            'id': img_id,
            'file_name': ann['filename'],
            'width': ann['width'],
            'height': ann['height'],
        })

        # 添加标注
        for obj in ann['objects']:
            if obj['name'] not in class_to_id:
                continue

            bbox = obj['bbox']  # [x1, y1, x2, y2]
            # 转为 COCO 格式 [x, y, w, h]
            coco_bbox = [
                bbox[0], bbox[1],
                bbox[2] - bbox[0],
                bbox[3] - bbox[1]
            ]
            area = coco_bbox[2] * coco_bbox[3]

            coco['annotations'].append({
                'id': ann_id,
                'image_id': img_id,
                'category_id': class_to_id[obj['name']],
                'bbox': coco_bbox,
                'area': area,
                'iscrowd': 0,
                'segmentation': [],
            })
            ann_id += 1

    return coco


def save_coco(coco_dict, json_path):
    """保存 COCO 格式 JSON"""
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(coco_dict, f, indent=2, ensure_ascii=False)
    print(f"  Saved COCO: {json_path} ({len(coco_dict['images'])} images, {len(coco_dict['annotations'])} annotations)")


# ==================== NMS 后处理 ====================

def apply_nms(pred_boxes, pred_scores, iou_threshold=0.5, method='hard', sigma=0.5, score_threshold=0.001):
    """
    应用 NMS 过滤检测结果
    pred_boxes: [N, 4] (x1,y1,x2,y2)
    pred_scores: [N]
    method: 'hard' 标准 NMS, 'soft' 高斯衰减 Soft-NMS
    sigma: Soft-NMS 的高斯衰减系数（越小衰减越快）
    score_threshold: Soft-NMS 保留的最低分数
    返回过滤后的索引
    """
    if len(pred_boxes) == 0:
        return torch.tensor([], dtype=torch.long)

    if method == 'soft':
        return soft_nms(pred_boxes, pred_scores, sigma=sigma, score_threshold=score_threshold)
    else:
        keep = ops.nms(pred_boxes, pred_scores, iou_threshold)
        return keep


def soft_nms(boxes, scores, sigma=0.5, score_threshold=0.001):
    """
    Soft-NMS (高斯衰减版)
    不直接删除重叠框，而是按 IoU 距离衰减分数，保留更多密集目标。

    原理: score_i *= exp(-(iou_i,j)^2 / sigma)
    - sigma 越小，衰减越快（接近 Hard-NMS）
    - sigma 越大，衰减越慢（保留更多框）

    boxes: [N, 4] (x1,y1,x2,y2)
    scores: [N]
    sigma: 高斯衰减参数，推荐 0.5（通用）或 0.3（更保守）
    score_threshold: 最低保留分数
    返回: 保留的索引
    """
    N = boxes.shape[0]
    if N == 0:
        return torch.tensor([], dtype=torch.long)

    # 按分数降序排列
    indices = scores.argsort(descending=True)
    indices = indices.clone()
    scores = scores.clone()

    keep = []

    for i in range(N):
        idx = indices[i].item()
        if scores[idx] < score_threshold:
            break

        keep.append(idx)

        if i == N - 1:
            break

        # 计算当前框与剩余框的 IoU
        remaining = indices[i + 1:]
        ious = ops.box_iou(
            boxes[idx:idx + 1],
            boxes[remaining]
        ).squeeze(0)  # [remaining_len]

        # 高斯衰减：IoU 越大，分数衰减越多
        decay = torch.exp(-(ious ** 2) / sigma)
        scores[remaining] *= decay

    return torch.tensor(keep, dtype=torch.long)


def box_cxcywh_to_xyxy(x):
    """将 (cx, cy, w, h) 格式转为 (x1, y1, x2, y2)"""
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    """将 (x1, y1, x2, y2) 格式转为 (cx, cy, w, h)"""
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_normalized(x, img_w, img_h):
    """将绝对坐标转为归一化坐标 [0, 1]"""
    x_norm = x.clone()
    x_norm[:, 0] /= img_w
    x_norm[:, 1] /= img_h
    x_norm[:, 2] /= img_w
    x_norm[:, 3] /= img_h
    return x_norm


# ==================== 可视化 ====================

def draw_boxes(image, boxes, labels=None, scores=None, color='red', width=2):
    """
    在图片上绘制边界框
    boxes: [N, 4] (x1, y1, x2, y2)
    """
    draw = ImageDraw.Draw(image)

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box

        # 绘制矩形
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)

        # 绘制标签
        if labels is not None and i < len(labels):
            label = labels[i]
            if scores is not None and i < len(scores):
                label = f"{label}: {scores[i]:.2f}"

            # 绘制背景框
            try:
                font = ImageFont.truetype("arial.ttf", 16)
            except OSError:
                font = ImageFont.load_default()

            bbox = font.getbbox(label)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            draw.rectangle([x1, y1 - text_h - 4, x1 + text_w, y1], fill=color)
            draw.text((x1, y1 - text_h - 4), label, fill='white', font=font)

    return image


def save_detection_visualization(image, gt_boxes, pred_boxes, pred_scores,
                                save_path, gt_labels=None, pred_labels=None):
    """
    保存检测结果可视化（GT绿色，预测红色）
    """
    vis_img = image.copy()

    # 绘制 GT（绿色）
    if gt_boxes is not None and len(gt_boxes) > 0:
        vis_img = draw_boxes(vis_img, gt_boxes, labels=gt_labels,
                            color='green', width=2)

    # 绘制预测（红色）
    if pred_boxes is not None and len(pred_boxes) > 0:
        vis_img = draw_boxes(vis_img, pred_boxes, labels=pred_labels,
                            scores=pred_scores, color='red', width=2)

    vis_img.save(save_path)
    return vis_img


# ==================== 训练历史保存 ====================

def save_history(history, save_dir):
    """保存训练历史为 JSON 和 CSV"""
    if not history:
        return

    # JSON
    json_path = os.path.join(save_dir, "history.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    # CSV
    csv_path = os.path.join(save_dir, "history.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)

    print(f"  History saved: {json_path}")


# ==================== Checkpoint ====================

def save_checkpoint(model, optimizer, epoch, metric, save_path):
    """保存训练 checkpoint"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metric': metric,
    }, save_path)


def load_checkpoint(model, optimizer, save_path):
    """加载训练 checkpoint"""
    checkpoint = torch.load(save_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint.get('epoch', 0), checkpoint.get('metric', 0)


class AverageMeter:
    """计算并存储平均值和当前值"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# ==================== mAP 计算 ====================

def compute_iou(box1, box2):
    """
    计算 IoU（高效广播，无显式 expand）
    box1: [N, 4] (x1, y1, x2, y2)
    box2: [M, 4] (x1, y1, x2, y2)
    返回: [N, M] IoU矩阵
    """
    # 广播: box1[N,1,4] vs box2[1,M,4] → 交集[N,M]
    inter_x1 = torch.max(box1[:, None, 0], box2[None, :, 0])
    inter_y1 = torch.max(box1[:, None, 1], box2[None, :, 1])
    inter_x2 = torch.min(box1[:, None, 2], box2[None, :, 2])
    inter_y2 = torch.min(box1[:, None, 3], box2[None, :, 3])

    inter_area = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)

    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])  # [N]
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])  # [M]
    union_area = area1[:, None] + area2[None, :] - inter_area

    return inter_area / (union_area + 1e-6)


def compute_ap(predictions, ground_truths, iou_threshold=0.5):
    """
    计算单个类别的 AP（Average Precision）
    predictions: list of {'bbox': [x1,y1,x2,y2], 'score': float, 'image_id': int}
    ground_truths: list of {'bbox': [x1,y1,x2,y2], 'image_id': int}
    """
    # 按 score 降序排列
    predictions = sorted(predictions, key=lambda x: x['score'], reverse=True)

    tp = []
    fp = []
    matched_gt = set()

    for pred in predictions:
        best_iou = 0
        best_gt_idx = None

        for gt_idx, gt in enumerate(ground_truths):
            if gt['image_id'] != pred['image_id']:
                continue
            if gt_idx in matched_gt:
                continue

            iou = compute_iou(
                torch.tensor(pred['bbox']).unsqueeze(0),
                torch.tensor(gt['bbox']).unsqueeze(0)
            ).item()

            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= iou_threshold and best_gt_idx is not None:
            tp.append(1)
            fp.append(0)
            matched_gt.add(best_gt_idx)
        else:
            tp.append(0)
            fp.append(1)

    tp = torch.cumsum(torch.tensor(tp, dtype=torch.float), dim=0)
    fp = torch.cumsum(torch.tensor(fp, dtype=torch.float), dim=0)

    precision = tp / (tp + fp)
    recall = tp / (len(ground_truths) + 1e-6)

    # AP: PR曲线下面积（全点插值，比11点更准确）
    recall_np = recall.numpy()
    precision_np = precision.numpy()
    # 使 precision 单调递减
    for i in range(len(precision_np) - 2, -1, -1):
        precision_np[i] = max(precision_np[i], precision_np[i + 1])
    # 计算 PR 曲线下面积
    recall_np = np.concatenate(([0.0], recall_np, [1.0]))
    precision_np = np.concatenate(([1.0], precision_np, [0.0]))
    ap = np.sum((recall_np[1:] - recall_np[:-1]) * precision_np[1:])

    return ap


if __name__ == "__main__":
    # 测试 VOC 解析
    print("Utils module loaded successfully!")
    print(f"  Functions: parse_voc_xml, voc_to_coco, apply_nms, compute_ap")
    print(f"  Visualization: draw_boxes, save_detection_visualization")
