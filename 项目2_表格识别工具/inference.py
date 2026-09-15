"""
推理脚本
端到端表格识别：检测 → 裁剪 → 结构识别 → 输出 HTML/JSON
"""
import os
import argparse
import json
import torch
from PIL import Image, ImageDraw
import torchvision.transforms as T

from config import (
    DEVICE, IMAGE_SIZE, STRUCTURE_CLASSES,
    DETECTION_SCORE_THRESHOLD, NMS_IOU_THRESHOLD, MAX_DETECTIONS,
    NMS_METHOD, SOFT_NMS_SIGMA, SOFT_NMS_SCORE_THRESH,
    STRUCTURE_NUM_CLASSES
)
from model import TableDetector
from utils import load_checkpoint, draw_boxes, soft_nms
import torchvision.ops as ops


class TableRecognitionPipeline:
    """
    端到端表格识别流水线
    1. 检测文档中的表格区域
    2. 裁剪表格
    3. 识别表格结构
    4. 输出结构化结果
    """

    def __init__(self, detector_path=None, structure_path=None, device=None):
        self.device = device or torch.device(DEVICE)

        # 加载检测模型（Faster R-CNN, 2类）
        self.detector = TableDetector(num_classes=2, pretrained=False).to(self.device)
        if detector_path and os.path.exists(detector_path):
            load_checkpoint(self.detector, None, detector_path)
            print(f"  Detector loaded: {detector_path}")
        else:
            print("  ⚠ No detector model loaded")

        # 加载结构识别模型（Faster R-CNN, 3类, 低内部 NMS）
        self.structure_recognizer = TableDetector(
            num_classes=STRUCTURE_NUM_CLASSES, pretrained=False,
            detections_per_img=2000, nms_thresh=STRUCTURE_NMS_THRESH
        ).to(self.device)
        if structure_path and os.path.exists(structure_path):
            load_checkpoint(self.structure_recognizer, None, structure_path)
            print(f"  Structure recognizer loaded: {structure_path}")
        else:
            print("  ⚠ No structure recognizer model loaded")

        # 图像预处理
        self.detector_transform = T.Compose([
            T.Resize([IMAGE_SIZE, IMAGE_SIZE]),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.structure_transform = T.Compose([
            T.Resize([IMAGE_SIZE, IMAGE_SIZE]),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def detect_tables(self, image, score_threshold=None):
        """
        检测文档中的表格
        返回: [{'bbox': [x1,y1,x2,y2], 'score': float}, ...]
        """
        self.detector.eval()
        score_threshold = score_threshold or DETECTION_SCORE_THRESHOLD

        # 预处理
        orig_w, orig_h = image.size
        img_tensor = self.detector_transform(image).unsqueeze(0).to(self.device)

        # 推理
        with torch.no_grad():
            outputs = self.detector([img_tensor[0]])

        output = outputs[0]
        boxes = output['boxes'].cpu()
        scores = output['scores'].cpu()
        labels = output['labels'].cpu()

        # 过滤
        keep = scores > score_threshold
        boxes = boxes[keep]
        scores = scores[keep]

        # NMS（支持 Soft-NMS 保留密集目标）
        if len(boxes) > 0:
            if NMS_METHOD == 'soft':
                nms_keep = soft_nms(boxes, scores,
                                    sigma=SOFT_NMS_SIGMA,
                                    score_threshold=SOFT_NMS_SCORE_THRESH)
            else:
                nms_keep = ops.nms(boxes, scores, NMS_IOU_THRESHOLD)
            boxes = boxes[nms_keep]
            scores = scores[nms_keep]

        # 映射回原图坐标
        scale_x = orig_w / IMAGE_SIZE
        scale_y = orig_h / IMAGE_SIZE
        boxes[:, 0] *= scale_x
        boxes[:, 1] *= scale_y
        boxes[:, 2] *= scale_x
        boxes[:, 3] *= scale_y

        # 限制检测数
        if len(boxes) > MAX_DETECTIONS:
            top_k = scores.topk(MAX_DETECTIONS)
            boxes = boxes[top_k.indices]
            scores = scores[top_k.indices]

        tables = []
        for i in range(len(boxes)):
            tables.append({
                'bbox': boxes[i].tolist(),
                'score': scores[i].item(),
            })

        return tables

    def recognize_structure(self, table_image):
        """
        识别表格结构（基于 Faster R-CNN 检测）
        返回: {'cells': [...], 'rows': [...], 'columns': [...]}
        """
        self.structure_recognizer.eval()

        # 预处理
        img_tensor = self.structure_transform(table_image).unsqueeze(0).to(self.device)

        # 推理（Faster R-CNN 输出: [{'boxes', 'scores', 'labels'}, ...]）
        with torch.no_grad():
            outputs = self.structure_recognizer([img_tensor[0]])

        output = outputs[0]
        boxes = output['boxes'].cpu()
        scores = output['scores'].cpu()
        labels = output['labels'].cpu()

        # 过滤低置信度
        keep = scores > DETECTION_SCORE_THRESHOLD
        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]

        # NMS
        if len(boxes) > 0:
            if NMS_METHOD == 'soft':
                nms_keep = soft_nms(boxes, scores,
                                    sigma=SOFT_NMS_SIGMA,
                                    score_threshold=SOFT_NMS_SCORE_THRESH)
            else:
                nms_keep = ops.nms(boxes, scores, NMS_IOU_THRESHOLD)
            boxes = boxes[nms_keep]
            scores = scores[nms_keep]
            labels = labels[nms_keep]

        # 映射回原图坐标
        scale_x = table_image.width / IMAGE_SIZE
        scale_y = table_image.height / IMAGE_SIZE
        if len(boxes) > 0:
            boxes[:, 0] *= scale_x
            boxes[:, 1] *= scale_y
            boxes[:, 2] *= scale_x
            boxes[:, 3] *= scale_y

        # 按类别分组
        cells = []
        rows = []
        columns = []
        headers = []
        id_to_name = {v: k for k, v in STRUCTURE_CLASSES.items()}

        for i in range(len(boxes)):
            cls_id = labels[i].item()
            cls_name = id_to_name.get(cls_id, 'unknown')

            element = {
                'class': cls_name,
                'bbox': boxes[i].tolist(),
                'confidence': scores[i].item(),
            }

            if cls_name == 'cell':
                cells.append(element)
            elif cls_name == 'row':
                rows.append(element)
            elif cls_name == 'column':
                columns.append(element)
            elif cls_name == 'header':
                headers.append(element)

        return {
            'cells': cells,
            'rows': rows,
            'columns': columns,
            'headers': headers,
        }

    def recognize(self, image, return_visualization=False):
        """
        完整的端到端表格识别
        返回: {'tables': [...]}
        """
        results = {'tables': [], 'image_size': list(image.size)}

        # 1. 检测表格
        tables = self.detect_tables(image)
        print(f"  Detected {len(tables)} table(s)")

        # 2. 对每个表格进行结构识别
        vis_image = image.copy() if return_visualization else None

        for table_idx, table in enumerate(tables):
            x1, y1, x2, y2 = [int(v) for v in table['bbox']]

            # 裁剪表格（带一点边距）
            margin = 5
            x1 = max(0, x1 - margin)
            y1 = max(0, y1 - margin)
            x2 = min(image.width, x2 + margin)
            y2 = min(image.height, y2 + margin)

            table_image = image.crop((x1, y1, x2, y2))

            # 3. 识别结构
            structure = self.recognize_structure(table_image)

            # 4. 构建 HTML
            html = self._structure_to_html(structure, table_image.width, table_image.height)

            table_result = {
                'index': table_idx,
                'bbox': table['bbox'],
                'detection_score': table['score'],
                'structure': structure,
                'html': html,
            }
            results['tables'].append(table_result)

            # 可视化
            if return_visualization and vis_image is not None:
                draw = ImageDraw.Draw(vis_image)
                draw.rectangle([x1, y1, x2, y2], outline='red', width=3)
                draw.text((x1, y1 - 15), f"Table {table_idx}: {table['score']:.2f}", fill='red')

        return results

    def _structure_to_html(self, structure, width, height):
        """
        将结构识别结果转为 HTML 表格
        """
        cells = structure.get('cells', [])
        rows = structure.get('rows', [])
        columns = structure.get('columns', [])
        headers = structure.get('headers', [])

        if not cells:
            return "<p>No cells detected</p>"

        # 按 y 坐标排序分组为行
        cells_sorted = sorted(cells, key=lambda c: (c['bbox'][1], c['bbox'][0]))

        # 简单的按 y 分组
        row_groups = []
        current_row = []
        last_y = None
        y_threshold = height / 10  # 行间最小距离

        for cell in cells_sorted:
            y = cell['bbox'][1]
            if last_y is None or abs(y - last_y) < y_threshold:
                current_row.append(cell)
            else:
                if current_row:
                    row_groups.append(sorted(current_row, key=lambda c: c['bbox'][0]))
                current_row = [cell]
            last_y = y
        if current_row:
            row_groups.append(sorted(current_row, key=lambda c: c['bbox'][0]))

        # 生成 HTML
        html = "<table border='1'>\n"
        for row_idx, row_cells in enumerate(row_groups):
            is_header = any(
                h['bbox'][1] < height * 0.2
                for h in headers
            ) and row_idx == 0

            tag = "th" if is_header else "td"
            html += "  <tr>\n"
            for cell in row_cells:
                html += f"    <{tag}></{tag}>\n"
            html += "  </tr>\n"
        html += "</table>"

        return html

    def _structure_to_json(self, structure):
        """将结构识别结果转为 JSON"""
        return {
            'cells': [
                {
                    'bbox': c['bbox'],
                    'confidence': c['confidence']
                }
                for c in structure.get('cells', [])
            ],
            'rows': len(structure.get('rows', [])),
            'columns': len(structure.get('columns', [])),
        }


def process_image(pipeline, image_path, output_dir, return_vis=False):
    """处理单张图片"""
    image = Image.open(image_path).convert('RGB')
    results = pipeline.recognize(image, return_visualization=return_vis)

    # 保存结果
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    # JSON 结果
    json_path = os.path.join(output_dir, f"{base_name}.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)

    # HTML 结果
    html_path = os.path.join(output_dir, f"{base_name}.html")
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write("<!DOCTYPE html>\n<html>\n<head>\n")
        f.write("<style>table { border-collapse: collapse; margin: 10px; }</style>\n")
        f.write("</head>\n<body>\n")
        f.write(f"<h2>Tables in {os.path.basename(image_path)}</h2>\n")
        for table in results['tables']:
            f.write(f"\n<!-- Table {table['index']}, score: {table['detection_score']:.3f} -->\n")
            f.write(table['html'])
            f.write("\n<br>\n")
        f.write("</body>\n</html>")

    print(f"  Results saved:")
    print(f"    JSON: {json_path}")
    print(f"    HTML: {html_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Table Recognition Inference')
    parser.add_argument('--input', type=str, required=True, help='输入图片或目录')
    parser.add_argument('--output', type=str, default='output', help='输出目录')
    parser.add_argument('--detector', type=str, default=None, help='检测模型路径')
    parser.add_argument('--structure', type=str, default=None, help='结构识别模型路径')
    parser.add_argument('--visualize', action='store_true', help='保存可视化结果')
    parser.add_argument('--device', type=str, default=None, help='设备')
    args = parser.parse_args()

    device = torch.device(args.device or DEVICE)
    print(f"\n  Device: {device}")

    # 初始化流水线
    print("\n  Initializing pipeline...")
    pipeline = TableRecognitionPipeline(
        detector_path=args.detector,
        structure_path=args.structure,
        device=device
    )

    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)

    # 处理输入
    input_path = args.input
    if os.path.isfile(input_path):
        # 单张图片
        print(f"\n  Processing: {input_path}")
        results = process_image(pipeline, input_path, args.output, args.visualize)
        print(f"  Detected {len(results['tables'])} table(s)")

    elif os.path.isdir(input_path):
        # 批量处理
        image_files = [
            f for f in os.listdir(input_path)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))
        ]
        print(f"\n  Found {len(image_files)} images in {input_path}")

        for i, img_file in enumerate(image_files):
            img_path = os.path.join(input_path, img_file)
            print(f"\n  [{i+1}/{len(image_files)}] Processing: {img_file}")
            try:
                results = process_image(pipeline, img_path, args.output, args.visualize)
                print(f"    Detected {len(results['tables'])} table(s)")
            except Exception as e:
                print(f"    Error: {e}")

    print(f"\n  Done! Results saved to: {args.output}")


if __name__ == "__main__":
    main()
