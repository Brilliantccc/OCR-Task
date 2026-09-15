"""
表格识别模型
TableDetector: Faster R-CNN 目标检测模型
- TrackA: 表格检测 (num_classes=2, 背景+表格)
- TrackB: 结构识别 (num_classes=3, 背景+表格+cell)
"""
import torch
import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from config import DETECTION_NUM_CLASSES, IMAGE_SIZE


class TableDetector(nn.Module):
    """
    通用目标检测模型：Faster R-CNN (ResNet50-FPN backbone)
    - 检测任务: num_classes=2 (background + table)
    - 结构任务: num_classes=3 (background + table + cell)
    """

    def __init__(self, num_classes=None, pretrained=True, detections_per_img=100, nms_thresh=None):
        super(TableDetector, self).__init__()
        self.num_classes = num_classes or DETECTION_NUM_CLASSES

        # torchvision 预训练 Faster R-CNN
        self.backbone = fasterrcnn_resnet50_fpn(
            weights='DEFAULT' if pretrained else None,
            trainable_backbone_layers=3
        )

        # 替换分类头
        in_features = self.backbone.roi_heads.box_predictor.cls_score.in_features
        self.backbone.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)

        # 每张图的最大检测数
        self.backbone.roi_heads.detections_per_img = detections_per_img

        # 内部 NMS 阈值
        if nms_thresh is not None:
            self.backbone.roi_heads.nms_thresh = nms_thresh

        # RPN 优化：大幅增加候选框数量（密集 cell 场景需要更多候选框）
        self.backbone.rpn._pre_nms_top_n = {'training': 6000, 'testing': 3000}
        self.backbone.rpn._post_nms_top_n = {'training': 3000, 'testing': 1500}

        # 调整采样器：增加正样本比例，适配密集目标场景
        self.backbone.roi_heads.fg_bg_sampler.batch_size_per_image = 512
        self.backbone.roi_heads.fg_bg_sampler.positive_fraction = 0.5

    def load_backbone_from(self, checkpoint_path):
        """
        从另一个 TableDetector 加载 backbone 权重（迁移学习）
        checkpoint_path: 检测模型的 checkpoint 路径
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        state_dict = checkpoint['model_state_dict']

        # 只加载 backbone 相关的权重（跳过 box_predictor）
        backbone_state_dict = {}
        for k, v in state_dict.items():
            if 'backbone' in k and 'box_predictor' not in k:
                new_key = k.replace('backbone.', '', 1)
                backbone_state_dict[new_key] = v

        missing, unexpected = self.backbone.load_state_dict(backbone_state_dict, strict=False)
        print(f"  Loaded backbone from {checkpoint_path}")
        print(f"  Matched: {len(backbone_state_dict) - len(missing)} keys")
        if missing:
            print(f"  Missing: {len(missing)} keys")
        return self

    def forward(self, images, targets=None):
        """
        训练时: images是tensor列表, targets是标注字典列表
        推理时: images是tensor列表
        """
        if self.training and targets is not None:
            loss_dict = self.backbone(images, targets)
            return loss_dict
        else:
            self.backbone.eval()
            with torch.no_grad():
                predictions = self.backbone(images)
            return predictions


# ==================== 权重初始化 ====================

def weights_init(m):
    """初始化模型权重"""
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0, 0.01)
        nn.init.constant_(m.bias, 0)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 测试表格检测模型 (2类)
    print("\n--- Table Detector (Faster R-CNN, 2 classes) ---")
    detector = TableDetector(num_classes=2, pretrained=False).to(device)
    dummy_img = torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE).to(device)
    detector.eval()
    with torch.no_grad():
        out = detector([dummy_img[0], dummy_img[1]])
    print(f"  Input: {dummy_img.shape}")
    print(f"  Output: {len(out)} predictions")
    if out:
        print(f"  Sample: {out[0]['boxes'].shape[0]} detections")
    print(f"  Params: {sum(p.numel() for p in detector.parameters()):,}")

    # 测试结构识别模型 (3类)
    print("\n--- Structure Detector (Faster R-CNN, 3 classes) ---")
    struct_model = TableDetector(num_classes=3, pretrained=False, detections_per_img=2000).to(device)
    struct_model.eval()
    with torch.no_grad():
        out = struct_model([dummy_img[0], dummy_img[1]])
    print(f"  Input: {dummy_img.shape}")
    print(f"  Output: {len(out)} predictions")
    if out:
        print(f"  Sample: {out[0]['boxes'].shape[0]} detections")
    print(f"  Params: {sum(p.numel() for p in struct_model.parameters()):,}")
