"""
数据集加载（纯 OpenCV + Torch，无 PIL 依赖）
支持: TrackA（表格检测）+ TrackB（结构识别）
"""
import os
import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader

from config import (
    TRACKA_TRAIN_IMG, TRACKA_TRAIN_ANN, TRACKA_TEST_IMG, TRACKA_TEST_ANN,
    TRACKB_TRAIN_IMG, TRACKB_TRAIN_ANN, TRACKB_TEST_IMG, TRACKB_TEST_ANN,
    DETECTION_CLASSES, STRUCTURE_CLASSES,
    DETECTION_BATCH_SIZE, STRUCTURE_BATCH_SIZE,
    IMAGE_SIZE, WORKERS, PIN_MEMORY,
    VAL_RATIO, AUGMENTATION
)
from utils import parse_voc_xml

# ImageNet 归一化常量（模块级，避免重复创建）
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


# ==================== 数据增强 ====================

class TableDetectionAugmentation:
    """表格检测数据增强（纯 OpenCV，bbox 跟随变换）"""

    def __init__(self, is_train=True):
        self.is_train = is_train
        self.aug_config = AUGMENTATION

    def __call__(self, image, target):
        """
        image: numpy array (H, W, 3) RGB
        target: dict with 'boxes' [N,4] (x1,y1,x2,y2) and 'labels' [N]
        """
        if not self.is_train:
            return image, target

        boxes = target['boxes'].clone()
        labels = target['labels'].clone()
        img_h, img_w = image.shape[:2]

        # ===== 1. 随机缩放裁剪（几何变换，需要同步 bbox）=====
        if random.random() < self.aug_config.get('scale_crop_prob', 0.0):
            image, boxes, labels = self._random_scale_crop(
                image, boxes, labels, img_h, img_w)
            img_h, img_w = image.shape[:2]

        # ===== 2. 随机翻转 =====
        if random.random() < self.aug_config.get('horizontal_flip_prob', 0.5):
            image = np.flip(image, axis=1).copy()
            x1 = boxes[:, 0].clone()
            x2 = boxes[:, 2].clone()
            boxes[:, 0] = img_w - x2
            boxes[:, 2] = img_w - x1

        if random.random() < self.aug_config.get('vertical_flip_prob', 0.0):
            image = np.flip(image, axis=0).copy()
            y1 = boxes[:, 1].clone()
            y2 = boxes[:, 3].clone()
            boxes[:, 1] = img_h - y2
            boxes[:, 3] = img_h - y1

        # ===== 3. 颜色变换 =====
        if random.random() < 0.5:
            alpha = random.uniform(*self.aug_config['brightness_range'])
            image = cv2.convertScaleAbs(image, alpha=alpha, beta=0)

        if random.random() < 0.5:
            alpha = random.uniform(*self.aug_config['contrast_range'])
            image = cv2.convertScaleAbs(image, alpha=alpha, beta=0)

        if random.random() < 0.5:
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
            s_factor = random.uniform(*self.aug_config.get('saturation_range', (0.8, 1.2)))
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * s_factor, 0, 255)
            image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

        # ===== 4. 随机灰度化 =====
        if random.random() < self.aug_config.get('grayscale_prob', 0.0):
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            image = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

        # ===== 5. 随机 JPEG 压缩 =====
        if random.random() < self.aug_config.get('jpeg_compress_prob', 0.0):
            quality = random.randint(*self.aug_config.get('jpeg_quality_range', (30, 90)))
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            _, buf = cv2.imencode('.jpg', cv2.cvtColor(image, cv2.COLOR_RGB2BGR), encode_param)
            image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # ===== 6. 噪声与模糊 =====
        if random.random() < self.aug_config['noise_prob']:
            noise = np.random.normal(0, 10, image.shape).astype(np.float32)
            image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        if random.random() < self.aug_config.get('blur_prob', 0.0):
            ksize = int(random.uniform(0.5, 1.5)) * 2 + 1
            image = cv2.GaussianBlur(image, (ksize, ksize), 0)

        # ===== 后处理 =====
        if len(boxes) > 0:
            boxes[:, 0] = boxes[:, 0].clamp(0, img_w)
            boxes[:, 1] = boxes[:, 1].clamp(0, img_h)
            boxes[:, 2] = boxes[:, 2].clamp(0, img_w)
            boxes[:, 3] = boxes[:, 3].clamp(0, img_h)

            valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            boxes = boxes[valid]
            labels = labels[valid]

        target['boxes'] = boxes
        target['labels'] = labels

        return image, target

    def _random_scale_crop(self, image, boxes, labels, img_h, img_w):
        """
        随机缩放裁剪：缩放图片和bbox，裁剪到原尺寸
        - scale > 1.0: 放大后裁剪（zoom in，看到更多细节）
        - scale < 1.0: 缩小后补灰边（zoom out，看到更多上下文）
        """
        scale_range = self.aug_config.get('scale_range', (0.8, 1.2))
        scale = random.uniform(*scale_range)

        new_h = int(img_h * scale)
        new_w = int(img_w * scale)

        # 缩放图片
        image_scaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # 缩放 bbox
        if len(boxes) > 0:
            boxes_scaled = boxes.clone()
            boxes_scaled[:, [0, 2]] *= scale
            boxes_scaled[:, [1, 3]] *= scale
        else:
            boxes_scaled = boxes.clone()

        if scale >= 1.0:
            # 放大：随机裁剪中心区域
            crop_x = random.randint(0, new_w - img_w)
            crop_y = random.randint(0, new_h - img_h)
            image_cropped = image_scaled[crop_y:crop_y + img_h, crop_x:crop_x + img_w]

            # bbox 坐标平移
            if len(boxes_scaled) > 0:
                boxes_scaled[:, [0, 2]] -= crop_x
                boxes_scaled[:, [1, 3]] -= crop_y
        else:
            # 缩小：居中放置，周围补灰边
            pad_x = (img_w - new_w) // 2
            pad_y = (img_h - new_h) // 2
            image_cropped = np.full((img_h, img_w, 3), 128, dtype=np.uint8)
            image_cropped[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = image_scaled

            # bbox 坐标平移
            if len(boxes_scaled) > 0:
                boxes_scaled[:, [0, 2]] += pad_x
                boxes_scaled[:, [1, 3]] += pad_y

        # 过滤：裁剪后 bbox 被截断超过 50% 的丢弃
        if len(boxes_scaled) > 0:
            orig_areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            # clip 到图片范围
            clipped = boxes_scaled.clone()
            clipped[:, 0] = clipped[:, 0].clamp(0, img_w)
            clipped[:, 1] = clipped[:, 1].clamp(0, img_h)
            clipped[:, 2] = clipped[:, 2].clamp(0, img_w)
            clipped[:, 3] = clipped[:, 3].clamp(0, img_h)
            new_areas = (clipped[:, 2] - clipped[:, 0]) * (clipped[:, 3] - clipped[:, 1])

            # 保留面积损失不超过 50% 的 bbox
            valid = new_areas > orig_areas * 0.5
            boxes_scaled = clipped[valid]
            labels = labels[valid]

        return image_cropped, boxes_scaled, labels


# ==================== 表格检测数据集 ====================

class TableDetectionDataset(Dataset):
    """
    表格检测数据集（TrackA）
    加载整页文档图片 + 表格边界框标注
    """

    def __init__(self, img_dir, ann_dir, class_to_id=None, transform=None,
                 is_train=True, max_objects=50):
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.class_to_id = class_to_id or DETECTION_CLASSES
        self.transform = transform
        self.is_train = is_train
        self.max_objects = max_objects
        self.augmentation = TableDetectionAugmentation(is_train)

        # 扫描标注文件
        self.samples = []
        if os.path.exists(ann_dir):
            for ann_file in sorted(os.listdir(ann_dir)):
                if ann_file.endswith('.xml'):
                    xml_path = os.path.join(ann_dir, ann_file)
                    ann = parse_voc_xml(xml_path)

                    # 检查图片是否存在
                    img_path = os.path.join(img_dir, ann['filename'])
                    if os.path.exists(img_path):
                        self.samples.append({
                            'img_path': img_path,
                            'annotation': ann
                        })

        print(f"  Loaded {len(self.samples)} samples from {ann_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        ann = sample['annotation']

        # OpenCV 加载图片（返回 numpy array）
        img_cv = cv2.imread(sample['img_path'], cv2.IMREAD_COLOR)
        if img_cv is None:
            raise FileNotFoundError(f"无法读取图片: {sample['img_path']}")
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        image = img_cv
        orig_h, orig_w = image.shape[:2]

        # cTDaR XML 不包含图片尺寸，从图片读取
        if ann['width'] == 0 or ann['height'] == 0:
            ann['width'] = orig_w
            ann['height'] = orig_h

        # 解析标注
        boxes = []
        labels = []
        for obj in ann['objects']:
            if obj['name'] not in self.class_to_id:
                continue
            if obj['difficult']:
                continue

            bbox = obj['bbox']
            # 确保 bbox 有效
            if bbox[2] > bbox[0] and bbox[3] > bbox[1]:
                boxes.append(bbox)
                labels.append(self.class_to_id[obj['name']])

        # 转为 tensor
        if boxes:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.long)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.long)

        target = {'boxes': boxes, 'labels': labels}

        # 数据增强（numpy array）
        image, target = self.augmentation(image, target)

        # Resize 到固定尺寸（OpenCV）
        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))

        # 调整 bbox 到新尺寸
        scale_x = IMAGE_SIZE / orig_w
        scale_y = IMAGE_SIZE / orig_h
        if len(target['boxes']) > 0:
            target['boxes'][:, 0] *= scale_x
            target['boxes'][:, 1] *= scale_y
            target['boxes'][:, 2] *= scale_x
            target['boxes'][:, 3] *= scale_y

        # numpy -> tensor: HWC -> CHW, uint8 -> float, /255, normalize
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        mean = _IMAGENET_MEAN
        std = _IMAGENET_STD
        image = (image - mean) / std

        target['image_id'] = torch.tensor([idx], dtype=torch.long)
        target['area'] = (target['boxes'][:, 2] - target['boxes'][:, 0]) * \
                         (target['boxes'][:, 3] - target['boxes'][:, 1])
        target['iscrowd'] = torch.zeros((len(target['labels']),), dtype=torch.long)

        return image, target


# ==================== 结构识别数据集 ====================

class TableStructureDataset(Dataset):
    """
    表格结构识别数据集（TrackB）
    加载裁剪后的表格图片 + 结构元素标注
    """

    def __init__(self, img_dir, ann_dir, class_to_id=None, transform=None,
                 is_train=True, max_objects=100):
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.class_to_id = class_to_id or STRUCTURE_CLASSES
        self.transform = transform
        self.is_train = is_train
        self.max_objects = max_objects
        self.augmentation = TableDetectionAugmentation(is_train)

        # 扫描标注文件
        self.samples = []
        if os.path.exists(ann_dir):
            for ann_file in sorted(os.listdir(ann_dir)):
                if ann_file.endswith('.xml'):
                    xml_path = os.path.join(ann_dir, ann_file)
                    ann = parse_voc_xml(xml_path)

                    img_path = os.path.join(img_dir, ann['filename'])
                    if os.path.exists(img_path):
                        self.samples.append({
                            'img_path': img_path,
                            'annotation': ann
                        })

        print(f"  Loaded {len(self.samples)} samples from {ann_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        ann = sample['annotation']

        # OpenCV 加载图片
        img_cv = cv2.imread(sample['img_path'], cv2.IMREAD_COLOR)
        if img_cv is None:
            raise FileNotFoundError(f"无法读取图片: {sample['img_path']}")
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        image = img_cv
        orig_h, orig_w = image.shape[:2]

        # 解析标注
        boxes = []
        labels = []
        for obj in ann['objects']:
            if obj['name'] not in self.class_to_id:
                continue
            if obj['difficult']:
                continue

            bbox = obj['bbox']
            if bbox[2] > bbox[0] and bbox[3] > bbox[1]:
                boxes.append(bbox)
                labels.append(self.class_to_id[obj['name']])

        if boxes:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.long)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.long)

        target = {'boxes': boxes, 'labels': labels}

        # 数据增强（numpy array）
        image, target = self.augmentation(image, target)

        # Resize（OpenCV）
        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))

        # 调整 bbox
        scale_x = IMAGE_SIZE / orig_w
        scale_y = IMAGE_SIZE / orig_h
        if len(target['boxes']) > 0:
            target['boxes'][:, 0] *= scale_x
            target['boxes'][:, 1] *= scale_y
            target['boxes'][:, 2] *= scale_x
            target['boxes'][:, 3] *= scale_y

        # numpy -> tensor: HWC -> CHW, uint8 -> float, /255, normalize
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        mean = _IMAGENET_MEAN
        std = _IMAGENET_STD
        image = (image - mean) / std

        target['image_id'] = torch.tensor([idx], dtype=torch.long)
        target['area'] = (target['boxes'][:, 2] - target['boxes'][:, 0]) * \
                         (target['boxes'][:, 3] - target['boxes'][:, 1])
        target['iscrowd'] = torch.zeros((len(target['labels']),), dtype=torch.long)

        return image, target


# ==================== 数据加载 ====================

def collate_fn(batch):
    """自定义 collate，处理不同数量的标注
    Faster R-CNN 需要 list of images，不是 batched tensor
    """
    images = []
    targets = []
    for img, tgt in batch:
        images.append(img)
        targets.append(tgt)
    return images, targets


def get_detection_loaders():
    """获取表格检测数据加载器"""
    print("\n  Loading detection dataset (TrackA)...")

    train_dataset = TableDetectionDataset(
        TRACKA_TRAIN_IMG, TRACKA_TRAIN_ANN, class_to_id=DETECTION_CLASSES, is_train=True
    )
    test_dataset = TableDetectionDataset(
        TRACKA_TEST_IMG, TRACKA_TEST_ANN, class_to_id=DETECTION_CLASSES, is_train=False
    )

    # 从训练集划分出验证集
    n_val = int(len(train_dataset) * VAL_RATIO)
    n_train = len(train_dataset) - n_val
    generator = torch.Generator().manual_seed(42)
    train_set, val_set = torch.utils.data.random_split(train_dataset, [n_train, n_val], generator=generator)

    worker_kwargs = {'persistent_workers': True, 'prefetch_factor': 3} if WORKERS > 0 else {}
    pin = PIN_MEMORY and torch.cuda.is_available()
    train_loader = DataLoader(train_set, batch_size=DETECTION_BATCH_SIZE, shuffle=True,
                             num_workers=WORKERS, pin_memory=pin, collate_fn=collate_fn,
                             **worker_kwargs)
    val_loader = DataLoader(val_set, batch_size=DETECTION_BATCH_SIZE, shuffle=False,
                           num_workers=WORKERS, pin_memory=pin, collate_fn=collate_fn,
                           **worker_kwargs)
    test_loader = DataLoader(test_dataset, batch_size=DETECTION_BATCH_SIZE, shuffle=False,
                            num_workers=WORKERS, pin_memory=pin, collate_fn=collate_fn,
                            **worker_kwargs)

    return train_loader, val_loader, test_loader


def get_structure_loaders():
    """获取结构识别数据加载器"""
    print("\n  Loading structure dataset (TrackB)...")

    train_dataset = TableStructureDataset(
        TRACKB_TRAIN_IMG, TRACKB_TRAIN_ANN, class_to_id=STRUCTURE_CLASSES, is_train=True
    )
    test_dataset = TableStructureDataset(
        TRACKB_TEST_IMG, TRACKB_TEST_ANN, class_to_id=STRUCTURE_CLASSES, is_train=False
    )

    n_val = int(len(train_dataset) * VAL_RATIO)
    n_train = len(train_dataset) - n_val
    generator = torch.Generator().manual_seed(42)
    train_set, val_set = torch.utils.data.random_split(train_dataset, [n_train, n_val], generator=generator)

    use_cuda = torch.cuda.is_available()
    worker_kwargs = {'persistent_workers': True, 'prefetch_factor': 3} if WORKERS > 0 else {}
    pin = PIN_MEMORY and use_cuda
    train_loader = DataLoader(train_set, batch_size=STRUCTURE_BATCH_SIZE, shuffle=True,
                             num_workers=WORKERS, pin_memory=pin, collate_fn=collate_fn,
                             **worker_kwargs)
    val_loader = DataLoader(val_set, batch_size=STRUCTURE_BATCH_SIZE, shuffle=False,
                           num_workers=WORKERS, pin_memory=pin, collate_fn=collate_fn,
                           **worker_kwargs)
    test_loader = DataLoader(test_dataset, batch_size=STRUCTURE_BATCH_SIZE, shuffle=False,
                            num_workers=WORKERS, pin_memory=pin, collate_fn=collate_fn,
                            **worker_kwargs)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    print("Dataset module loaded successfully!")
    print(f"  TrackA: {TRACKA_TRAIN_IMG}")
    print(f"  TrackB: {TRACKB_TRAIN_IMG}")

    # 测试解析（如果有数据）
    if os.path.exists(TRACKA_TRAIN_ANN):
        xml_files = [f for f in os.listdir(TRACKA_TRAIN_ANN) if f.endswith('.xml')][:3]
        for xml_file in xml_files:
            ann = parse_voc_xml(os.path.join(TRACKA_TRAIN_ANN, xml_file))
            print(f"\n  Sample: {ann['filename']} ({ann['width']}x{ann['height']})")
            for obj in ann['objects']:
                print(f"    {obj['name']}: {obj['bbox']}")
