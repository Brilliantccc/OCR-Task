"""
OCR数据集类 - 使用OpenCV加载图像
将每张图片的每个文本行裁剪为单独样本
"""
import os
import csv
import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader


def load_image_fixed(img_path):
    """加载JPEG图像"""
    with open(img_path, 'rb') as f:
        img_bytes = np.frombuffer(f.read(), dtype=np.uint8)
        image = cv2.imdecode(img_bytes, cv2.IMREAD_GRAYSCALE)
    return image

from config import (
    IMG_DIR, BOX_DIR, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS,
    CHAR_TO_IDX, BATCH_SIZE, WORKERS, PIN_MEMORY,
    TRAIN_RATIO, VAL_RATIO, AUGMENTATION
)


class OCRDataset(Dataset):
    """OCR数据集 - 每个样本是一个文本行"""

    def __init__(self, samples, transform=None, augment=False):
        """
        Args:
            samples: [(img_id, box_index), ...] 每个样本的标识
            transform: 图像变换
            augment: 是否进行数据增强
        """
        self.samples = samples
        self.transform = transform
        self.augment = augment

        # 预加载所有标注到内存
        self.annotations = {}
        for img_id in set(s[0] for s in samples):
            self.annotations[img_id] = self._load_annotations(img_id)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_id, box_idx = self.samples[idx]

        # 加载完整图像（自动修复JPEG）
        img_path = os.path.join(IMG_DIR, f"{img_id}.jpg")
        full_image = load_image_fixed(img_path)

        if full_image is None:
            raise FileNotFoundError(f"无法加载图像: {img_path}")

        # 获取标注
        annotations = self.annotations[img_id]
        if box_idx >= len(annotations):
            # 如果索引越界，使用第一条标注
            box_idx = 0

        bbox, text = annotations[box_idx]

        # 根据bounding box裁剪文本行图像
        x1, y1, x2, y2, x3, y3, x4, y4 = bbox
        # 使用四边形的外接矩形
        min_x = min(x1, x2, x3, x4)
        max_x = max(x1, x2, x3, x4)
        min_y = min(y1, y2, y3, y4)
        max_y = max(y1, y2, y3, y4)

        # 添加一些padding
        padding = 2
        min_x = max(0, min_x - padding)
        min_y = max(0, min_y - padding)
        max_x = min(full_image.shape[1], max_x + padding)
        max_y = min(full_image.shape[0], max_y + padding)

        # 裁剪文本行
        cropped = full_image[min_y:max_y, min_x:max_x]

        # 数据增强
        if self.augment:
            cropped = self._augment(cropped)

        # 预处理图像
        image = self._preprocess(cropped)

        # 转换为tensor
        if self.transform:
            image = self.transform(image)
        else:
            image = torch.from_numpy(image).float().unsqueeze(0)

        # 编码文本
        label = self._encode_text(text)
        label_length = len(label)

        return image, torch.tensor(label, dtype=torch.long), torch.tensor(label_length, dtype=torch.long)

    def _load_annotations(self, img_id):
        """加载图像的所有标注"""
        box_path = os.path.join(BOX_DIR, f"{img_id}.csv")
        annotations = []

        with open(box_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 9:
                    # CSV格式: x1,y1,x2,y2,x3,y3,x4,y4,text
                    bbox = [int(row[i]) for i in range(8)]
                    text = ','.join(row[8:]).strip()
                    if text:
                        annotations.append((bbox, text))

        return annotations if annotations else [([0, 0, 0, 0, 0, 0, 0, 0], '')]

    def _augment(self, image):
        """数据增强"""
        if image.size == 0:
            return image

        # 随机旋转
        if random.random() < 0.3:
            angle = random.uniform(-AUGMENTATION["rotation_range"], AUGMENTATION["rotation_range"])
            h, w = image.shape[:2]
            if h > 0 and w > 0:
                M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
                image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)

        # 随机噪声
        if random.random() < AUGMENTATION["noise_prob"]:
            noise = np.random.normal(0, 25, image.shape).astype(np.float32)
            image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        # 随机模糊
        if random.random() < AUGMENTATION["blur_prob"]:
            k = random.choice([3, 5])
            image = cv2.GaussianBlur(image, (k, k), 0)

        # 随机亮度调整
        if random.random() < 0.3:
            brightness = random.uniform(*AUGMENTATION["brightness_range"])
            image = np.clip(image.astype(np.float32) * brightness, 0, 255).astype(np.uint8)

        return image

    def _preprocess(self, image):
        """图像预处理"""
        if image.size == 0:
            image = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.float32)

        h, w = image.shape[:2]

        # 调整高度
        if h != IMG_HEIGHT:
            ratio = IMG_HEIGHT / h
            new_w = int(w * ratio)
            image = cv2.resize(image, (new_w, IMG_HEIGHT))

        # 调整宽度
        h, w = image.shape[:2]
        if w > IMG_WIDTH:
            # 裁剪
            image = image[:, :IMG_WIDTH]
        elif w < IMG_WIDTH:
            # 填充
            pad_w = IMG_WIDTH - w
            image = cv2.copyMakeBorder(image, 0, 0, 0, pad_w, cv2.BORDER_CONSTANT, value=0)

        # 归一化
        image = image.astype(np.float32) / 255.0

        return image

    def _encode_text(self, text):
        """将文本编码为数字序列"""
        encoded = []
        for ch in text:
            if ch in CHAR_TO_IDX:
                encoded.append(CHAR_TO_IDX[ch])
        return encoded if encoded else [0]


def collate_fn(batch):
    """自定义collate函数，处理不同长度的标签"""
    images, labels, label_lengths = zip(*batch)

    # 填充标签到相同长度，用-1填充
    max_label_length = max(l.item() for l in label_lengths)
    padded_labels = torch.full((len(labels), max_label_length), -1, dtype=torch.long)

    for i, label in enumerate(labels):
        padded_labels[i, :len(label)] = label

    images = torch.stack(images, dim=0)
    label_lengths = torch.stack(label_lengths, dim=0)

    return images, padded_labels, label_lengths


def get_data_loaders():
    """获取训练、验证、测试数据加载器"""
    # 获取所有图像ID
    all_img_ids = [f.replace('.jpg', '') for f in os.listdir(IMG_DIR) if f.endswith('.jpg')]

    # 过滤掉无法读取的图像（使用修复函数）
    valid_img_ids = []
    for img_id in all_img_ids:
        img_path = os.path.join(IMG_DIR, f"{img_id}.jpg")
        if os.path.exists(img_path):
            img = load_image_fixed(img_path)
            if img is not None:
                valid_img_ids.append(img_id)

    print(f"有效图像数: {len(valid_img_ids)} / {len(all_img_ids)}")

    # 生成所有样本 (img_id, box_index)
    all_samples = []
    for img_id in valid_img_ids:
        box_path = os.path.join(BOX_DIR, f"{img_id}.csv")
        num_boxes = 0
        with open(box_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 9 and ','.join(row[8:]).strip():
                    num_boxes += 1
        # 每张图片的每个box作为一个样本
        for i in range(max(1, num_boxes)):
            all_samples.append((img_id, i))

    print(f"总样本数（文本行）: {len(all_samples)}")

    # 随机打乱
    random.seed(42)
    random.shuffle(all_samples)

    # 划分数据集
    n = len(all_samples)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    train_samples = all_samples[:n_train]
    val_samples = all_samples[n_train:n_train + n_val]
    test_samples = all_samples[n_train + n_val:]

    print(f"数据集划分: 训练={len(train_samples)}, 验证={len(val_samples)}, 测试={len(test_samples)}")

    # 创建数据加载器
    train_dataset = OCRDataset(train_samples, augment=True)
    val_dataset = OCRDataset(val_samples, augment=False)
    test_dataset = OCRDataset(test_samples, augment=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=WORKERS,
        pin_memory=PIN_MEMORY,
        collate_fn=collate_fn,
        drop_last=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=WORKERS,
        pin_memory=PIN_MEMORY,
        collate_fn=collate_fn,
        persistent_workers=True,
        prefetch_factor=2
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=WORKERS,
        pin_memory=PIN_MEMORY,
        collate_fn=collate_fn,
        persistent_workers=True,
        prefetch_factor=2
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # 测试数据集
    train_loader, val_loader, test_loader = get_data_loaders()

    # 获取一个batch
    for images, labels, label_lengths in train_loader:
        print(f"图像形状: {images.shape}")
        print(f"标签形状: {labels.shape}")
        print(f"标签长度: {label_lengths[:5]}")
        print(f"标签最小值: {labels.min().item()}")
        print(f"标签最大值: {labels.max().item()}")
        break
