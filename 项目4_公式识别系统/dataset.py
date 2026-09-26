"""
公式识别数据集
支持 CROHME（灰度）+ HME100K（RGB）统一加载
标签格式: filename\tLaTeX tokens (空格分隔)
"""
import os
import random
import numpy as np
import cv2
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Sampler

from config import (
    IMG_HEIGHT, IMG_MAX_WIDTH, PAD_TOKEN, MAX_SEQ_LEN,
    CROHME_TRAIN_IMG, CROHME_TRAIN_LABEL, CROHME_EVAL_IMG, CROHME_EVAL_LABEL,
    HME100K_TRAIN_IMG, HME100K_TRAIN_LABEL, HME100K_TEST_IMG, HME100K_TEST_LABEL,
    BATCH_SIZE, NUM_WORKERS, PERSISTENT_WORKERS, PREFETCH_FACTOR,
    BUCKET_BY_WIDTH, BUCKET_BATCHES_PER_BUCKET, WIDTH_SCAN_THREADS,
)


# ==================== 数据增强 ====================

class FormulaAugmentation:
    """
    公式图像数据增强

    注意调用时机：应在**原始分辨率**上做，之后才等比缩放到 IMG_HEIGHT。
    在 64px 高的图上旋转 ±3° 会把细笔画糊成灰带，反而损害识别。
    """

    def __init__(self, is_train=True):
        self.is_train = is_train

    def __call__(self, image):
        """
        Args:
            image: numpy array, shape (H, W) 灰度 或 (H, W, 3) RGB
        Returns:
            augmented image
        """
        if not self.is_train:
            return image

        # 随机旋转 ±3°
        if random.random() < 0.5:
            h, w = image.shape[:2]
            angle = random.uniform(-3, 3)
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            image = cv2.warpAffine(image, M, (w, h),
                                    borderMode=cv2.BORDER_REPLICATE)

        # 随机仿射（轻微缩放+平移）
        if random.random() < 0.3:
            h, w = image.shape[:2]
            scale = random.uniform(0.95, 1.05)
            tx = random.uniform(-w * 0.02, w * 0.02)
            ty = random.uniform(-h * 0.05, h * 0.05)
            M = np.float32([[scale, 0, tx], [0, scale, ty]])
            image = cv2.warpAffine(image, M, (w, h),
                                    borderMode=cv2.BORDER_REPLICATE)

        # 高斯噪声
        if random.random() < 0.3:
            noise = np.random.normal(0, 3, image.shape).astype(np.float32)
            image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        # 亮度抖动
        if random.random() < 0.3:
            factor = random.uniform(0.85, 1.15)
            image = np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)

        return image


# ==================== 图像预处理 ====================

def load_gray_image(image_path: str) -> np.ndarray:
    """
    读取公式图像并统一成「白底黑字」灰度图（保持原始分辨率）
    1. 读取图像（兼容中文路径）
    2. 转灰度
    3. 反色确保白底黑字

    单独拆出来是为了让数据增强能在**原始分辨率**上做：
    之前是先缩到 64px 再旋转/仿射，笔画会被糊掉。
    """
    # 中文路径兼容
    buf = np.fromfile(image_path, dtype=np.uint8)
    image = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"无法读取图像: {image_path}")

    # 转灰度
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 反色（确保白底黑字）—— 降采样判断，避免全像素扫描
    if np.mean(image[::8, ::8]) < 128:
        image = 255 - image

    return image


def resize_to_height(image: np.ndarray, target_height: int = IMG_HEIGHT,
                     max_width: int = None) -> np.ndarray:
    """
    等比缩放到指定高度（宽度按比例，不做压扁）

    **缩小时用 INTER_AREA**：公式笔画细，INTER_LINEAR 缩小时会严重混叠，
    笔画断裂或糊成灰带，直接影响识别。
    """
    h, w = image.shape[:2]
    if h != target_height:
        scale = target_height / h
        new_w = max(int(round(w * scale)), 1)
        # 放大用 INTER_LINEAR，缩小用 INTER_AREA
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        image = cv2.resize(image, (new_w, target_height), interpolation=interp)

    if max_width is not None and image.shape[1] > max_width:
        # 仅作为内存安全网；实测最大宽高比 16.38，IMG_MAX_WIDTH 已覆盖全部数据，
        # 正常情况下不会走到这里
        image = cv2.resize(image, (max_width, target_height),
                           interpolation=cv2.INTER_AREA)

    return image


def preprocess_image(image_path: str, target_height: int = IMG_HEIGHT,
                     max_width: int = IMG_MAX_WIDTH) -> np.ndarray:
    """
    加载并预处理公式图像（保持对外签名，推理脚本依赖）
    Returns:
        numpy array (H, W) uint8, 白底黑字，高度 = target_height
    """
    return resize_to_height(load_gray_image(image_path), target_height, max_width)



# ==================== 宽度分桶 ====================

def scan_image_widths(image_dir, names, target_height=IMG_HEIGHT,
                      max_width=IMG_MAX_WIDTH, num_threads=16):
    """
    并行扫描图像宽度，返回 {文件名: 缩放后宽度}

    只读文件头（`Image.open(...).size`），不解码像素，所以很快：
    8 线程实测 74502 张约 8 秒。

    为什么需要它：collate 会按 batch 内最大宽度补齐，而公式宽度差异极大，
    随机组批时大量样本被 padding 到远超自身长度，ViT 对这些空白 patch
    照样跑完整网络 —— 实测浪费 2.50 倍算力。
    """
    from concurrent.futures import ThreadPoolExecutor

    def _one(name):
        path = os.path.join(image_dir, name)
        try:
            with Image.open(path) as im:
                w, h = im.size
            if h <= 0:
                return name, None
            return name, max(int(round(w * target_height / h)), 1)
        except Exception:
            return name, None      # 读不到的样本用中位数兜底，不因个别坏图中断

    with ThreadPoolExecutor(max_workers=num_threads) as ex:
        results = dict(ex.map(_one, names))

    known = [v for v in results.values() if v]
    fallback = int(np.median(known)) if known else max_width
    return {k: min(v or fallback, max_width) for k, v in results.items()}


class BucketBatchSampler(Sampler):
    """
    按宽度分桶组批：样本按宽度排序后切块成批，再打乱批的顺序。

    组批策略：先按宽度排序，切成若干"超级桶"（每个含 N 个 batch），
    桶内打乱后切成 batch。这样既保证**批内宽度相近**（padding 少），
    又让每轮的样本组合有变化（不至于每轮都是完全相同的分组）。

    代价：批内样本相关性略增（同宽度的公式可能来自相似的题型），
    这是 OCR/ASR 领域的常规取舍，实测收益远大于影响。
    """

    def __init__(self, widths, batch_size, shuffle=True, drop_last=True,
                 batches_per_bucket=10, seed=0):
        self.widths = list(widths)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.bucket_size = max(batch_size * batches_per_bucket, batch_size)
        self.seed = seed
        self._iter_count = 0
        self._groups = self._make_groups()

    def _make_groups(self):
        """按宽度排序后切成超级桶"""
        order = sorted(range(len(self.widths)), key=lambda i: self.widths[i])
        return [order[i:i + self.bucket_size]
                for i in range(0, len(order), self.bucket_size)]

    def __iter__(self):
        # 每次迭代自增计数 —— DataLoader 每个 epoch 重新取一次迭代器，
        # 所以桶内打乱序列会自动随 epoch 变化，调用方无需手动 set_epoch
        self._iter_count += 1
        rng = random.Random(self.seed + self._iter_count)
        batches = []
        for group in self._groups:
            g = list(group)
            if self.shuffle:
                rng.shuffle(g)              # 桶内打乱 → 每轮组合不同
            for i in range(0, len(g), self.batch_size):
                b = g[i:i + self.batch_size]
                if len(b) < self.batch_size and self.drop_last:
                    continue
                batches.append(b)
        if self.shuffle:
            rng.shuffle(batches)            # 打乱批顺序，避免宽度单调变化
        return iter(batches)

    def __len__(self):
        total = 0
        for g in self._groups:
            n = len(g) // self.batch_size if self.drop_last \
                else -(-len(g) // self.batch_size)
            total += n
        return total


# ==================== Dataset ====================

class FormulaDataset(Dataset):
    """
    公式识别数据集
    标签文件格式: image_filename\tLaTeX token sequence (空格分隔)
    """

    def __init__(self, image_dir: str, labels_file: str, tokenizer,
                 transform=None, max_width: int = IMG_MAX_WIDTH,
                 precompute_widths=False):
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_width = max_width

        # 加载标签
        self.samples = []  # [(image_filename, latex_str)]
        with open(labels_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('\t', 1)
                if len(parts) < 2:
                    continue
                self.samples.append((parts[0].strip(), parts[1].strip()))

        # 预扫描宽度，供 BucketBatchSampler 按宽度分桶组批
        self.widths = None
        if precompute_widths:
            wmap = scan_image_widths(image_dir, [s[0] for s in self.samples],
                                     IMG_HEIGHT, max_width, WIDTH_SCAN_THREADS)
            self.widths = [wmap[s[0]] for s in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name, latex = self.samples[idx]
        img_path = os.path.join(self.image_dir, img_name)

        # 先读原图 → 在原分辨率上做增强 → 最后才等比缩放
        # （此前是「先缩到 64px 再增强」，旋转/仿射会把细笔画糊掉）
        image = load_gray_image(img_path)
        if self.transform is not None:
            image = self.transform(image)
        image = resize_to_height(image, IMG_HEIGHT, self.max_width)

        # encode LaTeX → token IDs
        token_ids = self.tokenizer.encode(latex, max_len=MAX_SEQ_LEN)

        # 转为 float tensor, 归一化到 [0, 1]
        image = image.astype(np.float32) / 255.0
        # 添加通道维度: (H, W) → (1, H, W)
        if len(image.shape) == 2:
            image = image[np.newaxis, :, :]
        else:
            image = image.transpose(2, 0, 1)  # (C, H, W)

        return {
            'image': torch.from_numpy(image),         # (1, H, W) float32
            'token_ids': torch.tensor(token_ids, dtype=torch.long),
            'latex': latex,
            'name': img_name,                          # 供评估按难度分档用
            'width': image.shape[2],                   # 原始宽度（pad 前）
        }


# ==================== Collate ====================

def collate_fn(batch):
    """
    动态宽度 padding
    将 batch 内图像 pad 到最大宽度
    将 token_ids pad 到最大长度
    """
    # 按宽度降序排列（方便 LSTM pack）
    batch = sorted(batch, key=lambda x: x['width'], reverse=True)

    images = [item['image'] for item in batch]
    token_ids_list = [item['token_ids'] for item in batch]
    latex_list = [item['latex'] for item in batch]
    name_list = [item['name'] for item in batch]
    widths = [item['width'] for item in batch]

    # Pad images 到 batch 内最大宽度
    max_w = max(img.shape[2] for img in images)
    padded_images = []
    for img in images:
        c, h, w = img.shape
        if w < max_w:
            pad = torch.full((c, h, max_w - w), 1.0, dtype=img.dtype)  # 白色=1.0
            img = torch.cat([img, pad], dim=2)
        padded_images.append(img)

    # Pad token_ids
    max_len = max(len(t) for t in token_ids_list)
    padded_tokens = []
    for t in token_ids_list:
        if len(t) < max_len:
            pad = torch.full((max_len - len(t),), PAD_TOKEN, dtype=t.dtype)
            t = torch.cat([t, pad])
        padded_tokens.append(t)

    return {
        'images': torch.stack(padded_images),     # (B, 1, H, max_W)
        'token_ids': torch.stack(padded_tokens),  # (B, max_len)
        'latex_list': latex_list,
        'name_list': name_list,
        'widths': widths,
    }


# ==================== DataLoader 工厂 ====================

def _resolve_tokenizer(tokenizer=None, vocab_path=None):
    """
    取词表：显式传入 > 指定 vocab_path > 报错

    词表现在跟着权重走（runs/v{N}/vocab.json），不再有共享的 runs/vocab.json，
    因此没有"默认路径"可回退 —— 拿错词表会让模型输出全错，比直接报错危险得多，
    所以这里宁可失败也不猜。
    """
    if tokenizer is not None:
        return tokenizer
    if not vocab_path:
        raise ValueError(
            "需要传入 tokenizer 或 vocab_path。\n"
            "  词表与权重同目录（train.py 会生成 runs/v{N}/vocab.json），"
            "不再有共享的 runs/vocab.json 可以回退。")
    from tokenizer import LaTeXTokenizer
    return LaTeXTokenizer.load(vocab_path)



def get_train_loader(batch_size=None, num_workers=None):
    """
    获取训练集 DataLoader（合并 CROHME + HME100K）

    词表由 train.py 存进**版本目录**（runs/v{N}/vocab.json），与权重放在一起。
    这里刻意不再写任何文件：加载器不该有写文件的副作用，而且原先会覆盖
    runs/vocab.json 这个共享文件，不同数据集配置之间会互相冲掉。
    """
    from tokenizer import LaTeXTokenizer

    # 构建词表（只从**训练集**标签统计，不能用测试集，否则是数据泄漏）
    label_files = [f for f in [CROHME_TRAIN_LABEL, HME100K_TRAIN_LABEL]
                   if os.path.exists(f)]
    tokenizer = LaTeXTokenizer()
    tokenizer.build_vocab(label_files)

    # 合并两个数据集（分桶需要宽度，所以要预扫描）
    from torch.utils.data import ConcatDataset
    datasets = []
    if os.path.exists(CROHME_TRAIN_LABEL):
        datasets.append(FormulaDataset(
            CROHME_TRAIN_IMG, CROHME_TRAIN_LABEL, tokenizer,
            transform=FormulaAugmentation(is_train=True),
            precompute_widths=BUCKET_BY_WIDTH))
    if os.path.exists(HME100K_TRAIN_LABEL):
        datasets.append(FormulaDataset(
            HME100K_TRAIN_IMG, HME100K_TRAIN_LABEL, tokenizer,
            transform=FormulaAugmentation(is_train=True),
            precompute_widths=BUCKET_BY_WIDTH))

    if not datasets:
        raise FileNotFoundError("未找到任何训练标签文件")

    dataset = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]
    bs = batch_size or BATCH_SIZE

    common = dict(
        num_workers=num_workers or NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
        prefetch_factor=PREFETCH_FACTOR,
    )

    if BUCKET_BY_WIDTH and all(getattr(d, 'widths', None) for d in datasets):
        widths = [w for d in datasets for w in d.widths]
        sampler = BucketBatchSampler(
            widths, bs, shuffle=True, drop_last=True,
            batches_per_bucket=BUCKET_BATCHES_PER_BUCKET)
        print(f"宽度分桶组批已启用: {len(sampler)} 个 batch/epoch "
              f"(桶大小 {sampler.bucket_size})")
        return DataLoader(
            dataset, batch_sampler=sampler,
            persistent_workers=PERSISTENT_WORKERS, **common), tokenizer

    # 回退：常规随机组批（padding 浪费更多，但分组更随机）
    return DataLoader(
        dataset, batch_size=bs, shuffle=True, drop_last=True,
        persistent_workers=PERSISTENT_WORKERS, **common), tokenizer


def _make_eval_loader(image_dir, label_file, tokenizer, batch_size, num_workers):
    """
    构造评估用 DataLoader

    同样启用宽度分桶：评估只是前向，但 padding 造成的浪费一样存在（实测 2.5 倍）。
    分桶只改变样本顺序，不影响任何指标（ExpRate/BLEU/编辑距离都与顺序无关）。
    """
    dataset = FormulaDataset(
        image_dir, label_file, tokenizer,
        transform=FormulaAugmentation(is_train=False),
        precompute_widths=BUCKET_BY_WIDTH)
    bs = batch_size or BATCH_SIZE

    if BUCKET_BY_WIDTH and dataset.widths:
        sampler = BucketBatchSampler(
            dataset.widths, bs, shuffle=False, drop_last=False,
            batches_per_bucket=BUCKET_BATCHES_PER_BUCKET)
        return DataLoader(dataset, batch_sampler=sampler,
                          num_workers=num_workers or NUM_WORKERS,
                          collate_fn=collate_fn, pin_memory=True)

    return DataLoader(dataset, batch_size=bs, shuffle=False,
                      num_workers=num_workers or NUM_WORKERS,
                      collate_fn=collate_fn, pin_memory=True)


def get_eval_loader(batch_size=None, num_workers=None, tokenizer=None,
                    vocab_path=None):
    """获取 CROHME 验证集 DataLoader"""
    tokenizer = _resolve_tokenizer(tokenizer, vocab_path)
    return _make_eval_loader(CROHME_EVAL_IMG, CROHME_EVAL_LABEL, tokenizer,
                             batch_size, num_workers)


def get_test_loader(batch_size=None, num_workers=None, tokenizer=None,
                    vocab_path=None):
    """获取 HME100K 测试集 DataLoader"""
    tokenizer = _resolve_tokenizer(tokenizer, vocab_path)
    return _make_eval_loader(HME100K_TEST_IMG, HME100K_TEST_LABEL, tokenizer,
                             batch_size, num_workers)


# ==================== 测试 ====================

if __name__ == "__main__":
    print("=" * 50)
    print("FormulaDataset 测试")
    print("=" * 50)

    loader, tokenizer = get_train_loader(batch_size=4, num_workers=0)
    print(f"词表大小: {tokenizer.vocab_size}")
    print(f"训练集大小: {len(loader.dataset)}")

    batch = next(iter(loader))
    print(f"\nBatch shapes:")
    print(f"  images:    {batch['images'].shape}")    # (B, 1, H, max_W)
    print(f"  token_ids: {batch['token_ids'].shape}") # (B, max_len)
    print(f"  widths:    {batch['widths']}")
    print(f"\n样本 LaTeX:")
    for i, latex in enumerate(batch['latex_list'][:3]):
        print(f"  [{i}] {latex[:80]}{'...' if len(latex) > 80 else ''}")

    # 测试 encode/decode
    ids = batch['token_ids'][0].tolist()
    decoded = tokenizer.decode(ids)
    print(f"\n解码还原: {decoded[:80]}{'...' if len(decoded) > 80 else ''}")
