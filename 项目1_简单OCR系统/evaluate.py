"""
评估脚本 - 用于测试和可视化
"""
import os
import argparse
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

from config import (
    DEVICE, VERSION, VERSION_DIR, IMG_DIR, BOX_DIR, IMG_HEIGHT, IMG_WIDTH,
    CHAR_TO_IDX, IDX_TO_CHAR
)
from dataset import get_data_loaders
from model import CRNN
from utils import decode_output, compute_accuracy, load_checkpoint


def load_model(model_path=None):
    """加载训练好的模型"""
    device = torch.device(DEVICE)
    model = CRNN().to(device)

    if model_path is None:
        model_path = os.path.join(VERSION_DIR, "best_model.pth")

    if os.path.exists(model_path):
        load_checkpoint(model, None, model_path)
    else:
        print(f"警告: 模型文件不存在 {model_path}，使用随机初始化模型")

    model.eval()
    return model, device


def predict_image(model, device, image_path):
    """
    对单张图像进行预测

    Args:
        model: CRNN模型
        device: 设备
        image_path: 图像路径

    Returns:
        预测的文本
    """
    # 使用imdecode加载图像（避免Windows中文路径问题）
    with open(image_path, 'rb') as f:
        img_bytes = np.frombuffer(f.read(), dtype=np.uint8)
        image = cv2.imdecode(img_bytes, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise FileNotFoundError(f"无法加载图像: {image_path}")

    # 预处理
    h, w = image.shape[:2]

    # 调整高度
    if h != IMG_HEIGHT:
        ratio = IMG_HEIGHT / h
        new_w = int(w * ratio)
        image = cv2.resize(image, (new_w, IMG_HEIGHT))

    # 调整宽度
    h, w = image.shape[:2]
    if w > IMG_WIDTH:
        image = image[:, :IMG_WIDTH]
    elif w < IMG_WIDTH:
        pad_w = IMG_WIDTH - w
        image = cv2.copyMakeBorder(image, 0, 0, 0, pad_w, cv2.BORDER_CONSTANT, value=0)

    # 归一化并转换为tensor
    image = image.astype(np.float32) / 255.0
    image = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    image = image.to(device)

    # 预测
    with torch.no_grad():
        output = model(image)

    # 解码
    pred_text = decode_output(output)[0]

    return pred_text


def visualize_predictions(model, device, num_samples=10):
    """可视化预测结果"""
    # 获取测试数据
    _, _, test_loader = get_data_loaders()

    # 获取样本
    images_list = []
    pred_texts = []
    gt_texts = []

    model.eval()
    with torch.no_grad():
        for batch_idx, (images, labels, label_lengths) in enumerate(test_loader):
            if batch_idx >= num_samples:
                break

            images = images.to(device)
            outputs = model(images)

            # 解码预测
            batch_pred_texts = decode_output(outputs)

            # 解码真实文本
            for i in range(labels.size(0)):
                gt_text = ''.join([chr(c) for c in labels[i, :label_lengths[i]].cpu().numpy()])
                gt_texts.append(gt_text)

            pred_texts.extend(batch_pred_texts)
            images_list.extend([img.cpu().numpy().squeeze() for img in images])

    # 绘制结果
    fig, axes = plt.subplots(num_samples, 1, figsize=(15, 3 * num_samples))
    if num_samples == 1:
        axes = [axes]

    for i in range(min(num_samples, len(images_list))):
        axes[i].imshow(images_list[i], cmap='gray')
        axes[i].set_title(f"预测: {pred_texts[i]}\n真实: {gt_texts[i]}",
                          color='green' if pred_texts[i] == gt_texts[i] else 'red')
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig('prediction_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("预测结果已保存到 prediction_results.png")


def evaluate_on_test_set(model, device):
    """在测试集上评估"""
    _, _, test_loader = get_data_loaders()

    all_pred_texts = []
    all_gt_texts = []

    model.eval()
    with torch.no_grad():
        for images, labels, label_lengths in test_loader:
            images = images.to(device)
            outputs = model(images)

            # 解码
            pred_texts = decode_output(outputs)
            all_pred_texts.extend(pred_texts)

            for i in range(labels.size(0)):
                gt_text = ''.join([chr(c) for c in labels[i, :label_lengths[i]].cpu().numpy()])
                all_gt_texts.append(gt_text)

    # 计算指标
    char_acc, word_acc = compute_accuracy(all_pred_texts, all_gt_texts)

    print(f"测试集结果:")
    print(f"  样本数: {len(all_gt_texts)}")
    print(f"  字符准确率: {char_acc:.4f}")
    print(f"  词准确率: {word_acc:.4f}")

    # 显示一些样本
    print("\n样本预测:")
    for i in range(min(10, len(all_gt_texts))):
        status = "✓" if all_pred_texts[i] == all_gt_texts[i] else "✗"
        print(f"  {status} 预测: {all_pred_texts[i][:50]:50s} | 真实: {all_gt_texts[i][:50]}")

    return char_acc, word_acc


def main():
    parser = argparse.ArgumentParser(description='OCR模型评估')
    parser.add_argument('--model', type=str, default=None, help='模型路径')
    parser.add_argument('--version', type=int, default=VERSION, help='模型版本号')
    parser.add_argument('--image', type=str, default=None, help='单张图像路径')
    parser.add_argument('--visualize', action='store_true', help='可视化预测结果')
    parser.add_argument('--test', action='store_true', help='在测试集上评估')

    args = parser.parse_args()

    # 加载模型
    model, device = load_model(args.model)

    if args.image:
        # 单张图像预测
        pred_text = predict_image(model, device, args.image)
        print(f"预测结果: {pred_text}")

    elif args.visualize:
        # 可视化
        visualize_predictions(model, device)

    elif args.test:
        # 测试集评估
        evaluate_on_test_set(model, device)

    else:
        # 默认进行测试集评估
        evaluate_on_test_set(model, device)


if __name__ == "__main__":
    main()
