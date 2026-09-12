"""
工具函数
"""
import os
import torch
import editdistance
from config import IDX_TO_CHAR, MODEL_DIR


def decode_output(output, label_lengths=None):
    """
    解码CTC输出为文本

    Args:
        output: 模型输出 (batch, seq_len, num_classes)
        label_lengths: 标签长度（可选，用于批量解码）

    Returns:
        解码后的文本列表
    """
    # 贪婪解码
    _, max_indices = output.max(dim=2)  # (batch, seq_len)

    texts = []
    for i in range(max_indices.size(0)):
        indices = max_indices[i].cpu().numpy()

        # 去除重复字符和blank
        text = []
        prev_idx = -1
        for idx in indices:
            if idx != prev_idx:
                if idx != 0:  # 0 是 CTC blank
                    text.append(IDX_TO_CHAR.get(idx, ''))
            prev_idx = idx

        texts.append(''.join(text))

    return texts


def compute_accuracy(pred_texts, gt_texts):
    """
    计算字符准确率和词准确率

    Args:
        pred_texts: 预测文本列表
        gt_texts: 真实文本列表

    Returns:
        char_accuracy: 字符准确率
        word_accuracy: 词准确率
    """
    correct_chars = 0
    total_chars = 0
    correct_words = 0
    total_words = len(gt_texts)

    for pred, gt in zip(pred_texts, gt_texts):
        # 字符准确率
        min_len = min(len(pred), len(gt))
        correct_chars += sum(1 for i in range(min_len) if pred[i] == gt[i])
        total_chars += max(len(pred), len(gt))

        # 词准确率
        if pred == gt:
            correct_words += 1

    char_accuracy = correct_chars / total_chars if total_chars > 0 else 0
    word_accuracy = correct_words / total_words if total_words > 0 else 0

    return char_accuracy, word_accuracy


def compute_edit_distance(pred_texts, gt_texts):
    """
    计算平均编辑距离

    Args:
        pred_texts: 预测文本列表
        gt_texts: 真实文本列表

    Returns:
        avg_distance: 平均编辑距离
    """
    total_distance = 0
    for pred, gt in zip(pred_texts, gt_texts):
        total_distance += editdistance.eval(pred, gt)

    avg_distance = total_distance / len(gt_texts) if gt_texts else 0
    return avg_distance


def save_checkpoint(model, optimizer, epoch, best_acc, filepath):
    """保存模型检查点"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_acc': best_acc,
    }, filepath)


def load_checkpoint(model, optimizer, filepath):
    """加载模型检查点"""
    if os.path.exists(filepath):
        checkpoint = torch.load(filepath, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint.get('epoch', 0)
        best_acc = checkpoint.get('best_acc', 0)
        return epoch, best_acc
    return 0, 0.0


def load_checkpoint(model, optimizer, filepath):
    """加载模型检查点"""
    if os.path.exists(filepath):
        checkpoint = torch.load(filepath, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        best_acc = checkpoint['best_acc']
        print(f"已加载模型: {filepath}, epoch={epoch}, best_acc={best_acc:.4f}")
        return epoch, best_acc
    else:
        print(f"模型文件不存在: {filepath}")
        return 0, 0


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
