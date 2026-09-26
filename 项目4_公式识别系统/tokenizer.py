"""
LaTeX Tokenizer
正则分词 + 词表构建 + encode/decode
标签格式: "filename\\tLaTeX tokens (空格分隔)"
"""
import re
import json
import os
from collections import Counter
from typing import List, Optional

from config import SPECIAL_TOKENS, PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN, MIN_FREQ


# ==================== 正则分词 ====================

# 匹配顺序: LaTeX命令(\frac) | 结构符号({}[]^_) | 字母数字 | 其他单字符
_TOKENIZE_RE = re.compile(
    r'\\[a-zA-Z]+'        # LaTeX 命令: \frac, \sqrt, \alpha
    r'|[{}()\[\]^_]'      # 结构符号
    r'|[a-zA-Z0-9]'       # 单个字母或数字
    r'|[^\\a-zA-Z0-9{}()\[\]^_\s]'  # 其他符号: +, =, <, >, %, etc.
)


def tokenize_latex(latex: str) -> List[str]:
    """将 LaTeX 字符串分词为 token 列表"""
    return _TOKENIZE_RE.findall(latex)


# ==================== Tokenizer 类 ====================

class LaTeXTokenizer:
    """
    LaTeX 分词器
    - 从训练标签构建词表
    - encode: LaTeX 字符串 → token ID 序列（含 SOS/EOS）
    - decode: token ID 序列 → LaTeX 字符串（去除 SOS/EOS/PAD）
    """

    def __init__(self):
        self.token2id = {}
        self.id2token = {}
        self.vocab_size = 0

    def build_vocab(self, label_files: List[str], min_freq: int = MIN_FREQ):
        """从多个标签文件构建词表"""
        counter = Counter()
        for fpath in label_files:
            with open(fpath, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split('\t', 1)
                    if len(parts) < 2:
                        continue
                    tokens = tokenize_latex(parts[1].strip())
                    counter.update(tokens)

        # 特殊 token
        self.token2id = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
        # 按频率降序添加
        for token, freq in counter.most_common():
            if freq >= min_freq and token not in self.token2id:
                self.token2id[token] = len(self.token2id)

        self.id2token = {v: k for k, v in self.token2id.items()}
        self.vocab_size = len(self.token2id)
        return self

    def encode(self, latex: str, max_len: Optional[int] = None) -> List[int]:
        """LaTeX 字符串 → token ID 列表（含 SOS/EOS）"""
        tokens = tokenize_latex(latex)
        ids = [SOS_TOKEN]
        for tok in tokens:
            ids.append(self.token2id.get(tok, UNK_TOKEN))
        ids.append(EOS_TOKEN)
        if max_len is not None and len(ids) > max_len:
            # 截断时保留 EOS
            ids = ids[:max_len - 1] + [EOS_TOKEN]
        return ids

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        """token ID 列表 → LaTeX 字符串"""
        tokens = []
        for idx in ids:
            if idx == EOS_TOKEN:
                break
            tok = self.id2token.get(idx, "<UNK>")
            if skip_special and tok in SPECIAL_TOKENS:
                continue
            if tok:  # 跳过空字符串
                tokens.append(tok)
        return " ".join(tokens)

    def save(self, path: str):
        """保存词表到 JSON"""
        data = {
            'token2id': self.token2id,
            'vocab_size': self.vocab_size,
        }
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> 'LaTeXTokenizer':
        """从 JSON 加载词表"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            raise RuntimeError(f"加载词表失败 {path}: {e}") from e
        tok = cls()
        tok.token2id = data['token2id']
        tok.id2token = {v: k for k, v in tok.token2id.items()}
        tok.vocab_size = data['vocab_size']
        assert tok.vocab_size == len(tok.token2id), \
            f"vocab_size({tok.vocab_size}) 与实际词表大小({len(tok.token2id)}) 不一致"
        return tok


# ==================== 测试 ====================

if __name__ == "__main__":
    # 只统计**训练集**标签：把测试集也算进词表是数据泄漏
    from config import CROHME_TRAIN_LABEL, HME100K_TRAIN_LABEL

    label_files = [f for f in [CROHME_TRAIN_LABEL, HME100K_TRAIN_LABEL]
                   if os.path.exists(f)]
    print(f"找到 {len(label_files)} 个训练标签文件")

    tokenizer = LaTeXTokenizer()
    tokenizer.build_vocab(label_files, min_freq=MIN_FREQ)
    print(f"词表大小: {tokenizer.vocab_size}")

    # 测试 encode/decode 往返
    test_cases = [
        "x + 1 = 2",
        r"\frac { a } { b }",
        r"\sqrt { x ^ { 2 } + y ^ { 2 } }",
        r"\int _ { 0 } ^ { 1 } f ( x ) d x",
    ]
    print("\n--- encode/decode 往返测试 ---")
    for latex in test_cases:
        ids = tokenizer.encode(latex, max_len=50)
        decoded = tokenizer.decode(ids)
        match = decoded.strip() == latex.strip()
        ok = "OK" if match else "FAIL"
        print(f"[{ok}] {latex}")
        print(f"  ids:     {ids}")
        print(f"  decoded: {decoded}")

    # 这里刻意**不保存**词表。
    # 词表由 train.py 生成到 runs/v{N}/vocab.json（与权重同目录），
    # 词表必须与权重一一配对；单独存一份到别处只会制造"来源不明的词表"，
    # 拿错词表不会报错、只会让输出全错。
    print("\n[OK] 以上仅为分词器自测。词表由 train.py 生成到 runs/v{N}/vocab.json，"
          "此处不保存。")
