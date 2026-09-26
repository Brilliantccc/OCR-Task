"""
公式识别模型

支持两套架构（config.MODEL_ARCH 切换）:
  vit_transformer : ViT-B/16 编码器 + Transformer 解码器（默认）
  cnn_lstm        : ResNet-34 + Bahdanau Attention + LSTM（基线，用于 A/B 对比）

两套架构对外暴露完全相同的接口，train / evaluate / inference 无需区分。
"""
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from config import (
    MODEL_ARCH, PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, MAX_DECODE_LEN, BEAM_WIDTH,
    BEAM_LEN_PENALTY, EMBED_DIM, DECODER_DIM, ATTN_DIM, ENCODER_DIM, DROPOUT,
    NUM_DECODER_LAYERS, VIT_PATCH_SIZE, VIT_HIDDEN, VIT_LAYERS, VIT_HEADS,
    D_MODEL, NUM_HEADS, FFN_DIM, TRANSFORMER_LAYERS, TRANSFORMER_DROPOUT,
    IMG_HEIGHT,
)


# ==================== 公共：2D sin-cos 位置编码 ====================

def build_2d_sincos_pos_embed(dim, rows, cols, device=None, dtype=torch.float32):
    """
    2D sin-cos 位置编码，**分辨率无关** —— 任意 (rows, cols) 网格都能现场生成。
    这正是自写 ViT 而不直接用 torchvision ViT 的原因：后者的位置编码是
    绑定 14x14=196 个 patch 的学习式参数，无法适应 4x50 这种扁平的公式网格。

    Args:
        dim: 编码维度，需能被 4 整除
        rows, cols: patch 网格的行列数
    Returns:
        (rows*cols, dim)，行优先（与 x.flatten(2) 的顺序一致）
    """
    assert dim % 4 == 0, f'位置编码维度需能被 4 整除，当前 {dim}'
    quarter = dim // 4

    # 每个方向用 quarter 个频率，sin/cos 各占一半 → 每方向 dim/2，两方向合计 dim
    omega = 1.0 / (10000 ** (torch.arange(quarter, dtype=torch.float32,
                                          device=device) / quarter))
    grid_h = torch.arange(rows, dtype=torch.float32, device=device)
    grid_w = torch.arange(cols, dtype=torch.float32, device=device)

    out_h = torch.einsum('m,d->md', grid_h, omega)      # (rows, quarter)
    out_w = torch.einsum('m,d->md', grid_w, omega)      # (cols, quarter)

    emb_h = torch.cat([torch.sin(out_h), torch.cos(out_h)], dim=1)  # (rows, dim/2)
    emb_w = torch.cat([torch.sin(out_w), torch.cos(out_w)], dim=1)  # (cols, dim/2)

    # 展开成行优先的 (rows*cols, dim)
    emb_h = emb_h.repeat_interleave(cols, dim=0)        # (rows*cols, dim/2)
    emb_w = emb_w.repeat(rows, 1)                       # (rows*cols, dim/2)

    return torch.cat([emb_h, emb_w], dim=1).to(dtype)


# ==================== ViT 编码器 ====================

class ViTBlock(nn.Module):
    """
    ViT 编码块，**子模块命名与 torchvision 的 EncoderBlock 完全一致**
    （ln_1 / self_attention / ln_2 / mlp.{0,3}），这样才能把预训练权重
    整块 load_state_dict 进来，同时又保有我们自己稳定的命名
    （不会把 torchvision 版本相关的 encoder_layer_0 之类名字写进 checkpoint）。
    """

    def __init__(self, dim=VIT_HIDDEN, num_heads=VIT_HEADS,
                 mlp_dim=VIT_HIDDEN * 4, dropout=TRANSFORMER_DROPOUT,
                 attn_dropout=0.0):
        super().__init__()
        self.ln_1 = nn.LayerNorm(dim, eps=1e-6)
        # 注意力 dropout 取 0.0，与预训练一致
        self.self_attention = nn.MultiheadAttention(
            dim, num_heads, dropout=attn_dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.ln_2 = nn.LayerNorm(dim, eps=1e-6)
        # 用 Sequential 以复现 torchvision 的 mlp.0 / mlp.3 键名
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        h = self.ln_1(x)
        h = self.self_attention(h, h, h, need_weights=False)[0]
        x = x + self.dropout(h)
        x = x + self.dropout(self.mlp(self.ln_2(x)))
        return x


class FormulaViTEncoder(nn.Module):
    """
    ViT-B/16 编码器，支持**任意宽度**的公式图像

    输入: (B, 1, H, W) 灰度 → 复制成 3 通道 → patch=16 → 网格 (H/16) x (W/16)
    输出: (B, rows*cols, VIT_HIDDEN)

    位置编码用现场生成的 2D sin-cos（见 build_2d_sincos_pos_embed），
    因此不受输入尺寸限制；预训练的 patch 投影与 12 层 Transformer 块
    从 torchvision vit_b_16 逐块拷贝。
    """

    def __init__(self, pretrained=True, patch_size=VIT_PATCH_SIZE,
                 hidden=VIT_HIDDEN, num_layers=VIT_LAYERS, num_heads=VIT_HEADS,
                 dropout=TRANSFORMER_DROPOUT):
        super().__init__()
        self.patch_size = patch_size
        self.hidden = hidden

        self.patch_embed = nn.Conv2d(3, hidden, kernel_size=patch_size,
                                     stride=patch_size)
        self.blocks = nn.Sequential(*[
            ViTBlock(hidden, num_heads, hidden * 4, dropout)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden, eps=1e-6)

        # 位置编码缓存：同一形状只算一次
        self._pos_cache = {}

        self.pretrain_report = None
        if pretrained:
            self.load_pretrained()

    # ---------- 预训练权重 ----------

    def load_pretrained(self):
        """
        从 torchvision vit_b_16 逐块拷入预训练权重。

        不用 load_state_dict(strict=True) 整体加载，因为：
          1) 我们要丢掉 class_token / 学习式 pos_embedding / 分类头
          2) torchvision 内部命名跨版本有差异（本地 0.28 / AutoDL 0.16），
             整体严格加载会在版本不匹配时直接崩
        逐块按子模块名搬移，并对每块单独 try —— 某块对不上只影响那一块。
        """
        try:
            tv = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        except Exception as e:
            self.pretrain_report = {'loaded': False, 'reason': f'{type(e).__name__}: {e}'}
            return

        ok, fail = 0, 0
        try:
            tv_blocks = list(tv.encoder.layers)
        except AttributeError as e:
            self.pretrain_report = {'loaded': False, 'reason': f'encoder.layers 缺失: {e}'}
            return

        # patch 投影
        try:
            self.patch_embed.load_state_dict(tv.conv_proj.state_dict())
            ok += 1
        except Exception:
            fail += 1

        # 逐块搬
        for mine, theirs in zip(self.blocks, tv_blocks):
            try:
                mine.load_state_dict(theirs.state_dict())
                ok += 1
            except Exception:
                fail += 1

        # 末端 LayerNorm
        try:
            self.norm.load_state_dict(tv.encoder.ln.state_dict())
            ok += 1
        except Exception:
            fail += 1

        self.pretrain_report = {
            'loaded': fail == 0,
            'copied': ok,
            'failed': fail,
            'total': len(tv_blocks) + 2,
        }

    def pretrain_summary(self):
        """返回一行人类可读的预训练加载结果，供 train.py 打印"""
        r = self.pretrain_report
        if r is None:
            return '未加载预训练权重'
        if not r.get('loaded'):
            return f"预训练权重加载失败: {r.get('reason', '部分模块未命中')}"
        if r.get('failed'):
            return (f"预训练权重部分命中: {r['copied']}/{r['total']} "
                    f"（{r['failed']} 个模块退化为随机初始化）")
        return f"预训练权重全部命中: {r['copied']}/{r['total']}"

    # ---------- 前向 ----------

    @property
    def out_dim(self):
        return self.hidden

    def _pos_embed(self, rows, cols, dtype, device):
        key = (rows, cols)
        if key not in self._pos_cache:
            self._pos_cache[key] = build_2d_sincos_pos_embed(
                self.hidden, rows, cols, device=device, dtype=torch.float32)
        return self._pos_cache[key].to(device=device, dtype=dtype)

    def forward(self, x):
        """
        Args:
            x: (B, 1, H, W) 灰度，或 (B, 3, H, W)
        Returns:
            (B, rows*cols, hidden)
        """
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)        # 灰度复制成 3 通道，复用 ImageNet 预训练

        x = self.patch_embed(x)             # (B, hidden, rows, cols)
        B, D, rows, cols = x.shape
        x = x.flatten(2).transpose(1, 2)    # (B, rows*cols, hidden)

        x = x + self._pos_embed(rows, cols, x.dtype, x.device)
        x = self.dropout(x)
        x = self.blocks(x)
        x = self.norm(x)
        return x


# ==================== CNN 编码器（cnn_lstm 基线） ====================

class FormulaEncoder(nn.Module):
    """
    基于 ResNet-34 的公式图像编码器
    输入: (B, 1, H, W) 灰度公式图像
    输出: (B, T_enc, 512) 特征序列

    注意: adaptive_pool 把高度压成 1，会抹掉上标/下标的垂直差异
    （x^2 与 x_2 字形相同、只差垂直位置）。这是被保留的基线行为，
    新架构请用 FormulaViTEncoder。
    """

    def __init__(self, pretrained=True):
        super().__init__()
        resnet = models.resnet34(
            weights=models.ResNet34_Weights.DEFAULT if pretrained else None
        )

        # 修改第一层：灰度图单通道输入（复制预训练权重的均值）
        old_conv1 = resnet.conv1
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if pretrained:
            self.conv1.weight.data = old_conv1.weight.data.mean(dim=1, keepdim=True)

        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1   # 64 通道
        self.layer2 = resnet.layer2   # 128 通道
        self.layer3 = resnet.layer3   # 256 通道
        self.layer4 = resnet.layer4   # 512 通道

        # 高度压缩为1，保留宽度方向序列信息
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, None))

    @property
    def out_dim(self):
        return 512

    def forward(self, x):
        """
        Args:
            x: (B, 1, H, W)
        Returns:
            (B, T_enc, 512) — 宽度方向作为序列维度
        """
        x = self.conv1(x)       # (B, 64, H/2, W/2)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)     # (B, 64, H/4, W/4)
        x = self.layer1(x)      # (B, 64, H/4, W/4)
        x = self.layer2(x)      # (B, 128, H/8, W/8)
        x = self.layer3(x)      # (B, 256, H/16, W/16)
        x = self.layer4(x)      # (B, 512, H/32, W/32)
        x = self.adaptive_pool(x)  # (B, 512, 1, W')
        x = x.squeeze(2)           # (B, 512, W')
        x = x.permute(0, 2, 1)     # (B, W', 512)
        return x


# ==================== Attention（cnn_lstm 基线） ====================

class BahdanauAttention(nn.Module):
    """
    Bahdanau (加性) Attention
    encoder_output: (B, T_enc, enc_dim)
    decoder_hidden: (B, dec_dim)
    输出: context (B, enc_dim), weights (B, T_enc)

    投影 W_enc(encoder_output) 只依赖编码器输出、与 decoder hidden 无关，
    因此解码循环里可用 project_encoder() 预先算一次复用，避免每步重算。
    """

    def __init__(self, enc_dim=ENCODER_DIM, dec_dim=DECODER_DIM, attn_dim=ATTN_DIM):
        super().__init__()
        self.W_enc = nn.Linear(enc_dim, attn_dim, bias=False)
        self.W_dec = nn.Linear(dec_dim, attn_dim, bias=False)
        self.V = nn.Linear(attn_dim, 1, bias=False)

    def project_encoder(self, encoder_output):
        """预计算编码器投影，可在解码循环外调用一次"""
        return self.W_enc(encoder_output)

    def forward(self, encoder_output, decoder_hidden, enc_proj=None):
        # encoder_output: (B, T_enc, enc_dim)
        # decoder_hidden: (B, dec_dim)
        if enc_proj is None:
            enc_proj = self.project_encoder(encoder_output)    # (B, T_enc, attn_dim)
        dec_proj = self.W_dec(decoder_hidden).unsqueeze(1)     # (B, 1, attn_dim)
        scores = self.V(torch.tanh(enc_proj + dec_proj))       # (B, T_enc, 1)
        scores = scores.squeeze(-1)                             # (B, T_enc)
        weights = F.softmax(scores, dim=-1)                     # (B, T_enc)
        context = torch.bmm(weights.unsqueeze(1), encoder_output)  # (B, 1, enc_dim)
        context = context.squeeze(1)                             # (B, enc_dim)
        return context, weights


# ==================== LSTM 解码器（cnn_lstm 基线） ====================

class FormulaDecoder(nn.Module):
    """
    LSTM + Bahdanau Attention 解码器
    """

    def __init__(self, vocab_size, embed_dim=EMBED_DIM,
                 enc_dim=ENCODER_DIM, dec_dim=DECODER_DIM,
                 attn_dim=ATTN_DIM, dropout=DROPOUT,
                 num_layers=NUM_DECODER_LAYERS):
        super().__init__()
        self.vocab_size = vocab_size
        self.dec_dim = dec_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_TOKEN)
        self.attention = BahdanauAttention(enc_dim, dec_dim, attn_dim)
        self.dropout = nn.Dropout(dropout)

        # LSTM 输入: embedding + context
        # 注意 num_layers 此前是死配置（config 定义了但从未传进来）
        self.lstm = nn.LSTM(embed_dim + enc_dim, dec_dim,
                            num_layers=num_layers, batch_first=True)

        # 输出层: hidden + context + embedding → vocab
        self.fc_out = nn.Linear(dec_dim + enc_dim + embed_dim, vocab_size)

    def forward_step(self, token, hidden, cell, encoder_output, enc_proj=None):
        """
        单步解码
        Args:
            token: (B,) 当前输入 token ID
            hidden: (num_layers, B, dec_dim) LSTM hidden
            cell: (num_layers, B, dec_dim) LSTM cell
            encoder_output: (B, T_enc, enc_dim)
            enc_proj: 预计算的 W_enc(encoder_output)，避免每步重算
        Returns:
            logits: (B, vocab_size)
            hidden, cell: 更新后的 LSTM 状态
            attn_weights: (B, T_enc)
        """
        embedded = self.dropout(self.embedding(token))         # (B, embed_dim)
        context, attn_weights = self.attention(
            encoder_output, hidden[-1], enc_proj=enc_proj)     # (B, enc_dim)

        lstm_input = torch.cat([embedded, context], dim=-1).unsqueeze(1)  # (B, 1, embed+enc)
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        output = output.squeeze(1)                              # (B, dec_dim)

        pred = self.fc_out(torch.cat([output, context, embedded], dim=-1))  # (B, vocab_size)
        return pred, hidden, cell, attn_weights


# ==================== Transformer 解码器（vit_transformer） ====================

class FormulaTransformerDecoder(nn.Module):
    """
    标准 Transformer 解码器（自注意力 + 交叉注意力）

    与 LSTM 解码器的关键差异: 训练时是**单次并行前向 + 因果掩码**，
    不存在逐步喂入，因此 Teacher Forcing 调度对新架构完全无效。
    """

    def __init__(self, vocab_size, d_model=D_MODEL, num_heads=NUM_HEADS,
                 ffn_dim=FFN_DIM, num_layers=TRANSFORMER_LAYERS,
                 dropout=TRANSFORMER_DROPOUT, enc_dim=VIT_HIDDEN,
                 max_len=MAX_DECODE_LEN):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.dec_dim = d_model          # 兼容旧代码对 decoder.dec_dim 的读取
        self.max_len = max_len

        # 编码器输出维度（768）→ 解码器维度（512）
        self.enc_proj = nn.Linear(enc_dim, d_model) if enc_dim != d_model else nn.Identity()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_TOKEN)
        # +8 留余量：贪心解码时生成序列会比 max_len 多走一格
        self.pos_embed = nn.Embedding(max_len + 8, d_model)
        self.dropout = nn.Dropout(dropout)

        layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=ffn_dim,
            dropout=dropout, activation='relu',
            batch_first=True,          # 默认是 False，必须显式设 True
            norm_first=False,
        )
        self.transformer = nn.TransformerDecoder(
            layer, num_layers=num_layers, norm=nn.LayerNorm(d_model))
        self.fc_out = nn.Linear(d_model, vocab_size)

        self._causal_cache = {}

    def _causal_mask(self, size, device):
        """上三角为 True 的因果掩码（屏蔽未来位置）"""
        key = (size, str(device))
        if key not in self._causal_cache:
            self._causal_cache[key] = torch.triu(
                torch.ones(size, size, dtype=torch.bool, device=device), diagonal=1)
        return self._causal_cache[key]

    def forward(self, tgt_ids, memory, tgt_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        """
        Args:
            tgt_ids: (B, L) 解码器输入 token id
            memory: (B, N, enc_dim) 编码器输出
            memory_key_padding_mask: (B, N) True 表示该位置是 padding，需忽略
        Returns:
            logits: (B, L, vocab_size)
        """
        memory = self.enc_proj(memory)
        L = tgt_ids.size(1)
        x = self.embedding(tgt_ids)
        x = x + self.pos_embed(
            torch.arange(L, device=tgt_ids.device).unsqueeze(0))
        x = self.dropout(x)

        if tgt_mask is None:
            tgt_mask = self._causal_mask(L, tgt_ids.device)

        h = self.transformer(
            x, memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return self.fc_out(h)


# ==================== 完整模型 ====================

class FormulaRecognizer(nn.Module):
    """
    Encoder-Decoder 公式识别模型（双架构门面）

    对外契约（两套架构完全一致）:
      - forward(images, token_ids=None, teacher_forcing_ratio=0.5, widths=None)
          训练模式 + token_ids  → (B, T, vocab_size) logits，第 j 列预测 targets[:, j]
          其余情况              → (B, L) token id，首 token 非 SOS、EOS 收尾
      - beam_search(images, beam_width=None, max_len=None) → (1, seq_len)
      - 保留 self.encoder / self.decoder 两个子模块（分层学习率依赖）
    """

    def __init__(self, vocab_size, arch=None, embed_dim=EMBED_DIM,
                 dec_dim=DECODER_DIM, dropout=DROPOUT):
        super().__init__()
        self.arch = arch or MODEL_ARCH
        self.vocab_size = vocab_size

        if self.arch == 'vit_transformer':
            self.encoder = FormulaViTEncoder(pretrained=True)
            self.decoder = FormulaTransformerDecoder(
                vocab_size=vocab_size, enc_dim=self.encoder.out_dim)
        elif self.arch == 'cnn_lstm':
            self.encoder = FormulaEncoder(pretrained=True)
            self.decoder = FormulaDecoder(
                vocab_size=vocab_size, embed_dim=embed_dim,
                enc_dim=self.encoder.out_dim, dec_dim=dec_dim, dropout=dropout)
        else:
            raise ValueError(
                f"未知架构 '{self.arch}'，可选: 'vit_transformer' | 'cnn_lstm'")

    # ---------- 编码 ----------

    @property
    def is_transformer(self):
        return self.arch == 'vit_transformer'

    def _patch_padding_mask(self, widths, cols):
        """
        由图像像素宽度推出 patch 级 padding mask。
        collate 是右侧补白到 batch 内最大宽度，这些补白 patch 不应参与交叉注意力。

        Args:
            widths: (B,) 每张图的**有效**像素宽度
            cols: patch 网格的列数
        Returns:
            (B, rows*cols) bool，True 表示该 patch 是 padding
        """
        patch = self.encoder.patch_size
        n_rows = IMG_HEIGHT // patch

        # 有效 patch 列数（向上取整：最后一块只覆盖到一半也算有效）
        valid_cols = torch.clamp((widths + patch - 1) // patch, max=cols)
        col_idx = torch.arange(cols, device=widths.device).unsqueeze(0)   # (1, cols)
        col_pad = col_idx >= valid_cols.unsqueeze(1)                      # (B, cols)

        # 展开成行优先，与 encoder 里 x.flatten(2) 的顺序保持一致
        return col_pad.unsqueeze(1).repeat(1, n_rows, 1).reshape(
            -1, n_rows * cols)

    def encode(self, images, widths=None):
        """
        编码图像
        Returns:
            memory: (B, N, enc_dim)
            memory_key_padding_mask: (B, N) 或 None
        """
        if not self.is_transformer:
            return self.encoder(images), None

        memory = self.encoder(images)
        mask = None
        if widths is not None:
            cols = memory.size(1) // (IMG_HEIGHT // self.encoder.patch_size)
            mask = self._patch_padding_mask(
                torch.as_tensor(widths, device=memory.device), cols)
        return memory, mask

    # ---------- 前向 ----------

    def forward(self, images, token_ids=None, teacher_forcing_ratio=0.5, widths=None):
        """
        推理前向（以及 cnn_lstm 的训练前向）

        ⚠️ vit_transformer 的训练前向请用 training_step() —— 它的张量对齐与
        贪心解码严格一致，而 forward 的训练分支沿用 LSTM 的旧约定，
        两套架构混用会静默错位（训练 loss 正常但 ExpRate 为 0）。

        Args:
            images: (B, 1, H, W)
            token_ids: (B, T) 训练标签（含 SOS 开头、EOS 结尾）
            teacher_forcing_ratio: 仅 cnn_lstm 生效
            widths: (B,) 每张图的有效像素宽度，用于生成 padding mask（可省）
        Returns:
            训练（仅 cnn_lstm）: (B, T, vocab_size)，第 j 列预测 targets[:, j]
            推理: (B, L) token id
        """
        memory, mem_mask = self.encode(images, widths)

        if self.training and token_ids is not None:
            if self.is_transformer:
                raise RuntimeError(
                    "vit_transformer 的训练前向请调用 training_step()，"
                    "它返回按架构对齐的 (logits, targets)。\n"
                    "  forward() 的训练分支只保留给 cnn_lstm 基线。")
            return self._train_decode(memory, token_ids, teacher_forcing_ratio)

        if self.is_transformer:
            return self._greedy_decode_transformer(memory, mem_mask)
        return self._greedy_decode(memory)

    # ---------- 训练解码 ----------

    def training_step(self, images, targets, teacher_forcing_ratio=1.0, widths=None):
        """
        训练前向：返回**已按架构对齐**的 (logits_flat, targets_flat)，
        可直接送 CrossEntropyLoss。把对齐放在模型内部，是因为两套架构的
        自然对齐方式本质不同，硬凑成一个契约必然出错。

        - vit_transformer（并行前向 + 因果掩码）:
              解码器输入 D = targets[:, :-1]，因果掩码下 output[j] 只看到
              targets[0..j]，预测 targets[j+1]。这是标准 seq2seq 对齐，
              与贪心/beam 解码的上下文**逐位一致**（贪心第一步看到 [SOS]，
              而 output[0] 也恰好只看到 [SOS]）。
          ⚠️ 曾经写错成 D = [SOS] + targets[:, :-1]，让 output[j] 预测 targets[j]，
              看似满足旧契约，实则上下文比解码时多一个 SOS，导致训练 loss 极低
              （0.03）但推理 ExpRate 为 0。改用标准对齐后两者一致。

        - cnn_lstm（逐步解码）: 输出第 j 列预测 targets[:, j]，第 0 列是占位零。
        """
        memory, mem_mask = self.encode(images, widths)

        if self.is_transformer:
            dec_in = targets[:, :-1]                     # (B, T-1)
            logits = self.decoder(
                dec_in, memory,
                tgt_mask=self.decoder._causal_mask(dec_in.size(1), targets.device),
                tgt_key_padding_mask=(dec_in == PAD_TOKEN),
                memory_key_padding_mask=mem_mask,
            )                                            # (B, T-1, vocab)
            return (logits.reshape(-1, self.vocab_size),
                    targets[:, 1:].reshape(-1))

        logits = self._train_decode(memory, targets, teacher_forcing_ratio)
        return (logits[:, 1:].reshape(-1, self.vocab_size),
                targets[:, 1:].reshape(-1))

    def _train_decode(self, encoder_output, targets, tf_ratio):
        """训练模式（LSTM）：Teacher Forcing"""
        B, T = targets.shape
        device = encoder_output.device

        hidden, cell = self._init_decoder_state(encoder_output)
        enc_proj = self.decoder.attention.project_encoder(encoder_output)
        outputs = []

        input_token = targets[:, 0]  # <SOS>

        for t in range(1, T):
            logits, hidden, cell, _ = self.decoder.forward_step(
                input_token, hidden, cell, encoder_output, enc_proj=enc_proj)
            outputs.append(logits)

            # Teacher Forcing
            if random.random() < tf_ratio:
                input_token = targets[:, t]
            else:
                input_token = logits.argmax(dim=-1)

        # (B, T-1, vocab_size) → 补齐第一列 SOS 位置为零
        sos_pad = torch.zeros(B, 1, self.vocab_size, device=device)
        return torch.cat([sos_pad, torch.stack(outputs, dim=1)], dim=1)

    def _init_decoder_state(self, encoder_output):
        """用 encoder 输出的均值初始化 decoder hidden state"""
        B = encoder_output.size(0)
        dec_dim = self.decoder.dec_dim
        num_layers = getattr(self.decoder, 'num_layers', 1)

        mean_enc = encoder_output.mean(dim=1)  # (B, enc_dim)
        if mean_enc.shape[-1] == dec_dim:
            hidden = mean_enc.unsqueeze(0).repeat(num_layers, 1, 1)
        else:
            # 维度不匹配时用零初始化，让 LSTM 自己学习
            hidden = torch.zeros(num_layers, B, dec_dim,
                                 device=encoder_output.device)
        cell = torch.zeros_like(hidden)
        return hidden.contiguous(), cell

    # ---------- 推理解码 ----------

    def _greedy_decode(self, encoder_output, max_len=None):
        """推理模式（LSTM）：Greedy 解码"""
        max_len = max_len or MAX_DECODE_LEN
        B = encoder_output.size(0)
        device = encoder_output.device

        hidden, cell = self._init_decoder_state(encoder_output)
        enc_proj = self.decoder.attention.project_encoder(encoder_output)
        input_token = torch.full((B,), SOS_TOKEN, dtype=torch.long, device=device)

        predictions = []

        for t in range(max_len):
            logits, hidden, cell, _ = self.decoder.forward_step(
                input_token, hidden, cell, encoder_output, enc_proj=enc_proj)
            input_token = logits.argmax(dim=-1)
            predictions.append(input_token)

            if (input_token == EOS_TOKEN).all():
                break

        return torch.stack(predictions, dim=1)  # (B, actual_len)

    def _greedy_decode_transformer(self, memory, mem_mask, max_len=None):
        """
        推理模式（Transformer）：Greedy 解码

        不带 KV cache —— 每步重算整个前缀，正确优先。
        典型公式 <60 token，代价可接受；KV cache 留作后续优化。
        """
        max_len = max_len or MAX_DECODE_LEN
        B = memory.size(0)
        device = memory.device

        generated = torch.full((B, 1), SOS_TOKEN, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        predictions = []

        for _ in range(max_len):
            logits = self.decoder(generated, memory,
                                  memory_key_padding_mask=mem_mask)
            next_token = logits[:, -1].argmax(dim=-1)          # (B,)
            predictions.append(next_token)
            generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)

            finished = finished | (next_token == EOS_TOKEN)
            if finished.all():
                break

        return torch.stack(predictions, dim=1)  # (B, actual_len)

    # ---------- Beam Search ----------

    def beam_search(self, images, beam_width=None, max_len=None, widths=None):
        """
        Beam Search 解码（仅支持 batch_size=1）
        Args:
            beam_width: 默认取 config.BEAM_WIDTH（此前是死配置，从不被读取）
        Returns:
            (1, seq_len) token id，首 token 非 SOS、EOS 收尾
        """
        assert images.size(0) == 1, 'beam_search only supports batch_size=1'
        beam_width = beam_width or BEAM_WIDTH
        max_len = max_len or MAX_DECODE_LEN

        with torch.no_grad():
            memory, mem_mask = self.encode(images, widths)

        if self.is_transformer:
            return self._beam_search_transformer(memory, mem_mask, beam_width, max_len)
        return self._beam_search_lstm(memory, beam_width, max_len)

    @staticmethod
    def _length_penalty(seq_len, alpha=BEAM_LEN_PENALTY):
        """GNMT 风格长度惩罚，alpha=0 表示不惩罚"""
        if not alpha:
            return 1.0
        return ((5.0 + seq_len) / 6.0) ** alpha

    def _beam_search_transformer(self, memory, mem_mask, beam_width, max_len):
        """Transformer 的 Beam Search（每步重算前缀，无 KV cache）"""
        device = memory.device
        # 所有 beam 共享同一张图的编码器输出
        mem = memory.expand(beam_width, -1, -1)
        mask = mem_mask.expand(beam_width, -1) if mem_mask is not None else None

        # 首步
        start = torch.full((1, 1), SOS_TOKEN, dtype=torch.long, device=device)
        logits = self.decoder(start, memory, memory_key_padding_mask=mem_mask)
        log_probs = F.log_softmax(logits[:, -1], dim=-1)           # (1, vocab)
        topk_lp, topk_ids = log_probs.topk(beam_width, dim=-1)     # (1, beam)

        sequences = topk_ids.squeeze(0).unsqueeze(1)               # (beam, 1)
        log_probs_sum = topk_lp.squeeze(0)                         # (beam,)
        finished = torch.zeros(beam_width, dtype=torch.bool, device=device)
        seq_len = 1

        for t in range(1, max_len):
            pfx = torch.cat([torch.full((beam_width, 1), SOS_TOKEN,
                                        dtype=torch.long, device=device),
                             sequences], dim=1)                    # (beam, t+1)
            logits = self.decoder(pfx, mem, memory_key_padding_mask=mask)
            step_logprobs = F.log_softmax(logits[:, -1], dim=-1)   # (beam, vocab)

            total = log_probs_sum.unsqueeze(1) + step_logprobs     # (beam, vocab)
            # 已结束的 beam 只允许继续选 PAD，且得分不再变化
            total[finished] = float('-inf')
            if finished.any():
                total[finished, PAD_TOKEN] = log_probs_sum[finished]

            topk_lp, topk_idx = total.view(-1).topk(beam_width)
            beam_idx = topk_idx // self.vocab_size
            token_idx = topk_idx % self.vocab_size

            sequences = torch.cat([sequences[beam_idx],
                                   token_idx.unsqueeze(1)], dim=1)
            log_probs_sum = topk_lp
            finished = finished[beam_idx] | (token_idx == EOS_TOKEN)
            seq_len = t + 1

            if finished.all():
                break

        # 长度惩罚后再选最优（否则偏向短序列）
        best_idx = (log_probs_sum / self._length_penalty(seq_len)).argmax()
        return sequences[best_idx].unsqueeze(0)[:, :seq_len]

    def _beam_search_lstm(self, encoder_output, beam_width, max_len):
        """LSTM 的 Beam Search（带 hidden/cell 状态）"""
        device = encoder_output.device

        hidden, cell = self._init_decoder_state(encoder_output)

        start_token = torch.tensor([SOS_TOKEN], device=device)
        logits, hidden, cell, _ = self.decoder.forward_step(
            start_token, hidden, cell, encoder_output)
        log_probs = F.log_softmax(logits, dim=-1)  # (1, vocab_size)

        topk_log_probs, topk_ids = log_probs.topk(beam_width, dim=-1)  # (1, beam)

        # 所有 beam 共享同一张图的编码器输出与其投影，故不重排 enc_out，
        # 投影也只算一次（此前每步对全同张量做重排 + 重算，纯属浪费）
        enc_out = encoder_output.expand(beam_width, -1, -1)   # (beam, T_enc, 512)
        enc_proj = self.decoder.attention.project_encoder(encoder_output)
        enc_proj = enc_proj.expand(beam_width, -1, -1)
        hidden = hidden.expand(-1, beam_width, -1).contiguous()
        cell = cell.expand(-1, beam_width, -1).contiguous()

        sequences = torch.zeros(beam_width, max_len, dtype=torch.long, device=device)
        sequences[:, 0] = topk_ids.squeeze(0)
        seq_len = 1
        log_probs_sum = topk_log_probs.squeeze(0)      # (beam,)
        finished = torch.zeros(beam_width, dtype=torch.bool, device=device)

        for t in range(1, max_len):
            input_tokens = sequences[:, t - 1]  # (beam,)

            logits, hidden, cell, _ = self.decoder.forward_step(
                input_tokens, hidden, cell, enc_out, enc_proj=enc_proj)
            log_probs = F.log_softmax(logits, dim=-1)  # (beam, vocab_size)

            total_log_probs = log_probs_sum.unsqueeze(1) + log_probs  # (beam, vocab)

            # 屏蔽已终止 beam
            total_log_probs[finished] = float('-inf')
            if finished.any():
                total_log_probs[finished, PAD_TOKEN] = log_probs_sum[finished]

            total_log_probs_flat = total_log_probs.view(-1)
            topk_lp, topk_idx = total_log_probs_flat.topk(beam_width)
            beam_idx = topk_idx // self.vocab_size
            token_idx = topk_idx % self.vocab_size

            sequences = sequences[beam_idx]
            sequences[:, t] = token_idx
            seq_len = t + 1
            log_probs_sum = topk_lp
            hidden = hidden[:, beam_idx, :]
            cell = cell[:, beam_idx, :]

            finished = finished[beam_idx] | (token_idx == EOS_TOKEN)
            if finished.all():
                break

        best_idx = (log_probs_sum / self._length_penalty(seq_len)).argmax()
        return sequences[best_idx].unsqueeze(0)[:, :seq_len]


# ==================== 测试 ====================

def _self_test(arch):
    print('=' * 62)
    print(f'FormulaRecognizer 模型测试 — arch = {arch}')
    print('=' * 62)

    vocab_size = 245
    model = FormulaRecognizer(vocab_size=vocab_size, arch=arch)
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f'模型参数量: {total_params:.1f}M (可训练 {trainable:.1f}M)')
    if arch == 'vit_transformer':
        print(f'预训练: {model.encoder.pretrain_summary()}')

    # 变长输入：batch 内宽度不同（padding 到最大宽度）
    B = 2
    widths = torch.tensor([300, 800])
    max_w = int(widths.max())
    images = torch.randn(B, 1, 64, max_w)

    targets = torch.randint(0, vocab_size, (B, 20))
    targets[:, 0] = SOS_TOKEN
    targets[1, 15:] = PAD_TOKEN

    model.train()
    logits, tgts = model.training_step(images, targets, widths=widths)
    assert logits.size(-1) == vocab_size, f'训练输出维度异常: {logits.shape}'
    assert logits.size(0) == tgts.size(0), \
        f'logits 与 targets 数量不一致: {logits.size(0)} vs {tgts.size(0)}'
    print(f'训练输出: logits {tuple(logits.shape)}  vs targets {tuple(tgts.shape)}')

    model.eval()
    with torch.no_grad():
        preds = model(images, widths=widths)
        print(f'贪心推理输出: {preds.shape}  (期望 ({B}, L))')

        beam = model.beam_search(images[:1], widths=widths[:1])
        print(f'Beam Search 输出: {beam.shape}  (期望 (1, L))')
        assert beam.size(0) == 1

    # 位置编码形状自检
    if arch == 'vit_transformer':
        for rows, cols in [(4, 19), (4, 50), (4, 75)]:
            pe = build_2d_sincos_pos_embed(VIT_HIDDEN, rows, cols)
            assert pe.shape == (rows * cols, VIT_HIDDEN), pe.shape
        print('2D 位置编码形状自检通过: 4x19 / 4x50 / 4x75')

    print(f'[OK] {arch} 所有维度检查通过')


if __name__ == '__main__':
    for a in ('vit_transformer', 'cnn_lstm'):
        _self_test(a)
        print()
