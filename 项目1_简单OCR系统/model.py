"""
CRNN模型 - CNN + RNN用于OCR
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import NUM_CLASSES, IMG_HEIGHT


class CNN(nn.Module):
    """CNN特征提取器"""

    def __init__(self):
        super(CNN, self).__init__()

        # 卷积层
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))

        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(512)

        self.conv6 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(512)
        self.pool6 = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))

        self.conv7 = nn.Conv2d(512, 512, kernel_size=2, stride=1, padding=0)
        self.bn7 = nn.BatchNorm2d(512)

    def forward(self, x):
        # x: (batch, 1, 32, 320)

        x = self.pool1(F.relu(self.bn1(self.conv1(x))))  # (batch, 64, 16, 160)
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))  # (batch, 128, 8, 80)
        x = F.relu(self.bn3(self.conv3(x)))  # (batch, 256, 8, 80)
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))  # (batch, 256, 4, 80)
        x = F.relu(self.bn5(self.conv5(x)))  # (batch, 512, 4, 80)
        x = self.pool6(F.relu(self.bn6(self.conv6(x))))  # (batch, 512, 2, 80)
        x = F.relu(self.bn7(self.conv7(x)))  # (batch, 512, 1, 79)

        return x


class RNN(nn.Module):
    """RNN序列建模"""

    def __init__(self, input_size, hidden_size, num_layers=2):
        super(RNN, self).__init__()

        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=0.3
        )

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        recurrent, _ = self.rnn(x)
        return recurrent


class CRNN(nn.Module):
    """CRNN模型: CNN + RNN + CTC"""

    def __init__(self, hidden_size=256):
        super(CRNN, self).__init__()

        self.cnn = CNN()
        self.rnn = RNN(512, hidden_size)

        # 全连接层映射到字符类别
        self.fc = nn.Linear(hidden_size * 2, NUM_CLASSES)  # *2 因为双向LSTM

    def forward(self, x):
        # CNN特征提取
        conv_feat = self.cnn(x)  # (batch, 512, 1, 79)

        # 重塑为序列 (batch, width, channels)
        b, c, h, w = conv_feat.size()
        conv_feat = conv_feat.squeeze(2)  # (batch, 512, 79)
        conv_feat = conv_feat.permute(0, 2, 1)  # (batch, 79, 512)

        # RNN序列建模
        rnn_feat = self.rnn(conv_feat)  # (batch, seq_len, hidden_size*2)

        # 全连接层
        output = self.fc(rnn_feat)  # (batch, seq_len, num_classes)

        # CTC需要log_softmax
        output = F.log_softmax(output, dim=2)

        return output


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
    # 测试模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CRNN().to(device)

    # 测试输入
    batch_size = 4
    x = torch.randn(batch_size, 1, 32, 320).to(device)

    # 前向传播
    output = model(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")

    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
