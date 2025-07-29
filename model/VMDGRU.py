import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from model.same_layers import *

# 自定义 FlipLayer（代替 nn.Lambda）
class FlipLayer(nn.Module):
    def __init__(self, dim):
        super(FlipLayer, self).__init__()
        self.dim = dim

    def forward(self, x):
        return torch.flip(x, [self.dim])

class CustomNet(nn.Module):
    def __init__(self):
        super(CustomNet, self).__init__()

        self.conv1 = Conv1dSame(1, 24, kernel_size=64, stride=4)
        self.bn1 = nn.BatchNorm1d(24)
        self.pool1 = MaxPool1dSame(2, 1)

        self.conv2 = Conv1dSame(24, 16, kernel_size=32, stride=4)
        self.bn2 = nn.BatchNorm1d(16)
        self.pool2 = MaxPool1dSame(2, 1)

        self.conv3 = Conv1dSame(16, 16, kernel_size=16, stride=4)
        self.bn3 = nn.BatchNorm1d(16)
        self.pool3 = MaxPool1dSame(2, 1)

        self.tp1 = ConvTranspose1dSame(16, 16, kernel_size=32, stride=4)
        self.tp2 = ConvTranspose1dSame(16, 16, kernel_size=32, stride=4)
        self.tp3 = ConvTranspose1dSame(16, 24, kernel_size=64, stride=4)

        self.tanh = nn.Tanh()

        self.gru1 = nn.GRU(24, 16, batch_first=True)
        self.flip = FlipLayer(dim=2)
        self.gru2 = nn.GRU(24, 16, batch_first=True)

        self.fc0 = nn.Linear(32, 16)
        self.dp0 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(16, 8)
        self.dp1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(8, 4)
        self.dp2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(4, 1)

    def forward(self, x):

        x = self.tanh(self.pool1(self.bn1(self.conv1(x))))
        x = self.tanh(self.pool2(self.bn2(self.conv2(x))))
        x = self.tanh(self.pool3(self.bn3(self.conv3(x))))

        x = self.tanh(self.tp1(x))
        x = self.tanh(self.tp2(x))
        x = self.tanh(self.tp3(x))

        x = x.permute(0,2,1)  # (B,1024,24)

        out1, _ = self.gru1(x)
        out2 = self.flip(x)
        out2, _ = self.gru2(out2)

        x = torch.cat((out1, out2), dim=-1)  # (B,1024,32)

        x = self.fc0(x)
        x = self.dp0(x)
        x = self.fc1(x)
        x = self.dp1(x)
        x = self.fc2(x)
        x = self.dp2(x)
        x = self.fc3(x)

        return x.permute(0,2,1)  # 输出回 (B,1,1024)

# 测试
if __name__ == '__main__':
    model = CustomNet()
    dummy = torch.randn(2, 1, 1024)
    out = model(dummy)
    print(out.shape)  # 应该是 (2, 1, 1024)

