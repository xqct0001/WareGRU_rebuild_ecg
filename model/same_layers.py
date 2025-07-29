import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Conv1dSame(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, bias=True):
        super(Conv1dSame, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation, bias=bias)
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        L_in = x.shape[-1]

        # 目标输出大小
        L_out = math.ceil(L_in / self.stride)

        # 需要的总padding量
        padding_needed = max(0, (L_out - 1) * self.stride + (self.kernel_size - 1) * self.dilation + 1 - L_in)

        pad_left = padding_needed // 2
        pad_right = padding_needed - pad_left

        x = F.pad(x, (pad_left, pad_right))
        return self.conv(x)

class ConvTranspose1dSame(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True):
        super(ConvTranspose1dSame, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.deconv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, bias=bias)

    def forward(self, x):
        # 计算目标输出长度
        L_in = x.shape[-1]
        L_out = L_in * self.stride

        x = self.deconv(x)

        # 裁剪或补0处理，保证输出长度= L_out
        if x.shape[-1] > L_out:
            x = x[..., :L_out]
        elif x.shape[-1] < L_out:
            pad_amount = L_out - x.shape[-1]
            x = F.pad(x, (0, pad_amount))
        return x

class MaxPool1dSame(nn.Module):
    def __init__(self, kernel_size, stride=None):
        super(MaxPool1dSame, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride else kernel_size

    def forward(self, x):
        L_in = x.shape[-1]
        L_out = math.ceil(L_in / self.stride)
        padding_needed = max(0, (L_out - 1) * self.stride + self.kernel_size - L_in)

        pad_left = padding_needed // 2
        pad_right = padding_needed - pad_left

        x = F.pad(x, (pad_left, pad_right), mode='replicate')
        return F.max_pool1d(x, self.kernel_size, self.stride)
