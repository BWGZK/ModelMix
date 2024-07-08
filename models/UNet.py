import torch
import torch.nn as nn
import numpy as np


class conv_block(nn.Module):
    def __init__(self, in_ch, out_ch, dropout_p):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()

        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.up(x)
        return x


class UNet(nn.Module):
    def __init__(self, in_ch):
        super(UNet, self).__init__()
        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        dropout_probs = np.array([0.05, 0.1, 0.2, 0.3, 0.5]) / 3

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch, filters[0], dropout_probs[0])
        self.Conv2 = conv_block(filters[0], filters[1], dropout_probs[1])
        self.Conv3 = conv_block(filters[1], filters[2], dropout_probs[2])
        self.Conv4 = conv_block(filters[2], filters[3], dropout_probs[3])
        self.Conv5 = conv_block(filters[3], filters[4], dropout_probs[4])

        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = conv_block(filters[4], filters[3], 0)

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2], 0)

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1], 0)

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0], 0)

        # self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)
        # self.active = nn.Softmax(dim=1)

    def forward(self, x):  # [bs, in_ch, H, W]

        e1 = self.Conv1(x)  # [bs, n1, H, W]

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)  # [bs, n1*2, H/2, W/2]

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)  # [bs, n1*4, H/4, W/4]]

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)  # [bs, n1*8, H/8, W/8]

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)  # [bs, n1*16, H/16, W/16]

        d5 = self.Up5(e5)
        d5 = torch.cat((e4, d5), dim=1)
        d5 = self.Up_conv5(d5)  # [bs, n1*8, H/8, W/8]

        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)  # [bs, n1*4, H/4, W/4]

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)  # [bs, n1*2, H/2, H/2]

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)  # [bs, n1, H, W]

        # d1 = self.Conv(d2)  # [bs, out_ch, H, W]
        # out = self.active(d1)

        return d2


class Head(nn.Module):
    def __init__(self, out_ch):
        super().__init__()
        self.Head = nn.Conv2d(64, out_ch, kernel_size=1, stride=1, padding=0)
        self.active = nn.Softmax(dim=1)

    def forward(self, features: torch.Tensor):
        out = self.Head(features)
        seg = self.active(out)
        out = {"seg": seg}
        return out
