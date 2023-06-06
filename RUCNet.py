import torch
from scse import SCSEBlock
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()

        if not mid_channels:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(4, mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(4, out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.identity = nn.Identity()

        self.conv0 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)

        self.conv1_0 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, padding=0, bias=False)
        self.conv1_1 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)

        self.scse = SCSEBlock(out_channels, reduction=2)

    def forward(self, x):

        x1 = self.identity(x)

        x1 = self.conv1_0(x1)
        x = self.conv0(x)
        x = self.conv1(x)
        x += x1

        x1 = self.identity(x)
        x1 = self.conv1_1(x1)
        x = self.conv2(x)
        x = self.conv3(x)
        x += x1

        return self.scse(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels)
        self.scse = SCSEBlock(out_channels, reduction=2)

    def forward(self, x1, x2):
        x = torch.cat((self.up(x1), x2), dim=1)
        x = self.conv(x)
        return self.scse(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class RUCNet(nn.Module):
    def __init__(self, n_channels, n_classes, reduction=2):
        super(RUCNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.reduction = reduction

        self.scse_down1 = SCSEBlock(64, reduction=self.reduction)
        self.scse_down2 = SCSEBlock(128, reduction=self.reduction)
        self.scse_down3 = SCSEBlock(256, reduction=self.reduction)
        self.scse_down4 = SCSEBlock(512, reduction=self.reduction)

        self.scse_up1 = SCSEBlock(256, reduction=self.reduction)
        self.scse_up2 = SCSEBlock(128, reduction=self.reduction)
        self.scse_up3 = SCSEBlock(64, reduction=self.reduction)
        self.scse_up4 = SCSEBlock(64, reduction=self.reduction)

        self.inc = (DoubleConv(n_channels, 64))

        self.down1 = (Down(64, 64))
        self.down2 = (Down(64, 128))
        self.down3 = (Down(128, 256))
        self.down4 = (Down(256, 512))

        self.up1 = (Up(768, 256))
        self.up2 = (Up(384, 128))
        self.up3 = (Up(192, 64))
        self.up4 = (Up(128, 64))

        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        x1 = self.inc(x)

        x2 = self.down1(x1)
        x2 = self.scse_down1(x2)

        x3 = self.down2(x2)
        x3 = self.scse_down2(x3)

        x4 = self.down3(x3)
        x4 = self.scse_down3(x4)

        x5 = self.down4(x4)
        x5 = self.scse_down4(x5)

        x = self.up1(x5, x4)
        x = self.scse_up1(x)

        x = self.up2(x, x3)
        x = self.scse_up2(x)

        x = self.up3(x, x2)
        x = self.scse_up3(x)

        x = self.up4(x, x1)
        x = self.scse_up4(x)

        return self.outc(x)
