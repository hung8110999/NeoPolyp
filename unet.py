# unet.py (sửa phần forward để pad khi cần)
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        self.down1 = DoubleConv(in_channels, 64)
        self.down2 = DoubleConv(64, 128)
        self.down3 = DoubleConv(128, 256)
        self.down4 = DoubleConv(256, 512)
        self.bottleneck = DoubleConv(512, 1024)

        self.pool = nn.MaxPool2d(2)

        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(128, 64)

        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

    def _pad_to_match(self, x, ref):
        """
        Pad tensor x so its (H,W) equals ref's (H,W).
        x, ref: tensors with shape [B, C, H, W]
        Uses F.pad with order (left, right, top, bottom)
        """
        diffY = ref.size(2) - x.size(2)
        diffX = ref.size(3) - x.size(3)
        if diffY == 0 and diffX == 0:
            return x
        pad_left = diffX // 2
        pad_right = diffX - pad_left
        pad_top = diffY // 2
        pad_bottom = diffY - pad_top
        # F.pad expects (left, right, top, bottom)
        x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom))
        return x

    def forward(self, x):
        x1 = self.down1(x)            # -> same as input
        x2 = self.down2(self.pool(x1))
        x3 = self.down3(self.pool(x2))
        x4 = self.down4(self.pool(x3))

        xb = self.bottleneck(self.pool(x4))

        # up4
        u4 = self.up4(xb)
        u4 = self._pad_to_match(u4, x4)
        u4 = torch.cat([u4, x4], dim=1)
        u4 = self.conv4(u4)

        # up3
        u3 = self.up3(u4)
        u3 = self._pad_to_match(u3, x3)
        u3 = torch.cat([u3, x3], dim=1)
        u3 = self.conv3(u3)

        # up2
        u2 = self.up2(u3)
        u2 = self._pad_to_match(u2, x2)
        u2 = torch.cat([u2, x2], dim=1)
        u2 = self.conv2(u2)

        # up1
        u1 = self.up1(u2)
        u1 = self._pad_to_match(u1, x1)
        u1 = torch.cat([u1, x1], dim=1)
        u1 = self.conv1(u1)

        return self.final(u1)
