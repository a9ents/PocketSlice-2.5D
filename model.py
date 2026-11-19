import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)

class TinyUNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=2, base=32): # out_ch 改为 2
        super().__init__()
        self.down1 = DoubleConv(in_ch, base)
        self.down2 = DoubleConv(base, base * 2)
        self.down3 = DoubleConv(base * 2, base * 4)
        self.pool = nn.MaxPool2d(2)
        
        self.up3 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.conv3 = DoubleConv(base * 4, base * 2)
        self.up2 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.conv2 = DoubleConv(base * 2, base)
        self.out = nn.Conv2d(base, out_ch, 1)

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(self.pool(d1))
        d3 = self.down3(self.pool(d2))
        
        u3 = self.up3(d3)
        if u3.shape[2:] != d2.shape[2:]:
            u3 = F.interpolate(u3, size=d2.shape[2:], mode='bilinear', align_corners=False)
        u3 = self.conv3(torch.cat([u3, d2], dim=1))
        
        u2 = self.up2(u3)
        if u2.shape[2:] != d1.shape[2:]:
            u2 = F.interpolate(u2, size=d1.shape[2:], mode='bilinear', align_corners=False)
        u2 = self.conv2(torch.cat([u2, d1], dim=1))
        
        out = self.out(u2)
        heatmap = torch.sigmoid(out[:, 0:1]) # 靶心概率 0~1
        scale = F.relu(out[:, 1:2])          # 口袋大小，必须非负
        return heatmap, scale