import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, p_drop=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU(inplace=True)
        self.drop2d= nn.Dropout2d(p_drop) if p_drop > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.drop2d(x)
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, p_drop=0.0):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = ConvBlock(in_ch, out_ch, p_drop)

    def forward(self, x):
        return self.conv(self.pool(x))


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, p_drop=0.0):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = ConvBlock(in_ch, out_ch, p_drop)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            dh = skip.shape[-2] - x.shape[-2]
            dw = skip.shape[-1] - x.shape[-1]
            x = F.pad(x, (0, dw, 0, dh))
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_ch=1, num_classes=2,
                 enc=(32, 32, 64, 128, 128, 128),
                 dec=(128, 128, 64, 32, 16),
                 p_drop=0.0):
        super().__init__()
        e0,e1,e2,e3,e4,e5 = enc
        d1,d2,d3,d4,d5 = dec

        self.in_conv = ConvBlock(in_ch, e0, p_drop)
        self.down1 = DownBlock(e0, e1, p_drop)
        self.down2 = DownBlock(e1, e2, p_drop)
        self.down3 = DownBlock(e2, e3, p_drop)
        self.down4 = DownBlock(e3, e4, p_drop)
        self.down5 = DownBlock(e4, e5, p_drop)

        self.up1 = UpBlock(e5 + e4, d1, p_drop)
        self.up2 = UpBlock(d1 + e3, d2, p_drop)
        self.up3 = UpBlock(d2 + e2, d3, p_drop)
        self.up4 = UpBlock(d3 + e1, d4, p_drop)
        self.up5 = UpBlock(d4 + e0, d5, p_drop)

        self.out_conv = nn.Conv2d(d5, num_classes, kernel_size=1)

    def forward(self, x):
        s0 = self.in_conv(x)
        s1 = self.down1(s0)
        s2 = self.down2(s1)
        s3 = self.down3(s2)
        s4 = self.down4(s3)
        b  = self.down5(s4)

        x  = self.up1(b,  s4)
        x  = self.up2(x,  s3)
        x  = self.up3(x,  s2)
        x  = self.up4(x,  s1)
        x  = self.up5(x,  s0)
        return self.out_conv(x)  # logits