import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, p_drop=0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=p_drop) if p_drop > 0 else nn.Identity(),
        )

    def forward(self, x):
        return self.block(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch, p_drop=0.0):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch, p_drop)

    def forward(self, x):
        return self.conv(self.pool(x))

class Up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True, p_drop=0.0):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_ch, out_ch, p_drop)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_ch, out_ch, p_drop)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, n_classes=1, base_ch=32, bilinear=True, dropout_p=0.1):
        super().__init__()
        self.inc = DoubleConv(in_channels, base_ch, p_drop=dropout_p)
        self.down1 = Down(base_ch, base_ch * 2, p_drop=dropout_p)
        self.down2 = Down(base_ch * 2, base_ch * 4, p_drop=dropout_p)
        self.down3 = Down(base_ch * 4, base_ch * 8, p_drop=dropout_p)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_ch * 8, (base_ch * 16) // factor, p_drop=dropout_p)
        self.up1 = Up(base_ch * 16, (base_ch * 8) // factor, bilinear, p_drop=dropout_p)
        self.up2 = Up(base_ch * 8, (base_ch * 4) // factor, bilinear, p_drop=dropout_p)
        self.up3 = Up(base_ch * 4, (base_ch * 2) // factor, bilinear, p_drop=dropout_p)
        self.up4 = Up(base_ch * 2, base_ch, bilinear, p_drop=dropout_p)
        self.outc = nn.Conv2d(base_ch, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x,  x3)
        x = self.up3(x,  x2)
        x = self.up4(x,  x1)
        logits = self.outc(x)
        return logits
