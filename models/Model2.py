import torch
import torch.nn as nn
import torch.nn.functional as F
from .BaseModel import BaseModel  # Correct import from the same directory
from typing import Optional  # For type hinting device


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(
        self,
        ch_in_lower: int,
        ch_in_skip: int,
        ch_out_doubleconv: int,
        bilinear: bool = True,
    ):
        super().__init__()
        self.bilinear = bilinear
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            # After upsampling, ch_in_lower remains the same.
            # After concat with skip, it's ch_in_lower + ch_in_skip
            self.conv = DoubleConv(ch_in_lower + ch_in_skip, ch_out_doubleconv)
        else:
            # ConvTranspose2d changes channel from ch_in_lower to ch_in_lower // 2 (conventionally)
            self.up = nn.ConvTranspose2d(
                ch_in_lower, ch_in_lower // 2, kernel_size=2, stride=2
            )
            # After concat, it's (ch_in_lower // 2) + ch_in_skip
            self.conv = DoubleConv((ch_in_lower // 2) + ch_in_skip, ch_out_doubleconv)

    def forward(
        self, x1: torch.Tensor, x2: torch.Tensor
    ) -> torch.Tensor:  # x1 from lower (e.g. x5), x2 from skip (e.g. x4)
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        # F.pad syntax: (padding_left, padding_right, padding_top, padding_bottom)
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Model2(BaseModel):
    """Pixel-wise Mask Generation Model using U-Net"""

    def __init__(
        self,
        n_channels: int = 3,
        n_classes: int = 1,
        bilinear: bool = True,
        device: Optional[str] = None,
    ):
        # If device is not specified, BaseModel will handle the default
        super().__init__(device=device)

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)  # Deepest layer feature map from encoder

        # Decoder path
        # Arguments for Up: ch_in_lower, ch_in_skip, ch_out_doubleconv, bilinear
        self.up1 = Up(
            1024, 512, 512, bilinear
        )  # Input from down4 (1024), skip from down3 (512) -> Output 512
        self.up2 = Up(
            512, 256, 256, bilinear
        )  # Input from up1 (512), skip from down2 (256) -> Output 256
        self.up3 = Up(
            256, 128, 128, bilinear
        )  # Input from up2 (256), skip from down1 (128) -> Output 128
        self.up4 = Up(
            128, 64, 64, bilinear
        )  # Input from up3 (128), skip from inc (64) -> Output 64
        self.outc = OutConv(64, n_classes)

        self.to(self.device)  # Ensure model is on the device specified in BaseModel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_on_device = x.to(self.device)  # Ensure input is on the correct device

        x1 = self.inc(x_on_device)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)  # Deepest features

        # Pass through decoder
        x = self.up1(x5, x4)  # x5 from below, x4 is skip
        x = self.up2(x, x3)  # x from below, x3 is skip
        x = self.up3(x, x2)  # x from below, x2 is skip
        x = self.up4(x, x1)  # x from below, x1 is skip

        logits = self.outc(x)
        # Return raw logits; sigmoid will be applied by BCEWithLogitsLoss or in post-processing
        return logits
