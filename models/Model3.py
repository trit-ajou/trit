import torch
import torch.nn as nn
import torch.nn.functional as F
from .BaseModel import BaseModel  # Correct import from the same directory
from typing import Optional  # For type hinting device


# --- U-Net Components (redefined for Model3 to be self-contained for this task) ---
class DoubleConv_M3(nn.Module):
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


class Down_M3(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv_M3(in_channels, out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up_M3(nn.Module):
    """Upscaling then double conv"""

    def __init__(
        self,
        ch_in_lower: int,
        ch_in_skip: int,
        ch_out_doubleconv: int,
        bilinear: bool = True,
    ):
        super().__init__()
        self.bilinear = bilinear  # Store for decision
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            # After upsampling, ch_in_lower remains the same.
            # After concat with skip, it's ch_in_lower + ch_in_skip
            self.conv = DoubleConv_M3(ch_in_lower + ch_in_skip, ch_out_doubleconv)
        else:
            # ConvTranspose2d changes channel from ch_in_lower to ch_in_lower // 2 (conventionally)
            self.up = nn.ConvTranspose2d(
                ch_in_lower, ch_in_lower // 2, kernel_size=2, stride=2
            )
            # After concat, it's (ch_in_lower // 2) + ch_in_skip
            self.conv = DoubleConv_M3(
                (ch_in_lower // 2) + ch_in_skip, ch_out_doubleconv
            )

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


class OutConv_M3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


# --- End of U-Net Components ---


class Model3(BaseModel):
    """Masked Inpainting Model using U-Net like architecture"""

    def __init__(
        self,
        img_channels: int = 3,
        mask_channels: int = 1,
        output_channels: int = 3,
        bilinear: bool = True,
        device: Optional[str] = None,
    ):
        # If device is not specified, BaseModel will handle the default
        super().__init__(device=device)

        self.img_channels = img_channels
        self.mask_channels = mask_channels
        self.output_channels = output_channels
        self.bilinear = bilinear

        n_input_channels = img_channels + mask_channels

        # Encoder
        self.inc = DoubleConv_M3(n_input_channels, 64)
        self.down1 = Down_M3(64, 128)
        self.down2 = Down_M3(128, 256)
        self.down3 = Down_M3(256, 512)
        self.down4 = Down_M3(512, 1024)  # Deepest layer

        # Decoder
        # Arguments for Up_M3: ch_in_lower, ch_in_skip, ch_out_doubleconv, bilinear
        self.up1 = Up_M3(1024, 512, 512, bilinear)
        self.up2 = Up_M3(512, 256, 256, bilinear)
        self.up3 = Up_M3(256, 128, 128, bilinear)
        self.up4 = Up_M3(128, 64, 64, bilinear)
        self.outc = OutConv_M3(64, output_channels)

        self.final_activation = (
            torch.sigmoid
        )  # Assuming output pixel values are in [0, 1]

        self.to(self.device)  # Ensure model is on the device specified in BaseModel

    def forward(
        self, image_with_text: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        # Ensure inputs are on the correct device
        image_with_text_dev = image_with_text.to(self.device)
        mask_dev = mask.to(self.device)

        # Concatenate image and mask along channel dimension
        # image_with_text: (B, C_img, H, W), mask: (B, C_mask, H, W)
        # Result x: (B, C_img + C_mask, H, W)
        x = torch.cat([image_with_text_dev, mask_dev], dim=1)

        # Encoder path
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)  # Deepest features

        # Decoder path
        out = self.up1(x5, x4)  # x5 from below, x4 is skip
        out = self.up2(out, x3)  # out from below, x3 is skip
        out = self.up3(out, x2)  # out from below, x2 is skip
        out = self.up4(out, x1)  # out from below, x1 is skip

        logits = self.outc(out)  # Raw output values from the network

        # Apply final activation to constrain output (e.g., to [0,1] for image pixels)
        return self.final_activation(logits)
