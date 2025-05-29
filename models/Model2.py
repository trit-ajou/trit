from torch import nn


class Model2(nn.Module):
    """Pixel-wise Mask Generation Model"""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
