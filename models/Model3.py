from torch import nn


class Model3(nn.Module):
    """Masked Inpainting Model"""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
