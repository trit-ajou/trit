from torch import nn


class Model1(nn.Module):
    """Text Object BBox Detection Model"""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
