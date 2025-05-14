import torch.nn as nn

from ..datas.Utils import TextedImage


class Model1(nn.Module):
    """Text Object BBox Detection Model"""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

    @staticmethod
    def postprocess(x) -> list[TextedImage]:
        pass
