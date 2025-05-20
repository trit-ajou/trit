import torch
from torch import nn

from ..datas.Utils import BBox


class Model1(nn.Module):
    """Text Object BBox Detection Model"""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

    @staticmethod
    def preprocess(bboxes: list[list[BBox]]) -> torch.Tensor:
        pass

    @staticmethod
    def postprocess(preds: torch.Tensor) -> list[list[BBox]]:
        pass
