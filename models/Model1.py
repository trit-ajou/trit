from torch import nn

from ..datas.Utils import BBox


class Model1(nn.Module):
    """Text Object BBox Detection Model"""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

    @staticmethod
    def postprocess(preds) -> list[list[BBox]]:
        """shuffle 되지 않은 torch.Tensor list(model 1 pred) 받아서 BBox 이중 리스트 반환"""
        pass
