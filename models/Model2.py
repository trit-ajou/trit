import torch.nn as nn


class Model2(nn.Module):
    """Pixel-wise Mask Generation Model"""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

    @staticmethod
    def union_masks():
        """mask 이미지를 받아서 원본 이미지 크기로 마스크 생성"""
