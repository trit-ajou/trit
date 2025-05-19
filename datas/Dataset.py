import torch
import torchvision.transforms.functional as VTF
from torch.utils.data import Dataset
from torch import Tensor as img_tensor

from .Utils import BBox
from .TextedImage import TextedImage


class MangaDataset(Dataset):
    def __init__(self, texted_images: list[TextedImage], input_size: tuple[int, int]):
        super().__init__()
        self.texted_images = texted_images
        self.input_size = input_size

    def __len__(self):
        return len(self.texted_images)

    def __getitem__(self, idx: int):
        texted_image = self.texted_images[idx]
        orig = TextedImage._resize(texted_image.orig, self.input_size)
        timg = TextedImage._resize(texted_image.timg, self.input_size)
        mask = TextedImage._resize(texted_image.mask, self.input_size)
        return orig, timg, mask, texted_image.bboxes
