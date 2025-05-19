import torch
import torchvision.transforms.functional as VTF
from torch.utils.data import Dataset
from torch import Tensor as img_tensor

from .Utils import BBox
from .TextedImage import TextedImage


class _BaseMangaDataset(Dataset):
    def __init__(
        self,
        texted_images: list[TextedImage],
        input_size: tuple[int, int],
    ):
        self.texted_images = texted_images
        self.input_size = input_size

    def __len__(self):
        return len(self.texted_images)


class _BaseMangaDataset_per_BBox(_BaseMangaDataset):
    def __init__(self, texted_images: list[TextedImage], input_size: tuple[int, int]):
        super().__init__(texted_images, input_size)
        # Create a list of (image_index, bbox) tuples
        self.item_map: list[tuple[int, BBox]] = []
        for img_idx, img in enumerate(texted_images):
            for bbox in img.bboxes:
                self.item_map.append((img_idx, bbox))

    def __len__(self):
        return len(self.item_map)


class MangaDataset1(_BaseMangaDataset):
    def __init__(self, texted_images: list[TextedImage], input_size: tuple[int, int]):
        super().__init__(texted_images, input_size)

    def __getitem__(self, idx: int):
        texted_image = self.texted_images[idx]

        # Model 1 Input: timg resized and padded
        model1_input = TextedImage.resize_keep_aspect(
            texted_image.timg, self.input_size
        )

        # Model 1 Target: bboxes (as a list of BBox objects for now)
        model1_target = texted_image.bboxes

        return model1_input, model1_target


class MangaDataset2(_BaseMangaDataset_per_BBox):
    def __init__(
        self, texted_images: list[TextedImage], input_size: tuple[int, int], margin: int
    ):
        super().__init__(texted_images, input_size)
        self.margin = margin

    def __getitem__(self, idx) -> tuple[img_tensor, img_tensor, int, BBox]:
        img_idx, bbox = self.item_map[idx]
        texted_image = self.texted_images[img_idx]

        # Model 2 Input: timg margin crop, resize, pad
        # 1. Expand bbox by margin
        expanded_bbox = bbox.expand(self.margin, texted_image.size)
        # 2. Crop timg using expanded bbox
        cropped_timg = texted_image.timg[expanded_bbox.slice]
        # 3. Resize and pad the cropped timg
        model2_input = TextedImage.resize_keep_aspect(cropped_timg, self.input_size)

        # Model 2 Target: mask precise crop, pad(margin), resize, pad
        # 1. Precise crop mask using original bbox
        cropped_mask = texted_image.mask[bbox.slice]
        # 2. Pad with margin(as same as expanded bbox)
        padded_mask = VTF.pad(cropped_mask, self.margin)
        # 3. Resize and pad
        model2_target = TextedImage.resize_keep_aspect(padded_mask, self.input_size)

        # Return original bbox for postprocessing (unioning masks)
        return model2_input, model2_target


class MangaDataset3(_BaseMangaDataset_per_BBox):
    def __init__(self, texted_images: list[TextedImage], input_size: tuple[int, int]):
        super().__init__(texted_images, input_size)

    def __getitem__(self, idx):
        img_idx, bbox = self.item_map[idx]
        texted_image = self.texted_images[img_idx]

        # Model 3 Input 1: timg center crop (USING SHIFTING LOGIC)
        model3_input1 = TextedImage.center_crop(
            texted_image.timg, bbox, self.input_size
        )

        # 2. Model 3 Input 2: mask precise crop -> paste on zero image -> center crop as same as timg center crop
        cropped_mask = texted_image.mask[bbox.slice]
        background = torch.zeros_like(texted_image.mask)
        background = TextedImage.alpha_blend_at_bbox(background, bbox, cropped_mask)
        model3_input2 = TextedImage.center_crop(background, bbox, self.input_size)

        # 3. Model 3 Target: timg with orig precise crop pasted, then center cropped (with shifting/padding)
        cropped_orig = texted_image.orig[bbox.slice]
        modified_timg = TextedImage.alpha_blend_at_bbox(
            texted_image.timg, bbox, cropped_orig
        )
        model3_target = TextedImage.center_crop(modified_timg, bbox, self.input_size)

        # Return original bbox and image index for postprocessing (pasting result back)
        return model3_input1, model3_input2, model3_target
