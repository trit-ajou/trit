import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Dict, Any # Import Any
from .TextedImage import TextedImage # TextedImage is in the same directory

class MangaDataset1(Dataset):
    def __init__(self, texted_image_list: List[TextedImage], transforms=None):
        super().__init__()
        self.texted_images = texted_image_list
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.texted_images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, List[Tuple[int, int, int, int]]]:
        texted_image = self.texted_images[idx]
        
        input_tensor = texted_image.timg # Should be (C, H, W)
        
        # BBox objects are subclasses of tuple, so they are already in (x1, y1, x2, y2) format.
        # Model1 will expect a list of such tuples.
        target_bboxes = texted_image.bboxes 
        
        if self.transforms:
            # Transforms might need to handle both image and bboxes.
            # This is a placeholder for now, assuming it only handles the image.
            # For a real object detection model, augmentations for bboxes would also be needed.
            input_tensor = self.transforms(input_tensor)
            
        return input_tensor, target_bboxes

class MangaDataset2(Dataset):
    def __init__(self, texted_image_list: List[TextedImage], transforms=None):
        super().__init__()
        self.texted_images = texted_image_list
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.texted_images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        texted_image = self.texted_images[idx]
        input_tensor = texted_image.timg
        target_mask = texted_image.mask # Should be (1, H, W)

        if self.transforms:
            # For segmentation, transforms might apply to both image and mask.
            # This is a placeholder for now, assuming it only handles the image.
            # If augmentations are applied, they often need to be consistent for image and mask.
            input_tensor = self.transforms(input_tensor)
            # target_mask = self.transforms(target_mask) # Example if mask also needs transform

        return input_tensor, target_mask

class MangaDataset3(Dataset):
    def __init__(self, texted_image_list: List[TextedImage], transforms=None):
        super().__init__()
        self.texted_images = texted_image_list
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.texted_images)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        texted_image = self.texted_images[idx]
        
        # Model3 input is typically the texted image and the mask
        input_dict = {
            "image_with_text": texted_image.timg, # (C, H, W)
            "mask": texted_image.mask             # (1, H, W)
        }
        target_image = texted_image.orig # (C, H, W)

        if self.transforms:
            # Transforms might apply to image_with_text, mask, and target_image.
            # This is a placeholder for now, assuming it only handles the image_with_text.
            # Consistent transforms would be crucial for all parts if augmentations are used.
            input_dict["image_with_text"] = self.transforms(input_dict["image_with_text"])
            # Example if other parts also need transform:
            # input_dict["mask"] = self.transforms(input_dict["mask"]) 
            # target_image = self.transforms(target_image)

        return input_dict, target_image
