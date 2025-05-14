import torch
import torchvision.transforms.functional as VTF
from enum import Enum
from PIL import Image


class Lang(Enum):
    EN = 1  # english
    KR = 2  # korean
    JP = 3  # japanese
    SP = 4  # special chars


UNICODE_RANGES = {
    Lang.EN: [(0x0020, 0x007E)],  # Basic Latin (English + Punctuation)
    Lang.KR: [(0xAC00, 0xD7A3)],  # Hangul Syllables
    Lang.JP: [
        (0x3040, 0x309F),  # Hiragana
        (0x30A0, 0x30FF),  # Katakana
        (0x4E00, 0x9FFF),  # CJK Unified Ideographs (Kanji) - Common
        (0xFF00, 0xFFEF),  # Halfwidth and Fullwidth Forms (some punctuation)
    ],
    Lang.SP: [
        (0x0021, 0x002F),
        (0x003A, 0x0040),
        (0x005B, 0x0060),
        (0x007B, 0x007E),  # Punctuation
        (0x2000, 0x206F),  # General Punctuation
        (0x3000, 0x303F),  # CJK Symbols and Punctuation
    ],
}


class BBox(tuple):
    def __new__(cls, x1: float, y1: float, x2: float, y2: float):
        # 간단히 float으로 변환하여 숫자임을 확인
        try:
            x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
        except (ValueError, TypeError):
            raise TypeError("Bounding box coordinates must be numbers.")

        # x1 > x2 또는 y1 > y2이면 순서를 바꿔주는 로직.
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1

        return super().__new__(cls, (x1, y1, x2, y2))

    @property
    def x1(self) -> float:
        return self[0]

    @property
    def y1(self) -> float:
        return self[1]

    @property
    def x2(self) -> float:
        return self[2]

    @property
    def y2(self) -> float:
        return self[3]

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    def expand(self, margin: int, img_size: tuple[int, int]) -> "BBox":
        w, h = img_size
        x1 = max(0, self.x1 - margin)
        y1 = max(0, self.y1 - margin)
        x2 = min(w, self.x2 + margin)
        y2 = min(h, self.y2 + margin)
        return BBox(x1, y1, x2, y2)

    def intersects(self, other: "BBox") -> bool:
        return not (
            self.x2 < other.x1
            or self.x1 > other.x2
            or self.y2 < other.y1
            or self.y1 > other.y2
        )

    def union(self, other: "BBox") -> "BBox":
        x1 = min(self.x1, other.x1)
        y1 = min(self.y1, other.y1)
        x2 = max(self.x2, other.x2)
        y2 = max(self.y2, other.y2)
        return BBox(x1, y1, x2, y2)


class TextedImage:
    def __init__(
        self,
        orig: torch.Tensor,
        timg: torch.Tensor,
        mask: torch.Tensor,
        bboxes: list[BBox],
        index: int,
    ):
        # Ensure all tensors are on the same device
        assert (
            timg.device == orig.device and mask.device == orig.device
        ), "All tensors must be on the same device"

        self.orig = orig  # Original image (C, H, W)
        self.timg = timg  # Image with text (C, H, W)
        self.mask = mask  # Pixel-wise mask (1, H, W), 1.0 for text, 0.0 for background
        self.bboxes = bboxes  # List of BBox objects
        self.index = index  # Unique index for this image in the dataset

        # Store original dimensions for reference
        _, self.height, self.width = orig.shape

    def to(self, device):
        self.orig = self.orig.to(device)
        self.timg = self.timg.to(device)
        self.mask = self.mask.to(device)
        return self

    def merge_bboxes_with_margin(self, margin: int, img_size: tuple[int, int]):
        # Expand bboxes
        expanded_bboxes = [bbox.expand(margin, img_size) for bbox in self.bboxes]

        new_bboxes = []
        # Keep track of which original bboxes have been merged
        merged_indices = set()

        for i in range(len(expanded_bboxes)):
            if i in merged_indices:
                continue

            current_union_bbox = self.bboxes[i]  # Start with the original bbox
            current_group_indices = {i}

            # Find all bboxes that intersect with the current expanded bbox group
            # This is a simple iterative approach. More efficient algorithms exist (e.g., using intervals).
            changed = True
            while changed:
                changed = False
                for j in range(len(expanded_bboxes)):
                    if j not in merged_indices and i != j:
                        # Check if expanded_bboxes[j] intersects with the expanded version of the current union
                        # Need to re-calculate the expanded union bbox in each iteration if we want to be precise
                        # Simpler: check if expanded_bboxes[j] intersects with *any* expanded bbox in the current group
                        intersects_with_group = False
                        for k in current_group_indices:
                            if expanded_bboxes[j].intersects(expanded_bboxes[k]):
                                intersects_with_group = True
                                break

                        if intersects_with_group:
                            # Merge original bbox[j] into the current union
                            current_union_bbox = current_union_bbox.union(
                                self.bboxes[j]
                            )
                            current_group_indices.add(j)
                            merged_indices.add(j)
                            changed = (
                                True  # Found a new bbox to merge, need another pass
                            )

            # Add the final merged bbox (based on original coordinates)
            new_bboxes.append(current_union_bbox)
            merged_indices.add(i)  # Mark the starting bbox as merged

        return new_bboxes

    @staticmethod
    def pil_to_img(pil: Image.Image, device) -> torch.Tensor:
        if pil.mode != "RGB":
            pil = pil.convert("RGB")
        return VTF.to_tensor(pil).to(device)

    @staticmethod
    def img_to_pil(img: torch.Tensor) -> Image.Image:
        img = img.detach().cpu()
        img = img.mul(255).byte().permute(1, 2, 0)
        return Image.fromarray(img.numpy(), "RGB")

    @staticmethod
    def resize_and_pad(img: torch.Tensor, target_size: tuple[int, int]) -> torch.Tensor:
        c, h, w = img.shape
        target_h, target_w = target_size

        # Calculate aspect ratios
        original_aspect = w / h
        target_aspect = target_w / target_h

        if original_aspect > target_aspect:
            # Image is wider than target, scale based on width
            new_w = target_w
            new_h = int(target_w / original_aspect)
        else:
            # Image is taller than target or same aspect, scale based on height
            new_h = target_h
            new_w = int(target_h * original_aspect)

        # Resize
        resized_tensor = VTF.resize(img, (new_h, new_w))

        # Calculate padding
        pad_left = (target_w - new_w) // 2
        pad_right = target_w - new_w - pad_left
        pad_top = (target_h - new_h) // 2
        pad_bottom = target_h - new_h - pad_top

        # zero pad
        padded_tensor = VTF.pad(
            resized_tensor, (pad_left, pad_top, pad_right, pad_bottom)
        )

        return padded_tensor

    @staticmethod
    def crop_with_bbox(img: torch.Tensor, bbox: "BBox") -> torch.Tensor:
        """Crops a tensor image (C, H, W) based on a BBox (x1, y1, x2, y2)."""
        x1, y1, x2, y2 = bbox
        # Ensure crop coordinates are within image bounds
        _, h, w = img.shape
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        # Calculate width and height for crop
        crop_width = x2 - x1
        crop_height = y2 - y1

        if crop_width <= 0 or crop_height <= 0:
            # Return a small dummy tensor or handle error
            print(f"Warning: Invalid crop bbox {bbox}. Returning empty tensor.")
            return torch.empty(
                img.shape[0], 1, 1, device=img.device
            )  # Return a minimal tensor

        # F.crop takes (top, left, height, width)
        cropped_tensor = VTF.crop(img, y1, x1, crop_height, crop_width)
        return cropped_tensor

    @staticmethod
    def paste_cropped_img(
        img: torch.Tensor, cropped_img: torch.Tensor, position: "BBox"
    ) -> torch.Tensor:
        """
        Pastes a crop_tensor into a base_image_tensor at the location specified by bbox.
        Handles cases where the bbox is partially outside the base image bounds.
        Assumes crop_tensor is the result of apply_crop on a source image
        that had the same dimensions as base_image_tensor.
        """
        # Create a copy to avoid modifying the original tensor in place if it's needed elsewhere
        output_image = img.clone()

        _, img_h, img_w = output_image.shape
        _, crop_h, crop_w = cropped_img.shape
        x1, y1, x2, y2 = position

        # Calculate the region in the base image where pasting occurs (clamped to bounds)
        paste_x1 = max(0, x1)
        paste_y1 = max(0, y1)
        paste_x2 = min(img_w, x2)
        paste_y2 = min(img_h, y2)

        # Calculate the corresponding region in the crop_tensor
        # This accounts for the part of the bbox that was outside the original image
        crop_x1_src = max(0, -x1)
        crop_y1_src = max(0, -y1)
        crop_x2_src = crop_w - max(0, x2 - img_w)
        crop_y2_src = crop_h - max(0, y2 - img_h)

        # Perform the paste operation
        output_image[:, paste_y1:paste_y2, paste_x1:paste_x2] = cropped_img[
            :, crop_y1_src:crop_y2_src, crop_x1_src:crop_x2_src
        ]

        return output_image

    @staticmethod
    def center_crop(
        image_tensor: torch.Tensor,
        bbox: "BBox",
        output_size: tuple,
    ) -> torch.Tensor:
        """
        Crops a region centered around the bbox to output_size (H_o, W_o).
        If the ideal window goes beyond image boundaries, the window is shifted
        to stay within bounds. If the image is smaller than output_size in any dimension,
        it falls back to padding the cropped valid region.
        """
        _, img_h, img_w = image_tensor.shape
        output_h, output_w = output_size

        bbox_x1, bbox_y1, bbox_x2, bbox_y2 = bbox

        bbox_center_x = (bbox_x1 + bbox_x2) // 2
        bbox_center_y = (bbox_y1 + bbox_y2) // 2

        # Case 1: Image is large enough to contain the output window - perform shifting
        if img_w >= output_w and img_h >= output_h:
            # Calculate ideal top-left corner based on center
            ideal_x1 = bbox_center_x - output_w // 2
            ideal_y1 = bbox_center_y - output_h // 2

            # Calculate bounds for the top-left corner
            min_x1 = 0
            min_y1 = 0
            max_x1 = img_w - output_w
            max_y1 = img_h - output_h

            # Clamp the ideal top-left corner within bounds (shifting)
            final_x1 = max(min_x1, ideal_x1)
            final_x1 = min(final_x1, max_x1)  # Clamp again from the right side

            final_y1 = max(min_y1, ideal_y1)
            final_y1 = min(final_y1, max_y1)  # Clamp again from the bottom side

            # Perform crop using the calculated final top-left corner and output size
            # F.crop(img, top, left, height, width)
            cropped_tensor = VTF.crop(
                image_tensor, final_y1, final_x1, output_h, output_w
            )

            # No padding or resizing needed here, as F.crop directly gives the output_size

            return cropped_tensor

        # Case 2: Image is smaller than output size in at least one dimension - must pad
        else:
            print(
                f"Warning: Image size ({img_w}x{img_h}) is smaller than output size ({output_w}x{output_h}). Padding instead of shifting."
            )

            # Calculate the ideal crop window coordinates
            crop_x1 = bbox_center_x - output_w // 2
            crop_y1 = bbox_center_y - output_h // 2
            crop_x2 = crop_x1 + output_w
            crop_y2 = crop_y1 + output_h

            # Calculate padding needed if this ideal window were applied
            pad_left = max(0, -crop_x1)
            pad_top = max(0, -crop_y1)
            pad_right = max(0, crop_x2 - img_w)
            pad_bottom = max(0, crop_y2 - img_h)

            # Calculate the actual region to crop from the image (within bounds)
            crop_x1_adj = max(0, crop_x1)
            crop_y1_adj = max(0, crop_y1)
            crop_x2_adj = min(img_w, crop_x2)
            crop_y2_adj = min(img_h, crop_y2)

            crop_width_adj = crop_x2_adj - crop_x1_adj
            crop_height_adj = crop_y2_adj - crop_y1_adj

            # Perform the crop on the valid region
            cropped_tensor = VTF.crop(
                image_tensor, crop_y1_adj, crop_x1_adj, crop_height_adj, crop_width_adj
            )

            # Pad the cropped tensor to the target output size
            # The padding values calculated earlier [pad_left, pad_top, pad_right, pad_bottom]
            # are relative to the original ideal crop window (crop_x1, crop_y1).
            # When padding the *cropped_tensor* (which starts at crop_x1_adj, crop_y1_adj),
            # the padding needed is [crop_x1_adj - crop_x1, crop_y1_adj - crop_y1, crop_x2 - crop_x2_adj, crop_y2 - crop_y2_adj]
            # which simplifies to [pad_left, pad_top, pad_right, pad_bottom].
            padded_tensor = VTF.pad(
                cropped_tensor, (pad_left, pad_top, pad_right, pad_bottom)
            )

            return padded_tensor
