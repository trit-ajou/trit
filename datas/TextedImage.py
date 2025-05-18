from torch import Tensor
import torchvision.transforms.functional as VTF
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

from .Utils import BBox


class TextedImage:
    def __init__(
        self,
        background: Tensor,
        texted: Tensor,
        mask: Tensor,
        bboxes: list[BBox],
    ):
        # Ensure all tensors are on the same device
        assert (
            texted.device == background.device and mask.device == background.device
        ), "All tensors must be on the same device"

        self.background = background  # Original image (C, H, W)
        self.texted = texted  # Image with text (C, H, W)
        self.mask = mask  # Pixel-wise mask (1, H, W), 1.0 for text, 0.0 for background
        self.bboxes = bboxes  # List of BBox objects

        # Store original dimensions for reference
        _, self.height, self.width = background.shape

    @property
    def size(self):
        return self.height, self.width

    def to(self, device):
        self.background = self.background.to(device)
        self.texted = self.texted.to(device)
        self.mask = self.mask.to(device)
        return self

    def merge_bboxes_with_margin(self, margin: int):
        expanded_bboxes = [
            bbox.expand(margin, (self.height, self.width)) for bbox in self.bboxes
        ]

        new_bboxes = []
        # Keep track of which original bboxes have been merged
        merged_indices = set()

        for i in range(len(expanded_bboxes)):
            if i in merged_indices:
                continue

            current_union_bbox = self.bboxes[i]
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
    def img2pil(img: Tensor) -> Image.Image:
        img = img.detach().cpu()
        img = img.mul(255).byte().permute(1, 2, 0)
        return Image.fromarray(img.numpy(), "RGB")

    @staticmethod
    def resize_keep_aspect(img: Tensor, target_size: tuple[int, int]) -> Tensor:
        _, original_h, original_w = img.shape
        target_h, target_w = target_size
        # Calculate aspect ratios
        original_aspect = original_w / original_h
        target_aspect = target_w / target_h
        # Image is wider than target, scale based on width
        if original_aspect > target_aspect:
            new_w = target_w
            new_h = int(target_w / original_aspect)
        # Image is taller than target or same aspect, scale based on height
        else:
            new_h = target_h
            new_w = int(target_h * original_aspect)
        # Resize
        resized = VTF.resize(img, (new_h, new_w))
        # Calculate padding
        pad_left = (target_w - new_w) // 2
        pad_right = target_w - new_w - pad_left
        pad_top = (target_h - new_h) // 2
        pad_bottom = target_h - new_h - pad_top
        # zero pad
        return VTF.pad(resized, (pad_left, pad_top, pad_right, pad_bottom))

    @staticmethod
    def alpha_blend_at_bbox(
        background_img: Tensor,
        bbox: BBox,  # bbox가 확실히 이미지 내부에 있음을 가정. 따로 clamping 안함.
        cropped_img: Tensor,  # bbox 크기의 rgb or monochrome image tensor
        cropped_alpha: Tensor = None,  # 이거 None이면 알파 블렌딩 대신 그냥 붙여넣기(alpha=1)
    ) -> Tensor:
        # Calc alpha blended region
        if cropped_alpha is None:
            blended_region = cropped_img
        else:
            cropped_img = background_img[bbox.slice]
            blended_region = (cropped_img * cropped_alpha) + (
                cropped_img * (1.0 - cropped_alpha)
            )
        # Paste
        output_image = background_img.clone()  # 원본수정방지
        output_image[bbox.slice] = blended_region

        return output_image

    @staticmethod
    def center_crop(
        image_tensor: Tensor,
        bbox: BBox,
        size: tuple[int, int],
    ) -> Tensor:
        _, img_h, img_w = image_tensor.shape
        output_h, output_w = size

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
            final_x1 = min(final_x1, max_x1)

            final_y1 = max(min_y1, ideal_y1)
            final_y1 = min(final_y1, max_y1)

            # Perform crop using the calculated final top-left corner and output size
            cropped_tensor = VTF.crop(
                image_tensor, final_y1, final_x1, output_h, output_w
            )

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
            padded_tensor = VTF.pad(
                cropped_tensor, (pad_left, pad_top, pad_right, pad_bottom)
            )

            return padded_tensor

    def visualize(self, filename="test.png"):
        pil_orig = TextedImage.img2pil(self.background)
        pil_timg = TextedImage.img2pil(self.texted)
        pil_mask = TextedImage.img2pil(self.mask)

        # Prepare texted image with bboxes
        pil_timg_with_bboxes = pil_timg.copy()
        draw = ImageDraw.Draw(pil_timg_with_bboxes)

        for bbox in self.bboxes:
            draw.rectangle(
                [(bbox.x1, bbox.y1), (bbox.x2, bbox.y2)],
                outline="red",
                width=1,
            )

        _, axes = plt.subplots(1, 3, figsize=(18, 6))
        axes[0].imshow(pil_orig)
        axes[0].axis("off")
        axes[1].imshow(pil_timg_with_bboxes)
        axes[1].axis("off")
        axes[2].imshow(pil_mask, cmap="gray" if pil_mask.mode == "L" else None)
        axes[2].axis("off")

        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
