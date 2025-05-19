import torch
from torch import Tensor as img_tensor
import torchvision.transforms.functional as VTF
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

from .Utils import BBox


class TextedImage:
    def __init__(
        self,
        orig: img_tensor,
        timg: img_tensor,
        mask: img_tensor,
        bboxes: list[BBox],
    ):
        self.orig = orig  # Original image (C, H, W)
        self.timg = timg  # Image with text (C, H, W)
        self.mask = mask  # Binay pixel-wise mask (1, H, W)
        self.bboxes = bboxes

    @property
    def img_size(self):
        _, h, w = self.orig.shape
        return h, w

    def merge_bboxes_with_margin(self, margin: int):
        expanded_bboxes = [
            bbox._safe_expand(margin, self.img_size) for bbox in self.bboxes
        ]
        new_bboxes = []
        # Keep track of which original bboxes have been merged
        merged_indices = set()
        for i in range(len(expanded_bboxes)):
            if i in merged_indices:
                continue
            merged_bbox = self.bboxes[i]
            current_group_indices = {i}
            # Find all bboxes that intersect with the current expanded bbox group
            changed = True
            while changed:
                changed = False
                for j in range(len(expanded_bboxes)):
                    if j not in merged_indices and i != j:
                        intersects_with_group = False
                        for k in current_group_indices:
                            if expanded_bboxes[j].intersects(expanded_bboxes[k]):
                                intersects_with_group = True
                                break
                        if intersects_with_group:
                            # Merge original bbox[j] into the current union
                            merged_bbox = merged_bbox.union(self.bboxes[j])
                            current_group_indices.add(j)
                            merged_indices.add(j)
                            changed = True
            new_bboxes.append(merged_bbox)
            merged_indices.add(i)
        self.bboxes = new_bboxes

    def split_margin_crop(self, margin=0):
        """margin crop 범위 내에 다른 텍스트 있을 일 없음.
        _bbox는 원래 이미지에 붙여넣을 때 잘라내야 할 정확한 크기를 저장함."""
        expanded_bboxes = [bbox._unsafe_expand(margin) for bbox in self.bboxes]
        texted_images: list[TextedImage] = []
        for bbox in expanded_bboxes:
            orig = TextedImage._margin_crop(self.orig, bbox, margin)
            timg = TextedImage._margin_crop(self.timg, bbox, margin)
            mask = TextedImage._margin_crop(self.mask, bbox, margin)
            _bbox = BBox(margin, margin, bbox.width - margin, bbox.height - margin)
            texted_images.append(TextedImage(orig, timg, mask, [_bbox]))
        return texted_images

    def split_center_crop(self, size: tuple[int, int]):
        """center crop 범위 내에 다른 텍스트 있을 수 있음.
        _bbox는 원래 이미지에 붙여넣을 때 잘라내야 할 정확한 크기를 저장함."""
        texted_images: list[TextedImage] = []
        for bbox in self.bboxes:
            orig = self.timg.clone()
            orig[bbox.slice] = self.orig[bbox.slice]
            orig, _bbox = TextedImage._center_crop(orig, bbox, size)
            timg, _ = TextedImage._center_crop(self.timg, bbox, size)
            mask = torch.zeros_like(self.mask)
            mask[bbox.slice] = self.mask[bbox.slice]
            mask, _ = TextedImage._center_crop(mask, bbox, size)
            texted_images.append(TextedImage(orig, timg, mask, [_bbox]))
        return texted_images

    def merge_cropped(self, cropped_texted_images: list["TextedImage"]):
        for bbox, cropped_texted_image in zip(self.bboxes, cropped_texted_images):
            _bbox = cropped_texted_image.bboxes[0]
            if cropped_texted_image.orig is not None:
                self.orig[bbox.slice] = cropped_texted_image.orig[_bbox.slice]
            if cropped_texted_image.timg is not None:
                self.timg[bbox.slice] = cropped_texted_image.timg[_bbox.slice]
            if cropped_texted_image.mask is not None:
                self.mask[bbox.slice] = cropped_texted_image.mask[_bbox.slice]

    @staticmethod
    def _resize(img: img_tensor, size: tuple[int, int]) -> img_tensor:
        _, orig_h, orig_w = img.shape
        target_h, target_w = size
        # Calculate aspect ratios
        original_aspect = orig_w / orig_h
        target_aspect = target_w / target_h
        # Resize
        if original_aspect > target_aspect:
            new_h = int(target_w / original_aspect)
            new_w = target_w
        else:
            new_h = target_h
            new_w = int(target_h * original_aspect)
        img = VTF.resize(img, (new_h, new_w))
        # pad
        pad_left = (target_w - new_w) // 2
        pad_right = target_w - new_w - pad_left
        pad_top = (target_h - new_h) // 2
        pad_bottom = target_h - new_h - pad_top
        img = VTF.pad(img, (pad_left, pad_top, pad_right, pad_bottom))
        return img

    @staticmethod
    def _alpha_blend(
        background_img: img_tensor,
        bbox: BBox,  # bbox가 확실히 이미지 내부에 있음을 가정. 따로 clamping 안함.
        img: img_tensor,  # bbox 크기의 tensor
        alpha: img_tensor,  # bbox 크기의 tensor
    ) -> img_tensor:
        # Calc alpha blended region
        cropped_background = background_img[bbox.slice]
        blended_img = img * alpha + (cropped_background * (1.0 - alpha))
        # Paste
        output_image = background_img.clone()  # 원본수정방지
        output_image[bbox.slice] = blended_img
        return output_image

    @staticmethod
    def _margin_crop(
        img: img_tensor,
        bbox: BBox,
        margin: int,
    ) -> img_tensor:
        _, h, w = img.shape
        expanded_bbox = bbox._unsafe_expand(margin)
        # 2. Determine the part of this target window that actually overlaps with the image
        src_x1 = max(0, expanded_bbox.x1)
        src_y1 = max(0, expanded_bbox.y1)
        src_x2 = min(w, expanded_bbox.x2)
        src_y2 = min(h, expanded_bbox.y2)
        # 3. Crop this overlapping part from the image
        cropped_from_img = img[:, src_y1:src_y2, src_x1:src_x2]
        # 4. Calculate padding needed to make `cropped_from_img` the `target_crop_size`
        pad_left = max(0, -expanded_bbox.x1)
        pad_top = max(0, -expanded_bbox.y1)
        pad_right = max(0, expanded_bbox.x2 - w)
        pad_bottom = max(0, expanded_bbox.y2 - h)
        # Apply padding if needed
        padded_crop = VTF.pad(
            cropped_from_img, (pad_left, pad_top, pad_right, pad_bottom)
        )
        return padded_crop

    @staticmethod
    def _center_crop(
        img: img_tensor,
        bbox: BBox,
        size: tuple[int, int],
    ) -> tuple[img_tensor, BBox]:
        _, h, w = img.shape
        output_h, output_w = size
        bbox_center_x, bbox_center_y = bbox.center
        # 1. bbox 중심으로 이상적인 crop 영역의 좌상단(x1, y1) 계산
        crop_x1 = bbox_center_x - output_w // 2
        crop_y1 = bbox_center_y - output_h // 2
        # 2. crop 영역이 이미지 경계를 벗어나지 않도록 조정 (평행 이동)
        # 왼쪽/위쪽 테두리 기준으로 밀어넣기
        if crop_x1 < 0:
            crop_x1 = 0
        if crop_y1 < 0:
            crop_y1 = 0
        # 오른쪽/아래쪽 테두리 기준으로 밀어넣기
        if crop_x1 + output_w > w:
            crop_x1 = w - output_w
        if crop_y1 + output_h > h:
            crop_y1 = h - output_h
        # 3. slice
        slice_bbox = BBox(crop_x1, crop_y1, crop_x1 + output_w, crop_y1 + output_h)
        _bbox = bbox.coord_trans(crop_x1, crop_y1)
        return img[slice_bbox.slice], _bbox

    def _to_pil(self):
        orig = self.orig.detach().cpu().mul(255).byte().permute(1, 2, 0).numpy()
        timg = self.timg.detach().cpu().mul(255).byte().permute(1, 2, 0).numpy()
        mask = self.mask.detach().cpu().mul(255).byte().permute(1, 2, 0).numpy()
        orig = Image.fromarray(orig, "RGB")
        timg = Image.fromarray(timg, "RGB")
        mask = Image.fromarray(mask.squeeze(), "L")
        return orig, timg, mask

    def visualize(self, dir=".", filename="test.png"):
        orig, timg, mask = self._to_pil()
        draw = ImageDraw.Draw(timg)
        for bbox in self.bboxes:
            draw.rectangle(
                [(bbox.x1, bbox.y1), (bbox.x2, bbox.y2)],
                outline="red",
                width=1,
            )
        _, axes = plt.subplots(1, 3, figsize=(18, 6))
        axes[0].imshow(orig)
        axes[0].axis("off")
        axes[1].imshow(timg)
        axes[1].axis("off")
        axes[2].imshow(mask, cmap="gray")
        axes[2].axis("off")
        plt.tight_layout()
        plt.savefig(dir + "/" + filename)
        plt.close()
