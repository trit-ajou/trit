import torch
import torchvision.transforms.functional as VTF
import matplotlib.pyplot as plt
import numpy as np
import os
from torch import Tensor as img_tensor
from PIL import Image, ImageDraw
from torch import Tensor as img_tensor
from typing import List, Optional, Tuple

from .Utils import BBox


class CharInfo:
    """
    개별 문자의 기하학적 정보 및 메타데이터를 저장하는 클래스.
    CRAFT 모델의 Ground Truth 생성에 사용됩니다.

    Attributes:
        polygon (np.ndarray): 문자의 경계 상자를 나타내는 4개 꼭짓점 좌표 (4x2 NumPy 배열).
        char_content (str): 실제 문자 값 (e.g., 'A', '가').
        word_id (int): 이 문자가 속한 단어(또는 텍스트 블록)의 고유 ID.
    """

    def __init__(self, polygon: np.ndarray, char_content: str, word_id: int):
        self.polygon = polygon
        self.char_content = char_content
        self.word_id = word_id


class TextedImage:
    """
    원본 이미지에 텍스트가 합성된 결과물과 관련 메타데이터를 관리하는 핵심 클래스.
    이미지 텐서, 바운딩 박스, CRAFT 학습용 GT 데이터 등을 포함합니다.
    """

    def __init__(
            self,
            orig: img_tensor,
            timg: img_tensor,
            mask: img_tensor,
            bboxes: List[BBox],
            all_char_infos: Optional[List[CharInfo]] = None,
            region_score_map: Optional[torch.Tensor] = None,
            affinity_score_map: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            orig (img_tensor): 원본 이미지 텐서 (C, H, W).
            timg (img_tensor): 텍스트가 합성된 이미지 텐서 (C, H, W).
            mask (img_tensor): 텍스트 영역을 나타내는 이진 마스크 텐서 (1, H, W).
            bboxes (List[BBox]): 텍스트 블록 또는 단어 단위의 바운딩 박스 리스트.
            all_char_infos (Optional[List[CharInfo]]): CRAFT용 문자 단위 정보 리스트.
            region_score_map (Optional[torch.Tensor]): CRAFT용 Region Score Map (1, H/2, W/2).
            affinity_score_map (Optional[torch.Tensor]): CRAFT용 Affinity Score Map (1, H/2, W/2).
        """
        self.orig = orig
        self.timg = timg
        self.mask = mask
        self.bboxes = bboxes
        self.all_char_infos = all_char_infos
        self.region_score_map = region_score_map
        self.affinity_score_map = affinity_score_map

    @property
    def img_size(self) -> Tuple[int, int]:
        """이미지의 높이(H)와 너비(W)를 반환합니다."""
        if self.orig is not None:
            _, H, W = self.orig.shape
        elif self.timg is not None:
            _, H, W = self.timg.shape
        else:
            return 0, 0
        return H, W

    def merge_bboxes_with_margin(self, margin: int):
        """
        여백(margin)을 기준으로 서로 교차하는 바운딩 박스들을 병합합니다.
        근접한 텍스트 덩어리들을 하나의 큰 박스로 묶는 데 사용됩니다.
        """
        if not self.bboxes:
            return

        expanded_bboxes = [bbox._safe_expand(margin, self.img_size) for bbox in self.bboxes]

        new_bboxes = []
        merged_indices = set()

        for i in range(len(expanded_bboxes)):
            if i in merged_indices:
                continue

            current_merged_bbox = self.bboxes[i]
            current_group_indices = {i}

            changed_in_iteration = True
            while changed_in_iteration:
                changed_in_iteration = False
                for j in range(len(expanded_bboxes)):
                    if j in merged_indices or j in current_group_indices:
                        continue

                    for k in current_group_indices:
                        if expanded_bboxes[j].intersects(expanded_bboxes[k]):
                            current_merged_bbox = current_merged_bbox.union(self.bboxes[j])
                            current_group_indices.add(j)
                            changed_in_iteration = True
                            break

            new_bboxes.append(current_merged_bbox)
            merged_indices.update(current_group_indices)

        self.bboxes = new_bboxes

    def split_margin_crop(self, margin: int) -> List['TextedImage']:
        """각 바운딩 박스를 지정된 여백만큼 확장하여 여러 개의 TextedImage 객체로 분리합니다."""
        texted_images: list[TextedImage] = []
        for bbox in self.bboxes:
            orig, _bbox = TextedImage._margin_crop(self.orig, bbox, margin, fill=1.0)
            timg, _ = TextedImage._margin_crop(self.timg, bbox, margin, fill=1.0)
            mask, _ = TextedImage._margin_crop(self.mask, bbox, margin, fill=0.0)
            texted_images.append(TextedImage(orig, timg, mask, [_bbox]))
        return texted_images

    def split_center_crop(self, size: tuple[int, int]) -> List['TextedImage']:
        """각 바운딩 박스의 중심을 기준으로 지정된 크기로 크롭하여 여러 개의 TextedImage 객체로 분리합니다."""
        texted_images: list[TextedImage] = []
        for bbox in self.bboxes:
            orig_bg = self.timg.clone()
            orig_bg[bbox.slice] = self.orig[bbox.slice]
            orig, _bbox = TextedImage._center_crop(orig_bg, bbox, size)

            timg, _ = TextedImage._center_crop(self.timg, bbox, size)

            mask_focused = torch.zeros_like(self.mask)
            mask_focused[bbox.slice] = self.mask[bbox.slice]
            mask, _ = TextedImage._center_crop(mask_focused, bbox, size)

            texted_images.append(TextedImage(orig, timg, mask, [_bbox]))
        return texted_images

    def merge_cropped(self, cropped_texted_images: list["TextedImage"]):
        for i, (bbox, cropped_texted_image) in enumerate(zip(self.bboxes, cropped_texted_images)):
            print(f"DEBUG merge_cropped: Processing bbox {i}")
            print(f"DEBUG merge_cropped: bbox = {bbox} (height={bbox.height}, width={bbox.width})")

            _bbox = cropped_texted_image.bboxes[0]
            print(f"DEBUG merge_cropped: _bbox = {_bbox}")
            print(f"DEBUG merge_cropped: cropped_texted_image.orig.shape before slice = {cropped_texted_image.orig.shape}")

            cropped_texted_image.orig = cropped_texted_image.orig[_bbox.slice]
            cropped_texted_image.timg = cropped_texted_image.timg[_bbox.slice]
            cropped_texted_image.mask = cropped_texted_image.mask[_bbox.slice]

            print(f"DEBUG merge_cropped: cropped_texted_image.orig.shape after slice = {cropped_texted_image.orig.shape}")
            print(f"DEBUG merge_cropped: About to call _resize with size = ({bbox.height}, {bbox.width})")

            cropped_texted_image._resize((bbox.height, bbox.width))
            self.orig[bbox.slice] = cropped_texted_image.orig
            self.timg[bbox.slice] = cropped_texted_image.timg
            self.mask[bbox.slice] = cropped_texted_image.mask

    def _resize(self, size: tuple[int, int]):
        """객체 자신을 지정된 크기로 리사이즈하고 패딩합니다. (인플레이스 연산)"""
        _, H, W = self.orig.shape
        TARGET_H, TARGET_W = size

        # DEBUG: Print values before potential division by zero
        print(f"DEBUG _resize: orig.shape={self.orig.shape}, H={H}, W={W}")
        print(f"DEBUG _resize: target size={size}, TARGET_H={TARGET_H}, TARGET_W={TARGET_W}")

        # Calculate aspect ratios
        if H == 0:
            print(f"ERROR: H is zero! Cannot calculate original_aspect")
            raise ValueError(f"Height is zero: H={H}, orig.shape={self.orig.shape}")
        if W == 0:
            print(f"ERROR: W is zero! Cannot calculate original_aspect")
            raise ValueError(f"Width is zero: W={W}, orig.shape={self.orig.shape}")

        original_aspect = W / H
        target_aspect = TARGET_W / TARGET_H
        print(f"DEBUG _resize: original_aspect={original_aspect}, target_aspect={target_aspect}")

        # Resize
        if original_aspect > target_aspect:
            new_w = TARGET_W
            new_h = int(TARGET_W / original_aspect)
        else:
            new_h = TARGET_H
            new_w = int(TARGET_H * original_aspect)

        print(f"DEBUG _resize: new_h={new_h}, new_w={new_w}")

        self.orig = VTF.resize(self.orig, (new_h, new_w))
        self.timg = VTF.resize(self.timg, (new_h, new_w))
        self.mask = VTF.resize(self.mask, (new_h, new_w))
        # pad

        pad_left = (TARGET_W - new_w) // 2
        pad_right = TARGET_W - new_w - pad_left
        pad_top = (TARGET_H - new_h) // 2
        pad_bottom = TARGET_H - new_h - pad_top

        self.orig = VTF.pad(
            self.orig, (pad_left, pad_top, pad_right, pad_bottom), fill=1
        )
        self.timg = VTF.pad(
            self.timg, (pad_left, pad_top, pad_right, pad_bottom), fill=1
        )
        self.mask = VTF.pad(
            self.mask, (pad_left, pad_top, pad_right, pad_bottom), fill=0
        )
        # resize bboxes
        new_bboxes: list[BBox] = []

        # DEBUG: Print values before potential division by zero
        print(f"DEBUG _resize: About to calculate scale_w = new_w / W = {new_w} / {W}")
        print(f"DEBUG _resize: About to calculate scale_h = new_h / H = {new_h} / {H}")

        if W == 0:
            print(f"ERROR: W is zero! Cannot calculate scale_w")
            print(f"ERROR: orig.shape={self.orig.shape}, new_w={new_w}, W={W}")
            raise ValueError(f"Width is zero when calculating scale_w: W={W}")
        if H == 0:
            print(f"ERROR: H is zero! Cannot calculate scale_h")
            print(f"ERROR: orig.shape={self.orig.shape}, new_h={new_h}, H={H}")
            raise ValueError(f"Height is zero when calculating scale_h: H={H}")


        scale_w = new_w / W
        scale_h = new_h / H
        for bbox in self.bboxes:
            new_bboxes.append(BBox(
                int(bbox.x1 * scale_w + pad_left),
                int(bbox.y1 * scale_h + pad_top),
                int(bbox.x2 * scale_w + pad_left),
                int(bbox.y2 * scale_h + pad_top),
            ))
        self.bboxes = new_bboxes

    @staticmethod
    def _alpha_blend(background_img: img_tensor, bbox: BBox, img: img_tensor, alpha: img_tensor) -> img_tensor:
        """지정된 bbox 영역에 전경 이미지를 알파 블렌딩합니다."""
        output_image = background_img.clone()
        cropped_background = background_img[bbox.slice]
        blended_img = img * alpha + cropped_background * (1.0 - alpha)
        output_image[bbox.slice] = blended_img
        return output_image

    @staticmethod

    def _margin_crop(img: img_tensor, bbox: BBox, margin: int, fill: float = 0.0):
        _, H, W = img.shape
        expanded_bbox = bbox._unsafe_expand(margin)
        crop_bbox = bbox._safe_expand(margin, (H, W))

        pad_left = max(0, -expanded_bbox.x1)
        pad_top = max(0, -expanded_bbox.y1)
        pad_right = max(0, expanded_bbox.x2 - W)
        pad_bottom = max(0, expanded_bbox.y2 - H)

        # Apply padding if needed
        img = img[crop_bbox.slice]
        img = VTF.pad(img, (pad_left, pad_top, pad_right, pad_bottom), fill=fill)
        return img, bbox.coord_trans(expanded_bbox.x1, expanded_bbox.y1)


    @staticmethod
    def _center_crop(img: img_tensor, bbox: BBox, size: tuple[int, int]) -> Tuple[img_tensor, BBox]:
        """BBox의 중심을 기준으로 지정된 size로 이미지를 크롭합니다."""
        _, H, W = img.shape
        output_h, output_w = size
        bbox_center_x, bbox_center_y = bbox.center

        crop_x1 = bbox_center_x - output_w // 2
        crop_y1 = bbox_center_y - output_h // 2

        crop_x1 = max(0, min(crop_x1, W - output_w))
        crop_y1 = max(0, min(crop_y1, H - output_h))

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
        # 출력 디렉토리가 없으면 생성
        os.makedirs(dir, exist_ok=True)
        orig, timg, mask = self._to_pil()
        draw = ImageDraw.Draw(timg)
        for bbox in self.bboxes:
            draw.rectangle([(bbox.x1, bbox.y1), (bbox.x2, bbox.y2)], outline="red", width=1)

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        axes[0].imshow(orig);
        axes[0].set_title("Original");
        axes[0].axis("off")
        axes[1].imshow(timg);
        axes[1].set_title("Texted + BBoxes");
        axes[1].axis("off")
        axes[2].imshow(mask, cmap="gray");
        axes[2].set_title("Mask");
        axes[2].axis("off")

        plt.tight_layout()
        plt.savefig(dir + "/" + filename)
        plt.close()
