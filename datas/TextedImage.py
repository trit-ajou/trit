import torch
import torchvision.transforms.functional as VTF
import matplotlib.pyplot as plt
from torch import Tensor as img_tensor
from PIL import Image, ImageDraw

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
        _, H, W = self.orig.shape
        return H, W

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

    def split_margin_crop(self, margin: int):
        """margin crop 범위 내에 다른 텍스트 있을 일 없음.
        _bbox는 원래 이미지에 붙여넣을 때 잘라내야 할 정확한 크기를 저장함."""
        texted_images: list[TextedImage] = []
        for bbox in self.bboxes:
            orig, _bbox = TextedImage._margin_crop(self.orig, bbox, margin, fill=1.0)
            timg, _ = TextedImage._margin_crop(self.timg, bbox, margin, fill=1.0)
            mask, _ = TextedImage._margin_crop(self.mask, bbox, margin, fill=0.0)
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
        """Note: this function does not create new `TextedImage` obejct but modifies itself."""
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
            new_h = int(TARGET_W / original_aspect)
            new_w = TARGET_W
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
            x1, y1, x2, y2 = bbox
            x1 *= scale_w
            x2 *= scale_w
            y1 *= scale_h
            y2 *= scale_h
            new_bboxes.append(
                BBox(
                    int(x1 + pad_left),
                    int(y1 + pad_top),
                    int(x2 + pad_left),
                    int(y2 + pad_top),
                )
            )
        self.bboxes = new_bboxes

    @staticmethod
    def _alpha_blend(
        background_img: img_tensor,
        bbox: BBox,  # bbox가 확실히 이미지 내부에 있음을 가정. 따로 clamping 안함.
        img: img_tensor,  # bbox 크기의 tensor
        alpha: img_tensor,  # bbox 크기의 tensor
    ):
        # Calc alpha blended region
        cropped_background = background_img[bbox.slice]
        blended_img = img * alpha + (cropped_background * (1.0 - alpha))
        # Paste
        output_image = background_img.clone()  # 원본수정방지
        output_image[bbox.slice] = blended_img
        return output_image

    @staticmethod
    def _margin_crop(img: img_tensor, bbox: BBox, margin: int, fill: float = 0.0):
        _, H, W = img.shape
        expanded_bbox = bbox._unsafe_expand(margin)
        crop_bbox = bbox._safe_expand(margin, (H, W))
        # calc pad
        pad_left = max(0, -expanded_bbox.x1)
        pad_top = max(0, -expanded_bbox.y1)
        pad_right = max(0, expanded_bbox.x2 - W)
        pad_bottom = max(0, expanded_bbox.y2 - H)
        # Apply padding if needed
        img = img[crop_bbox.slice]
        img = VTF.pad(img, (pad_left, pad_top, pad_right, pad_bottom), fill=fill)
        return img, bbox.coord_trans(expanded_bbox.x1, expanded_bbox.y1)

    @staticmethod
    def _center_crop(
        img: img_tensor,
        bbox: BBox,
        size: tuple[int, int],
    ):
        _, H, W = img.shape
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
        if crop_x1 + output_w > W:
            crop_x1 = W - output_w
        if crop_y1 + output_h > H:
            crop_y1 = H - output_h
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