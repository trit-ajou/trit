import torch
import torchvision.transforms.functional as VTF
import matplotlib.pyplot as plt
from torch import Tensor as img_tensor
from PIL import Image, ImageDraw

from .Utils import BBox
import json, glob, random, os
from torchvision.transforms.functional import to_pil_image, to_tensor
from PIL import Image
from typing import List, Optional, Tuple, Dict, Any
from concurrent.futures import ThreadPoolExecutor
import numpy as np

class CharInfo:
    def __init__(self, polygon: np.ndarray, char_content: str, word_id: int):
        """
        CRAFT GT 생성을 위한 문자 정보.
        polygon: 문자의 4개 꼭짓점 좌표 (4x2 NumPy array, [[x1,y1], [x2,y2], [x3,y3], [x4,y4]])
                 이미지 전체 좌표계 기준.
        char_content: 실제 문자 (e.g., 'A', '가')
        word_id: 이 문자가 속한 단어(또는 텍스트 덩어리)의 고유 ID
        """
        self.polygon = polygon
        self.char_content = char_content
        self.word_id = word_id
class TextedImage:
    def __init__(
        self,
        orig: img_tensor,
        timg: img_tensor,
        mask: img_tensor,
        bboxes: list[BBox],
        # --- CRAFT 학습용 추가 필드 (Optional) ---
        all_char_infos: Optional[List[CharInfo]] = None, # 이미지 내 모든 문자의 정보 리스트
        region_score_map: Optional[torch.Tensor] = None, # (1, H/2, W/2)
        affinity_score_map: Optional[torch.Tensor] = None, # (1, H/2, W/2)
    ):
        self.orig = orig  # Original image (C, H, W)
        self.timg = timg  # Image with text (C, H, W)
        self.mask = mask  # Binay pixel-wise mask (1, H, W)
        self.bboxes = bboxes

        self.all_char_infos = all_char_infos
        self.region_score_map = region_score_map
        self.affinity_score_map = affinity_score_map

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
            orig, _bbox = TextedImage._margin_crop(self.orig, bbox, margin)
            timg, _ = TextedImage._margin_crop(self.timg, bbox, margin)
            mask, _ = TextedImage._margin_crop(self.mask, bbox, margin)
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
            cropped_texted_image.orig = cropped_texted_image.orig[_bbox.slice]
            cropped_texted_image.timg = cropped_texted_image.timg[_bbox.slice]
            cropped_texted_image.mask = cropped_texted_image.mask[_bbox.slice]
            cropped_texted_image._resize((bbox.height, bbox.width))
            self.orig[bbox.slice] = cropped_texted_image.orig
            self.timg[bbox.slice] = cropped_texted_image.timg
            self.mask[bbox.slice] = cropped_texted_image.mask

    def _resize(self, size: tuple[int, int]):
        """Note: this function does not create new `TextedImage` obejct but modifies itself."""
        _, H, W = self.orig.shape
        TARGET_H, TARGET_W = size
        # Calculate aspect ratios
        original_aspect = W / H
        target_aspect = TARGET_W / TARGET_H
        # Resize
        if original_aspect > target_aspect:
            new_h = int(TARGET_W / original_aspect)
            new_w = TARGET_W
        else:
            new_h = TARGET_H
            new_w = int(TARGET_H * original_aspect)
        self.orig = VTF.resize(self.orig, (new_h, new_w))
        self.timg = VTF.resize(self.timg, (new_h, new_w))
        self.mask = VTF.resize(self.mask, (new_h, new_w))
        # pad
        pad_left = (TARGET_W - new_w) // 2
        pad_right = TARGET_W - new_w - pad_left
        pad_top = (TARGET_H - new_h) // 2
        pad_bottom = TARGET_H - new_h - pad_top
        self.orig = VTF.pad(self.orig, (pad_left, pad_top, pad_right, pad_bottom))
        self.timg = VTF.pad(self.timg, (pad_left, pad_top, pad_right, pad_bottom))
        self.mask = VTF.pad(self.mask, (pad_left, pad_top, pad_right, pad_bottom))
        # resize bboxes
        new_bboxes: list[BBox] = []
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
    def _margin_crop(
        img: img_tensor,
        bbox: BBox,
        margin: int,
    ):
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
        img = VTF.pad(img, (pad_left, pad_top, pad_right, pad_bottom))
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

'''
    TextedImage 저장 유틸
'''
def _save_one(idx: int, txt_img, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    oimg_fn = f"oimg_{idx:04d}.png"
    timg_fn = f"timg_{idx:04d}.png"
    mask_fn = f"timg_{idx:04d}_mask.png"
    meta_fn = f"timg_{idx:04d}.json"

    char_infos_fn: Optional[str] = None
    # --- Score Map 파일 이름을 이미지 확장자로 변경 ---
    region_map_img_fn: Optional[str] = None
    affinity_map_img_fn: Optional[str] = None

    try:
        # 기본 이미지 및 마스크 저장 (기존과 동일)
        to_pil_image(txt_img.orig.cpu()).save(os.path.join(out_dir, oimg_fn))
        to_pil_image(txt_img.timg.cpu()).save(os.path.join(out_dir, timg_fn))
        # 마스크는 이미 0 또는 1의 값을 가지는 텐서일 수 있으므로,
        # to_pil_image가 처리할 수 있도록 (0-255 스케일링 또는 모드 명시)
        # 여기서는 (0,1) 범위의 float 텐서가 (0,255)로 자동 스케일링된다고 가정
        mask_squeezed = txt_img.mask.squeeze(0).cpu()
        if mask_squeezed.dtype == torch.bool:  # bool 타입이면 float으로 변환
            mask_squeezed = mask_squeezed.float()
        to_pil_image(mask_squeezed).save(os.path.join(out_dir, mask_fn))

        data: Dict[str, Any] = {
            "orig": oimg_fn,
            "img": timg_fn,
            "mask": mask_fn,
            "bboxes": [[b.x1, b.y1, b.x2, b.y2] for b in txt_img.bboxes if b is not None],
        }

        # CharInfo 저장 (기존과 동일 - .npy)
        if txt_img.all_char_infos is not None:
            char_infos_fn = f"timg_{idx:04d}_char_infos.npy"
            char_infos_to_save = [
                {"polygon": ci.polygon.tolist(), "char_content": ci.char_content, "word_id": ci.word_id}
                for ci in txt_img.all_char_infos
            ]
            np.save(os.path.join(out_dir, char_infos_fn), char_infos_to_save, allow_pickle=True)
            data["char_infos_file"] = char_infos_fn

        # --- Region Score Map을 이미지로 저장 ---
        if txt_img.region_score_map is not None:
            region_map_img_fn = f"timg_{idx:04d}_region_map.png"  # PNG로 저장
            # Score Map은 (0,1) 범위의 float 텐서 (예: (1, H/2, W/2))
            # 이를 (0,255) 범위의 uint8 PIL 이미지로 변환하여 저장
            region_map_tensor = txt_img.region_score_map.squeeze(0).cpu()  # (H/2, W/2)
            # 0-1 범위를 0-255로 스케일링
            region_map_scaled = (region_map_tensor * 255.0).byte()  # .byte()는 .to(torch.uint8)와 유사
            to_pil_image(region_map_scaled).save(os.path.join(out_dir, region_map_img_fn))
            data["region_map_image_file"] = region_map_img_fn  # 키 이름 변경

        # --- Affinity Score Map을 이미지로 저장 ---
        if txt_img.affinity_score_map is not None:
            affinity_map_img_fn = f"timg_{idx:04d}_affinity_map.png"  # PNG로 저장
            affinity_map_tensor = txt_img.affinity_score_map.squeeze(0).cpu()  # (H/2, W/2)
            affinity_map_scaled = (affinity_map_tensor * 255.0).byte()
            to_pil_image(affinity_map_scaled).save(os.path.join(out_dir, affinity_map_img_fn))
            data["affinity_map_image_file"] = affinity_map_img_fn  # 키 이름 변경

        with open(os.path.join(out_dir, meta_fn), "w") as f:
            json.dump(data, f, indent=2)

    except Exception as e:
        print(f"[save error] idx {idx} ({meta_fn}): {e}")


def save_timgs(timgs: List[TextedImage], out_dir: str, num_threads: int = 8):
    """병렬로 TextedImage 리스트를 디스크에 저장한다."""
    with ThreadPoolExecutor(max_workers=num_threads) as pool:
        for i, ti in enumerate(timgs):
            pool.submit(_save_one, i, ti, out_dir)
    print(f"[Pipeline] all({len(timgs)}) timg/meta saved.")


# ─────────────────────────────────────────────

def _load_one(json_path: str, base_dir: str, device):
    from .Utils import BBox       # 로컬에서 불러도 되도록

    with open(json_path) as f:
        meta = json.load(f)

    # 기본 이미지 및 마스크 로드 (기존과 거의 동일)
    orig_pil = Image.open(os.path.join(base_dir, meta["orig"])).convert("RGB")
    orig = to_tensor(orig_pil).to(device)
    if orig.dim() == 4 and orig.shape[0] == 1: orig = orig.squeeze(0)

    timg_pil = Image.open(os.path.join(base_dir, meta["img"])).convert("RGB")
    timg = to_tensor(timg_pil).to(device)
    if timg.dim() == 4 and timg.shape[0] == 1: timg = timg.squeeze(0)

    mask_pil = Image.open(os.path.join(base_dir, meta["mask"])).convert("L")  # Grayscale (L)
    mask = to_tensor(mask_pil).to(device)  # (1,H,W) 또는 (H,W)
    if mask.dim() == 2: mask = mask.unsqueeze(0)
    # 마스크가 0 또는 1의 값을 가지도록 변환 (만약 0-255로 로드되었다면)
    # Image.open().convert("L")은 0-255 범위. to_tensor는 이를 0-1로 스케일링.
    # 만약 이진 마스크가 필요하면, mask = (mask > 0.5).float() 등으로 처리.
    # 여기서는 to_tensor의 스케일링을 믿고 그대로 사용.

    bboxes_data = meta.get("bboxes")
    bboxes = [BBox(*pts) for pts in bboxes_data] if bboxes_data else []

    loaded_all_char_infos: Optional[List[CharInfo]] = None
    loaded_region_map: Optional[torch.Tensor] = None
    loaded_affinity_map: Optional[torch.Tensor] = None

    # CharInfo 로드 (기존과 동일 - .npy)
    char_infos_file = meta.get("char_infos_file")
    if char_infos_file:
        try:
            char_infos_saved_data = np.load(os.path.join(base_dir, char_infos_file), allow_pickle=True)
            loaded_all_char_infos = [
                CharInfo(np.array(ci_data["polygon"]), ci_data["char_content"], ci_data["word_id"])
                for ci_data in char_infos_saved_data
            ]
        except Exception as e:
            print(f"Error loading char_infos_file {char_infos_file}: {e}")

    # --- Region Score Map을 이미지 파일로부터 로드 ---
    region_map_img_file = meta.get("region_map_image_file")  # 키 이름 변경
    if region_map_img_file:
        try:
            region_map_pil = Image.open(os.path.join(base_dir, region_map_img_file)).convert("L")  # Grayscale
            # PIL 이미지를 (0,1) 범위의 float 텐서로 변환
            region_map_tensor = to_tensor(region_map_pil).to(device)  # to_tensor는 (0,255) uint8 -> (0,1) float 자동 변환
            # (1, H/2, W/2) 형태여야 함
            if region_map_tensor.dim() == 2: region_map_tensor = region_map_tensor.unsqueeze(0)
            loaded_region_map = region_map_tensor
        except Exception as e:
            print(f"Error loading region_map_image_file {region_map_img_file}: {e}")

    # --- Affinity Score Map을 이미지 파일로부터 로드 ---
    affinity_map_img_file = meta.get("affinity_map_image_file")  # 키 이름 변경
    if affinity_map_img_file:
        try:
            affinity_map_pil = Image.open(os.path.join(base_dir, affinity_map_img_file)).convert("L")
            affinity_map_tensor = to_tensor(affinity_map_pil).to(device)
            if affinity_map_tensor.dim() == 2: affinity_map_tensor = affinity_map_tensor.unsqueeze(0)
            loaded_affinity_map = affinity_map_tensor
        except Exception as e:
            print(f"Error loading affinity_map_image_file {affinity_map_img_file}: {e}")

    return TextedImage(
        orig, timg, mask, bboxes,
        all_char_infos=loaded_all_char_infos,
        region_score_map=loaded_region_map,
        affinity_score_map=loaded_affinity_map
    )


def load_timgs(base_dir: str, device, max_num:int|None = None, shuffle:bool = False, num_threads: int = 8) -> List[TextedImage]:
    """저장된 timg_XXXX.json 세트를 병렬 로드하여 TextedImage 리스트로 반환."""
    json_files = sorted(glob.glob(base_dir + "/timg_*.json"))
    if not json_files:
        raise RuntimeError(f"No JSON meta files in {base_dir}")
    # ── 개수 제한 & 셔플 ───────────────────────────
    if shuffle:
        random.shuffle(json_files)
    if max_num is not None:
        json_files = json_files[:max_num]

    timgs: List[TextedImage] = []
    with ThreadPoolExecutor(max_workers=num_threads) as pool:
        futures = [pool.submit(_load_one, jf, base_dir, device) for jf in json_files]
        for fu in futures:
            timgs.append(fu.result())
    return timgs