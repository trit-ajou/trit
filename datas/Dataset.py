import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Dict, Any  # Import Any
from .TextedImage import TextedImage  # TextedImage is in the same directory


class MangaDataset1(Dataset):
    # -------- 수정 시작 (__init__ 시그니처 및 __getitem__ 반환 값 - 이전 답변과 동일) --------
    def __init__(self, texted_image_list: List[TextedImage],
                 generate_craft_gt: bool = False,  # 플래그 추가
                 transforms=None):  # 이미지에만 적용될 transform
        super().__init__()
        self.texted_images = texted_image_list
        self.generate_craft_gt = generate_craft_gt  # 플래그 저장
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.texted_images)

    def __getitem__(self, idx: int) -> Any:
        texted_image = self.texted_images[idx]
        input_tensor = texted_image.timg  # (C, H, W)

        if self.transforms:
            input_tensor = self.transforms(input_tensor)  # 이미지에만 적용

        if self.generate_craft_gt:
            # CRAFT 학습 모드: (이미지, Region GT, Affinity GT) 반환
            region_gt = texted_image.region_score_map
            affinity_gt = texted_image.affinity_score_map

            # ImageLoader에서 GT 생성 실패 시 None이 올 수 있으므로, 기본 0 텐서로 대체
            if region_gt is None:
                out_h = input_tensor.shape[1] // 2;
                out_w = input_tensor.shape[2] // 2
                # region_gt = torch.zeros((1, out_h, out_w), device=input_tensor.device)
                region_gt = torch.zeros((1, out_h, out_w), device='cpu')
            if affinity_gt is None:
                out_h = input_tensor.shape[1] // 2;
                out_w = input_tensor.shape[2] // 2
                # affinity_gt = torch.zeros((1, out_h, out_w), device=input_tensor.device)
                affinity_gt = torch.zeros((1, out_h, out_w), device='cpu')

            return input_tensor, region_gt, affinity_gt
        else:
            # 기존 모드 또는 다른 모델용: (이미지, 단어/텍스트 덩어리 BBox 리스트) 반환
            target_bboxes_original = texted_image.bboxes
            # 기존 collate_fn이 튜플을 기대한다면 튜플로, 아니면 딕셔너리로.
            # 제공해주신 코드에서는 (images, targets) 튜플을 사용하므로, targets에 bboxes를 넣음.
            return input_tensor, target_bboxes_original
            # -------- 수정 끝 --------


class MangaDataset2(Dataset):
    def __init__(
        self, texted_image_list: List[TextedImage], input_size: Tuple[int, int]
    ):
        super().__init__()
        self.texted_images = texted_image_list
        self.input_size = input_size

    def __len__(self) -> int:
        return len(self.texted_images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        texted_image = self.texted_images[idx]
        texted_image._resize(self.input_size)
        input_tensor = texted_image.timg
        target_mask = texted_image.mask  # Should be (1, H, W)

        return input_tensor, target_mask


class MangaDataset3(Dataset):
    def __init__(
        self, texted_image_list: List[TextedImage], input_size: Tuple[int, int]
    ):
        super().__init__()
        self.texted_images = texted_image_list
        self.input_size = input_size

    def __len__(self) -> int:
        return len(self.texted_images)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        texted_image = self.texted_images[idx]
        texted_image._resize(self.input_size)

        # Model3 input is typically the texted image and the mask
        input_dict = {
            "image_with_text": texted_image.timg,  # (C, H, W)
            "mask": texted_image.mask,  # (1, H, W)
        }
        target_image = texted_image.orig  # (C, H, W)

        return input_dict, target_image
