import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from .TextedImage import TextedImage
from typing import List, Tuple, Dict, Any  # Import Any
from .TextedImage import TextedImage  # TextedImage is in the same directory
from . import imgproc
from torch.autograd import Variable
import cv2
import numpy as np


class MangaDataset1(Dataset):
    def __init__(self,
                 texted_image_list: List['TextedImage'],
                 canvas_size: int = 1280,
                 mag_ratio: float = 1.5,
                 ):
        """
        CRAFT 모델 학습을 위한 전용 데이터셋.
        __getitem__에서 이미지 및 GT 맵에 대한 리사이즈 및 정규화를 수행합니다.

        Args:
            texted_image_list (List[TextedImage]): ImageLoader가 생성한 TextedImage 객체 리스트.
            canvas_size (int): 리사이즈 시 기준이 되는 캔버스 크기.
            mag_ratio (float): 원본 이미지의 긴 변을 기준으로 확대할 비율.
        """
        super().__init__()
        self.texted_images = texted_image_list
        self.canvas_size = canvas_size
        self.mag_ratio = mag_ratio
        self.debug_dataset1: bool = False

    def __len__(self) -> int:
        return len(self.texted_images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        하나의 샘플에 대해 전처리를 적용하고 (이미지, Region GT, Affinity GT)를 반환합니다.
        """
        texted_image = self.texted_images[idx]

        # 1. 데이터를 NumPy 배열로 변환
        # timg: (C, H, W) -> (H, W, C) for OpenCV, 0-255 범위의 uint8
        image_np = texted_image.timg.permute(1, 2, 0).numpy() * 255
        image_np = image_np.astype(np.uint8)

        # GT 맵: (1, H_map, W_map) -> (H_map, W_map, 1) for OpenCV, 0-1 범위의 float32
        region_gt_np = texted_image.region_score_map.permute(1, 2, 0).numpy()
        affinity_gt_np = texted_image.affinity_score_map.permute(1, 2, 0).numpy()

        # GT 맵들을 하나로 합쳐서 한 번에 리사이즈
        gt_maps_np = np.concatenate([region_gt_np, affinity_gt_np], axis=2)

        # 2. 리사이즈 (CRAFT의 resize_aspect_ratio 사용)
        # 이미지 리사이즈
        img_resized, target_ratio, _ = imgproc.resize_aspect_ratio(
            image_np,
            self.canvas_size,
            interpolation=cv2.INTER_LINEAR,
            mag_ratio=self.mag_ratio
        )

        # GT 맵 리사이즈 (이미지와 동일한 비율로, INTER_NEAREST 사용 권장)
        # resize_aspect_ratio는 3채널 이미지를 가정하므로, 2채널인 GT맵에는 직접 적용 불가.
        # 대신 계산된 target_ratio를 사용하여 직접 리사이즈.
        height, width, _ = gt_maps_np.shape
        target_h, target_w = int(height * target_ratio), int(width * target_ratio)

        # GT 맵은 히트맵이므로 Nearest-neighbor 보간법이 더 적합할 수 있음
        gt_maps_resized = cv2.resize(gt_maps_np, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

        # 만약 gt_maps_resized가 (H,W) 형태로 (채널 차원이 사라짐) 리턴되면 다시 확장
        if gt_maps_resized.ndim == 2:
            gt_maps_resized = np.expand_dims(gt_maps_resized, axis=2)

        # 리사이즈된 이미지와 동일한 캔버스 크기로 패딩
        # img_resized와 gt_maps_resized의 최종 크기는 heatmap 크기 관계를 고려해야 함
        # resize_aspect_ratio는 이미지 크기를 32의 배수로 맞추므로, gt_map도 그에 맞춰야 함
        # img_resized.shape[0]는 리사이즈된 이미지의 높이 (패딩 포함)
        # gt_map의 최종 크기는 (img_h/2, img_w/2)가 되어야 함.
        final_h, final_w, _ = img_resized.shape
        final_gt_h, final_gt_w = final_h // 2, final_w // 2

        final_gt_maps = np.zeros((final_gt_h, final_gt_w, 2), dtype=np.float32)
        # 리사이즈된 gt_maps_resized를 최종 캔버스에 붙여넣기
        # target_h, target_w는 리사이즈 후 패딩 전 크기
        paste_h, paste_w = target_h // 2, target_w // 2

        # gt_maps_resized도 1/2 스케일로 다시 리사이즈
        gt_maps_resized_half = cv2.resize(gt_maps_resized, (paste_w, paste_h), interpolation=cv2.INTER_NEAREST)
        if gt_maps_resized_half.ndim == 2:
            gt_maps_resized_half = np.expand_dims(gt_maps_resized_half, axis=2)

        final_gt_maps[0:paste_h, 0:paste_w, :] = gt_maps_resized_half

        region_gt_final = final_gt_maps[:, :, 0]
        affinity_gt_final = final_gt_maps[:, :, 1]

        # 3. 이미지 정규화 (CRAFT의 normalizeMeanVariance 사용)
        img_normalized = imgproc.normalizeMeanVariance(img_resized)

        # 4. PyTorch 텐서로 변환
        # 이미지: (H, W, C) -> (C, H, W)
        image_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1)

        # GT 맵: (H_map, W_map) -> (1, H_map, W_map) 형태로 채널 차원(channel dimension) 추가
        region_gt_tensor = torch.from_numpy(region_gt_final).unsqueeze(0)
        affinity_gt_tensor = torch.from_numpy(affinity_gt_final).unsqueeze(0)

        if self.debug_dataset1:
            print("--------------------Dataset1.getitem-----------------")
            print("Image Shape: ", image_tensor.shape)
            print("Region GT Shape: ", region_gt_tensor.shape)
            print("Affinity GT Shape: ", affinity_gt_tensor.shape)
            print("-----------------------------------------------------")
            # 원본 이미지와 마스크도 반환 (시각화용)
            # 전처리가 안 된 원본 텐서를 반환

        original_timg = self.texted_images[idx].timg
        original_mask = self.texted_images[idx].mask

        return image_tensor, region_gt_tensor, affinity_gt_tensor, original_timg, original_mask


class MangaDataset2(Dataset):
    def __init__(self, texted_images, transform=False):
        super().__init__()
        self.texted_images = texted_images
        self.transform = transform

        if self.transform:
            self.img_transform = transforms.Compose(
                [
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                    transforms.RandomHorizontalFlip(p=0.3),
                ]
            )

    def __len__(self):
        return len(self.texted_images)

    def __getitem__(self, idx: int):
        texted_image = self.texted_images[idx]

        # Get image and mask
        timg = texted_image.timg  # (C, H, W)
        mask = texted_image.mask  # (1, H, W)

        # Apply transforms if enabled
        if self.transform:
            # Apply same transform to both image and mask
            # Convert to PIL for transforms
            timg_pil = transforms.ToPILImage()(timg)
            mask_pil = transforms.ToPILImage()(mask)

            # Apply transforms
            timg_pil = self.img_transform(timg_pil)

            # Convert back to tensor
            timg = transforms.ToTensor()(timg_pil)
            mask = transforms.ToTensor()(mask_pil)

        # Ensure mask is in correct format for training
        if mask.dim() == 3 and mask.shape[0] == 1:
            mask = mask.squeeze(0)  # (1, H, W) -> (H, W)
        elif mask.dim() == 2:
            pass  # Already (H, W)
        else:
            raise ValueError(f"Unexpected mask dimensions: {mask.shape}")

        return timg, mask


class MangaDataset3(Dataset): # 초간단 model3 데이터셋
    def __init__(self, data: list[TextedImage]):
        super().__init__()
        self.data = data
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]