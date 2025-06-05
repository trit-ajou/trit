# visualization_utils.py (또는 PipelineMgr.py 상단 등 적절한 위치에 정의)
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import Optional  # TextedImage 타입 힌팅을 위해 필요할 수 있음
import torchvision.transforms.functional as TF # TF로 alias 사용
from PIL import Image

def visualize_craft_gt_components(
        texted_image_obj: 'TextedImage',
        idx: int,
        save_dir: str,
        alpha_score_overlay: float = 0.7 # 스코어맵 오버레이 시 투명도 (스코어맵이 더 잘 보이도록)
):
    """
    TextedImage 객체로부터 원본, 마스크, Region Map, Affinity Map,
    그리고 마스크와 각 스코어맵을 혼합한 이미지를 시각화합니다.
    """
    if texted_image_obj is None:
        print(f"[Visualizing GT Components] Sample {idx}: TextedImage object is None. Skipping.")
        return

    # 필수 데이터 존재 여부 확인
    has_orig = hasattr(texted_image_obj, 'orig') and texted_image_obj.orig is not None
    has_mask = hasattr(texted_image_obj, 'mask') and texted_image_obj.mask is not None
    has_region = hasattr(texted_image_obj, 'region_score_map') and texted_image_obj.region_score_map is not None
    has_affinity = hasattr(texted_image_obj, 'affinity_score_map') and texted_image_obj.affinity_score_map is not None

    if not (has_orig and has_mask and has_region and has_affinity):
        print(f"[Visualizing GT Components] Sample {idx}: Missing one or more essential components. Skipping.")
        return

    # 2행 3열의 서브플롯 생성
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))  # figsize 조절 가능
    fig.suptitle(f"CRAFT GT Components - Sample {idx}", fontsize=16)
    axes = axes.ravel()  # 1D 배열로 만들어 인덱싱 용이하게

    # --- 데이터 준비 ---
    try:
        # 1. 원본 이미지 (timg 사용: 텍스트가 합성된 이미지)
        #    orig 대신 timg를 사용해야 텍스트 위치와 스코어맵을 비교하기 용이
        img_tensor = texted_image_obj.timg
        img_pil = TF.to_pil_image(img_tensor.cpu() if img_tensor.is_cuda else img_tensor)

        # 2. 마스크
        mask_tensor = texted_image_obj.mask
        # 마스크는 (1, H, W) 이므로 squeeze(). 값은 0 또는 1.
        mask_np = mask_tensor.squeeze().cpu().numpy() if mask_tensor.is_cuda else mask_tensor.squeeze().numpy()
        # 2번 서브플롯 표시용 마스크 (회색조, 0-255)
        mask_for_plot2 = (mask_np * 255).astype(np.uint8)

        # 5번, 6번 서브플롯 오버레이용 배경 마스크 (RGB, 0-255)
        # 흰색 텍스트(255), 검은색 배경(0)으로 만듭니다.
        mask_rgb_for_overlay = np.stack([(mask_np * 255).astype(np.uint8)] * 3, axis=-1)

        # 3. Region Score Map
        region_tensor = texted_image_obj.region_score_map
        # (1, H/2, W/2) -> (H/2, W/2)
        region_np = region_tensor.squeeze().cpu().numpy() if region_tensor.is_cuda else region_tensor.squeeze().numpy()
        # Region Map을 원본 이미지 크기로 리사이즈 (혼합용)
        # PIL 이미지로 변환 후 리사이즈
        region_pil_for_resize = TF.to_pil_image(torch.from_numpy(region_np))  # float32 텐서로 변환
        region_resized_pil = region_pil_for_resize.resize(img_pil.size,
                                                          resample=Image.Resampling.NEAREST)  # 또는 BILINEAR
        region_resized_np = np.array(region_resized_pil)  # 혼합을 위해 NumPy로

        # 4. Affinity Score Map
        affinity_tensor = texted_image_obj.affinity_score_map
        affinity_np = affinity_tensor.squeeze().cpu().numpy() if affinity_tensor.is_cuda else affinity_tensor.squeeze().numpy()
        affinity_pil_for_resize = TF.to_pil_image(torch.from_numpy(affinity_np))
        affinity_resized_pil = affinity_pil_for_resize.resize(img_pil.size, resample=Image.Resampling.NEAREST)
        affinity_resized_np = np.array(affinity_resized_pil)

    except Exception as e:
        print(f"[Visualizing GT Components] Sample {idx}: Error preparing data - {e}. Skipping.")
        plt.close(fig)
        return

    # --- 시각화 ---
    titles = [
        "1) Original Image (timg)", "2) Text Mask", "3) Region Score Map",
        "4) Affinity Score Map", "5) Mask + Region (Overlay)", "6) Mask + Affinity (Overlay)"
    ]

    # 1. 원본 이미지 (timg)
    axes[0].imshow(img_pil)
    axes[0].set_title(titles[0])
    axes[0].axis('off')

    # 2. 마스크
    axes[1].imshow(mask_for_plot2, cmap='gray', vmin=0, vmax=255)
    axes[1].set_title(titles[1])
    axes[1].axis('off')

    # 3. Region Score Map (원본 크기, jet cmap)
    im_region = axes[2].imshow(region_resized_np, cmap='jet', vmin=0,
                               vmax=1 if region_resized_np.max() <= 1 else 255)  # vmax는 데이터 범위에 따라 조정
    axes[2].set_title(titles[2])
    axes[2].axis('off')
    fig.colorbar(im_region, ax=axes[2], fraction=0.046, pad=0.04)

    # 4. Affinity Score Map (원본 크기, jet cmap)
    im_affinity = axes[3].imshow(affinity_resized_np, cmap='jet', vmin=0,
                                 vmax=1 if affinity_resized_np.max() <= 1 else 255)
    axes[3].set_title(titles[3])
    axes[3].axis('off')
    fig.colorbar(im_affinity, ax=axes[3], fraction=0.046, pad=0.04)

    # 5. 마스크 위에 Region Score Map 오버레이
    try:
        # 오버레이 배경으로 mask_rgb_for_overlay 사용
        background_for_region = mask_rgb_for_overlay.copy()
        region_rgba = plt.cm.jet(region_resized_np)
        region_rgb = region_rgba[:, :, :3] * 255
        region_alpha = region_rgba[:, :, 3] * alpha_score_overlay

        for c in range(3):
            background_for_region[:, :, c] = background_for_region[:, :, c] * (1 - region_alpha) + \
                                             region_rgb[:, :, c] * region_alpha

        axes[4].imshow(background_for_region.astype(np.uint8))
    except Exception as e:
        axes[4].text(0.5, 0.5, f"Error overlaying Region:\n{e}", ha='center', va='center', color='red')
    axes[4].set_title(titles[4])
    axes[4].axis('off')

    # 6. 마스크 위에 Affinity Score Map 오버레이
    try:
        # 오버레이 배경으로 mask_rgb_for_overlay 사용
        background_for_affinity = mask_rgb_for_overlay.copy()
        affinity_rgba = plt.cm.jet(affinity_resized_np)
        affinity_rgb = affinity_rgba[:, :, :3] * 255
        affinity_alpha = affinity_rgba[:, :, 3] * alpha_score_overlay

        for c in range(3):
            background_for_affinity[:, :, c] = background_for_affinity[:, :, c] * (1 - affinity_alpha) + \
                                               affinity_rgb[:, :, c] * affinity_alpha

        axes[5].imshow(background_for_affinity.astype(np.uint8))
    except Exception as e:
        axes[5].text(0.5, 0.5, f"Error overlaying Affinity:\n{e}", ha='center', va='center', color='red')
    axes[5].set_title(titles[5])
    axes[5].axis('off')

    # 레이아웃 및 저장
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # suptitle 공간 확보
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"gt_components_sample_{idx:04d}.png")
    try:
        plt.savefig(save_path)
        print(f"  GT Components Visualization for sample {idx} saved to {save_path}")
    except Exception as e:
        print(f"  Error saving GT Components visualization for sample {idx} to {save_path}: {e}")
    finally:
        plt.close(fig)