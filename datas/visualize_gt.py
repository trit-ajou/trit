# visualization_utils.py (또는 PipelineMgr.py 상단 등 적절한 위치에 정의)
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import Optional  # TextedImage 타입 힌팅을 위해 필요할 수 있음


# TextedImage 클래스 정의가 이 파일에서 보이거나 임포트되어야 함.
# 예: from your_project.TextedImage import TextedImage
# (실제 프로젝트 구조에 맞게 경로 수정 필요)

def visualize_craft_score_maps_only(
        texted_image_obj: 'TextedImage',  # TextedImage 객체 (타입 힌트)
        idx: int,  # 현재 샘플의 인덱스 (파일 이름에 사용)
        save_dir: str  # 시각화 이미지를 저장할 디렉토리 경로
):
    """
    TextedImage 객체에 포함된 Region Score Map과 Affinity Score Map만을 시각화하고
    지정된 디렉토리에 저장합니다.
    """
    if texted_image_obj is None:
        print(f"[Visualizing Score Maps] Sample {idx}: TextedImage object is None. Skipping.")
        return

    # Score Map 존재 여부 확인
    has_region_map = hasattr(texted_image_obj, 'region_score_map') and texted_image_obj.region_score_map is not None
    has_affinity_map = hasattr(texted_image_obj,
                               'affinity_score_map') and texted_image_obj.affinity_score_map is not None

    if not (has_region_map and has_affinity_map):
        print(
            f"[Visualizing Score Maps] Sample {idx}: Region or Affinity Score Map not found in TextedImage object. Skipping visualization.")
        return

    # 1행 2열의 서브플롯 생성 (Region Map, Affinity Map)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(f"CRAFT GT Score Maps - Sample {idx}", fontsize=16)

    # --- 1. Region Score Map 시각화 ---
    ax_region = axes[0]
    try:
        # 텐서가 GPU에 있을 수 있으므로 CPU로 옮기고 NumPy 배열로 변환
        # Score Map은 보통 (1, H_map, W_map) 형태이므로 squeeze()로 채널 차원 제거
        region_map_tensor = texted_image_obj.region_score_map
        region_map_for_display = region_map_tensor.squeeze().cpu().numpy() \
            if region_map_tensor.is_cuda else region_map_tensor.squeeze().numpy()

        im_region = ax_region.imshow(region_map_for_display, cmap='jet', vmin=0, vmax=1)
        ax_region.set_title("Region Score Map (GT)")
        fig.colorbar(im_region, ax=ax_region, fraction=0.046, pad=0.04)  # 컬러바 추가
    except Exception as e:
        # 오류 발생 시 해당 서브플롯에 에러 메시지 표시
        ax_region.text(0.5, 0.5, f"Error displaying Region Map:\n{e}",
                       ha='center', va='center', color='red')
        ax_region.set_title("Region Score Map (Error)")
    ax_region.axis('off')  # 축 정보 숨기기

    # --- 2. Affinity Score Map 시각화 ---
    ax_affinity = axes[1]
    try:
        affinity_map_tensor = texted_image_obj.affinity_score_map
        affinity_map_for_display = affinity_map_tensor.squeeze().cpu().numpy() \
            if affinity_map_tensor.is_cuda else affinity_map_tensor.squeeze().numpy()

        im_affinity = ax_affinity.imshow(affinity_map_for_display, cmap='jet', vmin=0, vmax=1)
        ax_affinity.set_title("Affinity Score Map (GT)")
        fig.colorbar(im_affinity, ax=ax_affinity, fraction=0.046, pad=0.04)  # 컬러바 추가
    except Exception as e:
        ax_affinity.text(0.5, 0.5, f"Error displaying Affinity Map:\n{e}",
                         ha='center', va='center', color='red')
        ax_affinity.set_title("Affinity Score Map (Error)")
    ax_affinity.axis('off')  # 축 정보 숨기기

    # 서브플롯 간 간격 및 전체 레이아웃 조정
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # rect는 [left, bottom, right, top] (suptitle 공간 확보)

    # 지정된 디렉토리가 없으면 생성
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)  # exist_ok=True로 이미 디렉토리가 있어도 에러 방지

    # 파일 저장 경로 설정
    save_path = os.path.join(save_dir, f"score_maps_gt_sample_{idx:04d}.png")

    try:
        plt.savefig(save_path)
        print(f"  Score Maps GT Visualization for sample {idx} saved to {save_path}")
    except Exception as e:
        print(f"  Error saving Score Maps GT visualization for sample {idx} to {save_path}: {e}")
    finally:
        plt.close(fig)  # 시각화가 끝난 figure 객체를 닫아 메모리 누수 방지