# model1_inference.py

import torch
import cv2
import numpy as np
from typing import Tuple, List

# CRAFT의 유틸리티 함수들을 임포트해야 함
from . import craft_utils, imgproc


def inference_model1(
        model: torch.nn.Module,
        image: np.ndarray,
        device: torch.device,
        canvas_size: int = 1280,
        mag_ratio: float = 1.5,
        text_threshold: float = 0.7,
        link_threshold: float = 0.4,
        low_text: float = 0.4,
        poly: bool = False
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    단일 이미지에 대해 CRAFT(Model1) 추론을 수행하여 바운딩 박스를 생성합니다.

    Args:
        model (nn.Module): 추론에 사용할 CRAFT 모델.
        image (np.ndarray): 입력 이미지 (OpenCV BGR 또는 RGB 형식의 NumPy 배열).
        device (torch.device): 추론에 사용할 디바이스.
        canvas_size (int): 리사이즈 시 기준이 되는 캔버스 크기.
        mag_ratio (float): 이미지 확대 비율.
        text_threshold (float): Region score의 임계값.
        link_threshold (float): Affinity score의 임계값.
        low_text (float): 텍스트 영역 확장을 위한 낮은 임계값.
        poly (bool): True이면 폴리곤 형태의 결과를, False이면 사각형 bbox를 반환.

    Returns:
        Tuple[List[np.ndarray], List[np.ndarray]]:
            - boxes: 각 텍스트 영역의 바운딩 박스 좌표 리스트.
            - polys: 각 텍스트 영역의 폴리곤 좌표 리스트 (poly=True일 때 의미 있음).
    """
    # 1. 모델을 평가 모드로 설정
    model.eval()
    model.to(device)

    # 2. 이미지 전처리 (test_net.py와 동일)
    # 가. 리사이즈
    img_resized, target_ratio, _ = imgproc.resize_aspect_ratio(
        image, canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=mag_ratio
    )
    ratio_h = ratio_w = 1 / target_ratio

    # 나. 정규화 및 텐서 변환
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)  # [h, w, c] -> [c, h, w]
    x = x.unsqueeze(0).to(device)  # [c, h, w] -> [b, c, h, w]

    # 3. 모델 순전파 (Forward Pass)
    with torch.no_grad():
        pred_maps, _ = model(x)

    # 4. 후처리 (Post-processing)
    # 가. score map 분리
    score_text = pred_maps[0, :, :, 0].cpu().numpy()
    score_link = pred_maps[0, :, :, 1].cpu().numpy()

    # 나. 바운딩 박스 추출
    boxes, polys = craft_utils.getDetBoxes(
        score_text, score_link, text_threshold, link_threshold, low_text, poly
    )

    # 다. 좌표 복원
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)

    # poly=False일 때 polys가 None일 수 있으므로 boxes로 채워줌
    for k in range(len(polys)):
        if polys[k] is None:
            polys[k] = boxes[k]

    return boxes, polys