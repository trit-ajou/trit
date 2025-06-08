import os
import json
import glob
import random
from concurrent.futures import ThreadPoolExecutor
from enum import Enum, auto
from typing import List, Optional, Dict, Any

import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor, to_pil_image

from .TextedImage import TextedImage, CharInfo
from .Utils import BBox

class LoadingMode(Enum):
    """
        TextedImage 데이터 로딩 모드를 정의하는 열거형(Enum).
        """
    FULL = auto()  # 전체 데이터 로드 (oimg, timg, mask, bboxes, craft_gt)
    MODEL1_TRAIN = auto() # oimg를 제외한 전체 데이터 로드
    MODEL1_INFER = auto() # timg만 로드
    MODEL2_TRAIN = auto() # timg, mask, bboxes만 로드
    MODEL2_INFER = auto() # timg, bboxes만 로드
    MODEL3_TRAIN = auto() # oimg, timg, mask, bboxes만 로드
    MODEL3_INFER = auto() # timg, mask, bbox만 로드
    INFERENCE = auto() # oimg만 로드


def _save_one(idx: int, txt_img: 'TextedImage', out_dir: str):
    """
    [Helper] 단일 TextedImage 객체의 모든 구성 요소를 파일로 저장합니다.
    - 이미지는 PNG, GT Score Map은 PNG (시각화용), CharInfo는 NPY로 저장됩니다.
    """
    try:
        os.makedirs(out_dir, exist_ok=True)
        # 파일명 정의
        oimg_fn = f"oimg_{idx:04d}.png"
        timg_fn = f"timg_{idx:04d}.png"
        mask_fn = f"timg_{idx:04d}_mask.png"
        meta_fn = f"timg_{idx:04d}.json"

        # 기본 이미지 저장
        to_pil_image(txt_img.orig.cpu()).save(os.path.join(out_dir, oimg_fn))
        to_pil_image(txt_img.timg.cpu()).save(os.path.join(out_dir, timg_fn))
        to_pil_image(txt_img.mask.cpu()).save(os.path.join(out_dir, mask_fn))

        # 메타데이터 구성
        data: Dict[str, Any] = {
            "orig_file": oimg_fn, "timg_file": timg_fn, "mask_file": mask_fn,
            "bboxes": [[b.x1, b.y1, b.x2, b.y2] for b in txt_img.bboxes if b is not None],
        }

        # CRAFT GT 데이터 저장 (있을 경우)
        if txt_img.all_char_infos:
            char_infos_fn = f"timg_{idx:04d}_char_infos.npy"
            char_infos_to_save = [
                {"polygon": ci.polygon.tolist(), "char_content": ci.char_content, "word_id": ci.word_id}
                for ci in txt_img.all_char_infos
            ]
            np.save(os.path.join(out_dir, char_infos_fn), char_infos_to_save)
            data["char_infos_file"] = char_infos_fn

        if txt_img.region_score_map is not None:
            region_map_fn = f"timg_{idx:04d}_region_map.png"
            to_pil_image(txt_img.region_score_map.cpu()).save(os.path.join(out_dir, region_map_fn))
            data["region_map_file"] = region_map_fn

        if txt_img.affinity_score_map is not None:
            affinity_map_fn = f"timg_{idx:04d}_affinity_map.png"
            to_pil_image(txt_img.affinity_score_map.cpu()).save(os.path.join(out_dir, affinity_map_fn))
            data["affinity_map_file"] = affinity_map_fn

        # 메타데이터 JSON 파일 저장
        with open(os.path.join(out_dir, meta_fn), "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    except Exception as e:
        print(f"[Save Error] Failed to save sample {idx}: {e}")


def save_timgs(timgs: List['TextedImage'], out_dir: str, num_threads: int = 8):
    """여러 개의 TextedImage 객체들을 병렬로 디스크에 저장합니다."""
    with ThreadPoolExecutor(max_workers=num_threads) as pool:
        futures = [pool.submit(_save_one, i, ti, out_dir) for i, ti in enumerate(timgs)]
        for future in futures:
            try:
                future.result()
            except Exception as e:
                print(f"An error occurred in a save worker thread: {e}")
    print(f"[Pipeline] All {len(timgs)} timg/meta saved to {out_dir}")


def _load_one(json_path: str, base_dir: str, device: torch.device, flags: Dict[str, bool]) -> 'TextedImage':
    """
    [Helper] 단일 메타데이터 파일을 기반으로 TextedImage 객체를 로드합니다.
    flags 딕셔셔리에 따라 필요한 구성 요소만 선택적으로 로드합니다.
    """

    with open(json_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)

    # 각 구성 요소를 None 또는 기본값으로 초기화
    orig, timg, mask, bboxes = None, None, None, []
    char_infos, region_map, affinity_map = None, None, None

    # flags 딕셔너리를 확인하여 선택적으로 파일 로드
    if flags.get("load_orig", False) and "orig_file" in meta:
        orig_path = os.path.join(base_dir, meta["orig_file"])
        orig = to_tensor(Image.open(orig_path).convert("RGB"))

    if flags.get("load_timg", False) and "timg_file" in meta:
        timg_path = os.path.join(base_dir, meta["timg_file"])
        timg = to_tensor(Image.open(timg_path).convert("RGB"))

    if flags.get("load_mask", False) and "mask_file" in meta:
        mask_path = os.path.join(base_dir, meta["mask_file"])
        mask = to_tensor(Image.open(mask_path).convert("L"))

    if flags.get("load_bboxes", False) and "bboxes" in meta:
        bboxes = [BBox(*pts) for pts in meta.get("bboxes", [])]

    # CRAFT GT 데이터는 하나의 플래그로 제어
    if flags.get("load_craft_gt", False):
        if "char_infos_file" in meta:
            try:
                char_data = np.load(os.path.join(base_dir, meta["char_infos_file"]), allow_pickle=True)
                char_infos = [CharInfo(np.array(ci["polygon"]), ci["char_content"], ci["word_id"]) for ci in char_data]
            except Exception as e:
                print(f"Warning: Could not load char_infos file {meta['char_infos_file']}: {e}")

        if "region_map_file" in meta:
            try:
                region_pil = Image.open(os.path.join(base_dir, meta["region_map_file"])).convert("L")
                region_map = to_tensor(region_pil)
            except Exception as e:
                print(f"Warning: Could not load region_map file {meta['region_map_file']}: {e}")

        if "affinity_map_file" in meta:
            try:
                affinity_pil = Image.open(os.path.join(base_dir, meta["affinity_map_file"])).convert("L")
                affinity_map = to_tensor(affinity_pil)
            except Exception as e:
                print(f"Warning: Could not load affinity_map file {meta['affinity_map_file']}: {e}")

    # 필수 필드(orig 또는 timg) 중 하나는 로드되어야 객체가 의미를 가짐
    if orig is None and timg is None:
        raise ValueError(f"Neither orig nor timg could be loaded for {json_path}. Cannot create TextedImage.")

    # orig나 timg가 없을 경우, 다른 하나로 채워줌 (방어적 코딩)
    ref_img = orig if orig is not None else timg
    if orig is None: orig = ref_img.clone()
    if timg is None: timg = ref_img.clone()

    # mask가 없을 경우, 이미지 크기에 맞는 빈 마스크 생성
    if mask is None:
        mask = torch.zeros(1, ref_img.shape[1], ref_img.shape[2])

    # 모든 텐서를 지정된 디바이스로 이동
    if orig is not None:
        orig = orig.to(device)
    if timg is not None:
        timg = timg.to(device)
    if mask is not None:
        mask = mask.to(device)
    if region_map is not None:
        region_map = region_map.to(device)
    if affinity_map is not None:
        affinity_map = affinity_map.to(device)

    return TextedImage(orig, timg, mask, bboxes, char_infos, region_map, affinity_map)


def load_timgs(
        base_dir: str,
        device: torch.device,
        mode: LoadingMode = LoadingMode.FULL,
        max_num: Optional[int] = None,
        shuffle: bool = False,
        num_threads: int = 8
) -> List['TextedImage']:
    """
    지정된 디렉토리에서 TextedImage 데이터 세트를 병렬로 로드합니다.

    Args:
        base_dir (str): 데이터가 저장된 기본 디렉토리.
        device (torch.device): 텐서를 로드할 디바이스.
        mode (LoadMode): 로딩 정책을 정의하는 Enum 멤버.
        max_num (Optional[int]): 로드할 최대 샘플 수. None이면 모든 샘플을 로드합니다.
        shuffle (bool): 로드하기 전에 파일 목록을 섞을지 여부.
        num_threads (int): 병렬 로딩에 사용할 스레드 수.

    Returns:
        List[TextedImage]: 로드된 TextedImage 객체들의 리스트.

    Raises:
        FileNotFoundError: 지정된 디렉토리에서 JSON 메타 파일을 찾을 수 없는 경우.
        ValueError: 유효하지 않은 `mode`가 입력된 경우.
    """
    # 1. 모드에 따른 로딩 플래그 설정
    flags: Dict[str, bool] = {
        "load_orig": False, "load_timg": False, "load_mask": False,
        "load_bboxes": False, "load_craft_gt": False,
    }

    if mode == LoadingMode.FULL:
        flags = {key: True for key in flags}

    elif mode == LoadingMode.MODEL1_TRAIN:  # oimg 제외 전체 (CRAFT 학습)
        flags = {key: True for key in flags}
        flags["load_orig"] = False

    elif mode == LoadingMode.MODEL1_INFER:  # timg만 로드
        flags["load_timg"] = True

    elif mode == LoadingMode.MODEL2_TRAIN:  # timg, mask, bboxes
        flags["load_timg"] = True
        flags["load_mask"] = True
        flags["load_bboxes"] = True

    elif mode == LoadingMode.MODEL2_INFER:  # timg, bboxes
        flags["load_timg"] = True
        flags["load_bboxes"] = True

    elif mode == LoadingMode.MODEL3_TRAIN:  # oimg, timg, mask, bboxes
        flags["load_orig"] = True
        flags["load_timg"] = True
        flags["load_mask"] = True
        flags["load_bboxes"] = True

    elif mode == LoadingMode.MODEL3_INFER:  # timg, mask, bboxes
        # 설명에 "bbox만" 이라고 되어 있지만, 일반적으로 bboxes를 의미하는 것으로 해석
        flags["load_timg"] = True
        flags["load_mask"] = True
        flags["load_bboxes"] = True

    elif mode == LoadingMode.INFERENCE:  # oimg만 로드
        flags["load_orig"] = True
    else:
        raise ValueError(f"Invalid load mode: {mode}. Please use a member of LoadMode Enum.")

    print(f"--- Loading TextedImages with mode: {mode.name} ---")
    print(f"Load flags: {flags}")

    # 2. 파일 목록 준비
    json_files = sorted(glob.glob(os.path.join(base_dir, "timg_*.json")))
    if not json_files:
        raise FileNotFoundError(f"No JSON meta files found in '{base_dir}'")

    if shuffle:
        random.shuffle(json_files)
    if max_num is not None and max_num > 0:
        json_files = json_files[:max_num]

    print(f"Found {len(json_files)} samples to load.")

    # 3. 병렬 로딩 실행
    timgs: List[TextedImage] = []
    with ThreadPoolExecutor(max_workers=num_threads) as pool:
        # submit 작업을 먼저 모두 등록
        futures = [pool.submit(_load_one, jf, base_dir, device, flags) for jf in json_files]
        # 모든 작업이 완료될 때까지 결과를 수집
        for future in futures:
            try:
                result = future.result()
                if result:  # 로딩이 성공적으로 완료된 경우에만 추가
                    timgs.append(result)
            except Exception as e:
                # 특정 파일 로딩 실패 시 에러 출력 후 계속 진행
                print(f"An error occurred in a load worker thread: {e}")

    print(f"Successfully loaded {len(timgs)} TextedImage objects from '{base_dir}'")
    return timgs