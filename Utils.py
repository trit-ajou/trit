import os
import torch
from dataclasses import dataclass

from .models.Utils import ModelMode


@dataclass
class PipelineSetting:
    device = torch.device("cpu")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 모델 돌릴 땐 GPU로 바꾸쇼
    use_amp = True
    num_workers = 4  # for imageloader

    _script_path = os.path.abspath(__file__)
    _script_dir = os.path.dirname(_script_path)
    font_dir = f"{_script_dir}/datas/fonts"
    clear_img_dir = f"{_script_dir}/datas/images/clear"
    output_img_dir = f"{_script_dir}/datas/images/output"  # Use for visualization
    ckpt_dir = f"{_script_dir}/datas/checkpoints"

    model1_input_size = (1700, 2400)
    model2_input_size = (256, 256)
    model3_input_size = (512, 512)

    num_images = 20
    use_noise = False
    margin = 4
    max_objects = 1024
    epochs = 100
    batch_size = 4
    lr = 0.001
    weight_decay = 3e-4
    train_valid_split = 0.2

    model1_mode = ModelMode.SKIP
    model2_mode = ModelMode.SKIP
    model3_mode = ModelMode.SKIP

    vis_interval = 5
    ckpt_interval = 1


from dataclasses import dataclass, field
from typing import Literal


@dataclass
class ImagePolicy:
    # --- 기본 설정 ---
    num_texts: tuple[int, int] = (1, 5)
    text_length_range: tuple[int, int] = (5, 20)
    # --- 폰트 크기 ---
    # 텍스트 높이를 이미지 높이에 대한 비율로 설정
    font_size_ratio_to_image_height_range: tuple[float, float] = (0.01, 0.015)

    # --- 여러 줄 텍스트 ---
    multiline_prob: float = 0.7  # 여러 줄 텍스트 사용 확률
    # 여러 줄 텍스트 시, 텍스트 박스 너비를 이미지 너비에 대한 비율로 설정
    textbox_width_ratio_to_image_width_range: tuple[float, float] = (0.2, 0.8)
    # 줄 간격을 폰트 크기에 대한 비율로 설정 (예: 폰트 크기의 1.0배 ~ 1.5배)
    line_spacing_ratio_to_font_size_range: tuple[float, float] = (1.0, 1.5)
    # 텍스트 정렬 (Pillow 기준: "left", "center", "right")
    text_align_options: list[Literal["left", "center", "right"]] = field(
        default_factory=lambda: ["left", "center", "right"]
    )

    # --- 텍스트 색상 ---
    # True면 완전 랜덤 RGB 색상, False면 fixed_text_color_options에서 무작위 선택
    # "주변 배경색 대비"는 실제 렌더링 시점에 해당 위치의 배경색을 분석하여 동적으로 결정해야 함.
    # 여기서는 정책으로 랜덤 또는 고정 색상 세트 중 선택으로 단순화.
    text_color_is_random: bool = True
    fixed_text_color_options: list[tuple[int, int, int]] = field(
        default_factory=lambda: [
            (0, 0, 0),
            (255, 255, 255),
            (200, 200, 200),
            (50, 50, 50),
        ]  # 검정, 흰색, 밝은회색, 어두운회색
    )
    # 불투명도
    opacity_range: tuple[int, int] = (220, 255)

    # --- 외곽선 ---
    stroke_prob: float = 0.6  # 외곽선 사용 확률
    # 외곽선 두께를 텍스트 크기(높이)에 대한 비율로 설정 + 최소/최대 픽셀 제한 가능. 예) 폰트 높이의 5~15%
    stroke_width_ratio_to_font_size_range: tuple[float, float] = (0.03, 0.5)
    stroke_width_limit_px: tuple[int, int] = (2, 5)  # 최소/최대 외곽선 두께 범위 (픽셀)
    # True면 완전 랜덤 RGB 색상, False면 fixed_stroke_color_options에서 무작위 선택
    stroke_color_is_random: bool = True
    fixed_stroke_color_options: list[tuple[int, int, int]] = field(
        default_factory=lambda: [
            (0, 0, 0),
            (255, 255, 255),
            (128, 128, 128),
            (255, 0, 0),
            (0, 0, 255),
        ]  # 검정, 흰색, 회색, 빨강, 파랑
    )

    # --- 그림자 ---
    shadow_prob: float = 0.4  # 그림자 사용 확률
    # 그림자 오프셋(x, y)을 텍스트 크기(높이)에 대한 비율로 설정
    shadow_offset_x_ratio_to_font_size_range: tuple[float, float] = (
        -0.1,
        0.1,
    )  # 음수/양수 가능(좌/우)
    shadow_offset_y_ratio_to_font_size_range: tuple[float, float] = (
        0.05,
        0.15,
    )  # 보통 아래로 떨어짐
    # 그림자 흐림 반경 (픽셀 단위). Pillow에는 직접적인 blur 기능이 없으므로, 여러 번 겹쳐 그리거나,
    # GaussianBlur 필터를 텍스트 레이어에 적용 후 합성하는 방식 필요.
    # 여기서는 단순화를 위해 그림자 색상의 알파값을 낮추는 것으로 간접 표현하거나,
    # 실제 구현 시 blur 반경 파라미터로 활용.
    shadow_blur_radius_range: tuple[int, int] = (1, 4)  # 실제 구현 시 이 값을 사용
    # 그림자 색상 (주로 검은색 계열). 알파값도 함께 설정.
    shadow_color: tuple[int, int, int] = (0, 0, 0)  # 그림자 기본 색상 (RGB)
    shadow_opacity: tuple[float, float] = (0.3, 0.7)  # 그림자 불투명도 (알파값)

    # --- 변형 (Transformations) ---
    # 회전 각도 범위 (도)
    rotation_angle_range: tuple[float, float] = (-20.0, 20.0)

    # 기울이기 (Shear) - Pillow의 Affine transform (a, b, c, d, e, f)에서 b(x축 shear), d(y축 shear) 값.
    # 여기서는 각 축에 대한 shear factor 범위를 지정.
    shear_x_factor_range: tuple[float, float] = (-0.3, 0.3)
    shear_y_factor_range: tuple[float, float] = (-0.3, 0.3)
    # 둘 다 0이면 shear 없음. 둘 중 하나 또는 둘 다 랜덤하게 적용.
    shear_apply_prob: float = 0.5  # Shear 자체를 적용할 확률

    # 왜곡 (Distortion) - 여기서는 Perspective Transform을 활용
    perspective_transform_enabled_prob: float = 0.3  # 원근 왜곡 사용 확률
    # 원근 왜곡 강도 (변화량의 최대 비율). 값이 클수록 왜곡이 심해짐.
    # 이미지의 네 꼭짓점을 얼마나 이동시킬지에 대한 비율.
    perspective_transform_strength_ratio_range: tuple[float, float] = (0.05, 0.2)

    # --- SFX (효과음) 스타일 ---
    # SFX 스타일은 위의 파라미터들을 더 극단적으로 사용하도록 유도.
    # 이 플래그가 True이면, 실제 값을 생성하는 로직에서
    # 예를 들어 font_size_ratio를 더 크게, rotation_angle 범위를 더 넓게,
    # 외곽선을 더 두껍고 화려하게, 변형을 더 강하게 적용.
    sfx_style_prob: float = 0.0  # SFX 스타일 적용 확률
