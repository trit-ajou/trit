import os
import random
import numpy as np
import torch
import torchvision.transforms.functional as VTF
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageChops
from typing import List, Tuple, Optional, Dict

from .Utils import TextedImage, BBox, Lang, UNICODE_RANGES
from ..Utils import PipelineSetting, ImagePolicy


class ImageLoader:
    def __init__(
        self,
        setting: PipelineSetting,
        policy: ImagePolicy,
        preload_font_sizes: Optional[List[int]] = None,
    ):  # 폰트 크기 미리 로드 옵션
        self.setting = setting
        self.policy = policy

        self.font_paths = self._load_font_paths(setting.font_dir)  # 폰트 경로만 로드
        if not self.font_paths:
            print(f"Warning: No font paths found in {setting.font_dir}.")

        # 폰트 객체 캐시 (경로 -> 크기 -> 폰트 객체) 또는 (경로_크기 -> 폰트 객체)
        # 여기서는 (경로, 크기) 튜플을 키로 사용하는 간단한 딕셔너리 사용
        self.font_cache: Dict[Tuple[str, int], ImageFont.FreeTypeFont] = {}

        # (선택적) 미리 정의된 크기로 폰트 미리 로드하기
        if preload_font_sizes and self.font_paths:
            print(f"Preloading fonts for sizes: {preload_font_sizes}...")
            for path in self.font_paths:
                for size in preload_font_sizes:
                    try:
                        self.font_cache[(path, size)] = ImageFont.truetype(path, size)
                    except Exception as e:
                        # print(f"Could not preload font {path} at size {size}: {e}")
                        pass
            print("Font preloading complete.")

    def _load_font_paths(self, font_dir: str) -> List[str]:  # _load_fonts에서 이름 변경
        if not os.path.isdir(font_dir):
            return []
        return [
            os.path.join(font_dir, item)
            for item in os.listdir(font_dir)
            if item.lower().endswith((".ttf", ".otf", ".ttc"))
        ]

    def _get_font(self, font_size_px: int) -> Optional[ImageFont.FreeTypeFont]:
        """캐시에서 폰트를 가져오거나, 없다면 로드하고 캐시에 추가합니다."""
        if not self.font_paths:
            return None

        font_path = random.choice(self.font_paths)  # 랜덤 폰트 경로 선택

        # 캐시 확인
        if (font_path, font_size_px) in self.font_cache:
            return self.font_cache[(font_path, font_size_px)]

        # 캐시에 없다면 로드
        try:
            font = ImageFont.truetype(font_path, font_size_px)
            self.font_cache[(font_path, font_size_px)] = font  # 캐시에 저장
            return font
        except Exception as e:
            # print(f"Font load error {font_path} at size {font_size_px}: {e}")
            # 캐시에 로드 실패 기록 (선택적, 반복적 시도 방지)
            self.font_cache[(font_path, font_size_px)] = None  # type: ignore
            return None

    @staticmethod
    def get_random_char(lang: Lang) -> str:
        if lang not in UNICODE_RANGES or not UNICODE_RANGES[lang]:
            return "?"
        block = random.choice(UNICODE_RANGES[lang])
        return chr(random.randint(block[0], block[1]))

    @staticmethod
    def get_random_text_content(min_length: int, max_length: int) -> str:
        content = ""
        length = random.randint(min_length, max_length)
        available_langs = list(Lang)
        if not available_langs:
            return "?"
        for _ in range(length):
            content += ImageLoader.get_random_char(random.choice(available_langs))
        return content

    def _get_random_font_path(self) -> Optional[str]:
        return random.choice(self.available_fonts) if self.available_fonts else None

    def _get_rgba_color(
        self,
        base_color_options: List[Tuple[int, int, int]],
        is_random: bool,
        opacity_range: Tuple[float, float],
    ) -> Tuple[int, int, int, int]:
        opacity = int(random.uniform(*opacity_range) * 255)
        if is_random:
            color_rgb = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255),
            )
        else:
            color_rgb = (
                random.choice(base_color_options) if base_color_options else (0, 0, 0)
            )
        return (*color_rgb, opacity)

    def _wrap_text_pil(
        self, text: str, font: ImageFont.FreeTypeFont, max_width: int
    ) -> str:
        lines = []
        words = list(text)
        current_line = ""
        _draw_dummy = ImageDraw.Draw(Image.new("L", (1, 1)))
        for char in words:
            bbox = _draw_dummy.textbbox((0, 0), current_line + char, font=font)
            line_width = bbox[2] - bbox[0]
            if line_width <= max_width or not current_line:
                current_line += char
            else:
                lines.append(current_line)
                current_line = char
        if current_line:
            lines.append(current_line)
        return "\n".join(lines)

    def _get_perspective_coeffs_numpy(
        self, width: int, height: int, strength_ratio_range: Tuple[float, float]
    ) -> Optional[Tuple[float, ...]]:
        if width <= 0 or height <= 0:
            return None
        src_points = np.array(
            [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
            dtype=np.float32,
        )
        dst_points = np.zeros_like(src_points)
        strength_w = width * random.uniform(*strength_ratio_range)
        strength_h = height * random.uniform(*strength_ratio_range)
        for i in range(4):
            dx = random.uniform(-strength_w, strength_w)
            dy = random.uniform(-strength_h, strength_h)
            dst_points[i, 0] = src_points[i, 0] + dx
            dst_points[i, 1] = src_points[i, 1] + dy
        A = np.zeros((8, 8), dtype=np.float32)
        b = np.zeros((8, 1), dtype=np.float32)
        for i in range(4):
            x, y = src_points[i]
            xp, yp = dst_points[i]
            A[2 * i] = [x, y, 1, 0, 0, 0, -x * xp, -y * xp]
            A[2 * i + 1] = [0, 0, 0, x, y, 1, -x * yp, -y * yp]
            b[2 * i] = xp
            b[2 * i + 1] = yp
        try:
            return tuple(np.linalg.solve(A, b).flatten().tolist())
        except np.linalg.LinAlgError:
            return None

    def _render_text_layer(
        self, text_content: str, font: ImageFont.FreeTypeFont, policy_is_sfx: bool
    ) -> Tuple[Optional[Image.Image], Optional[Tuple[int, int, int, int]]]:
        text_color = self._get_rgba_color(
            self.policy.fixed_text_color_options,
            self.policy.text_color_is_random,
            self.policy.opacity,
        )
        stroke_width = 0
        stroke_fill = None
        if random.random() < self.policy.stroke_prob or policy_is_sfx:
            s_ratio = random.uniform(*self.policy.stroke_width_ratio_to_font_size_range)
            stroke_width = max(
                self.policy.stroke_width_limit_px[0],
                min(int(font.size * s_ratio), self.policy.stroke_width_limit_px[1]),
            )
            if policy_is_sfx:
                stroke_width = min(
                    max(stroke_width, int(font.size * 0.15)),
                    self.policy.stroke_width_limit_px[1] * 2,
                )
            stroke_fill = self._get_rgba_color(
                self.policy.fixed_stroke_color_options,
                self.policy.stroke_color_is_random,
                self.policy.opacity,
            )

        shadow_params = None
        if random.random() < self.policy.shadow_prob or policy_is_sfx:
            s_off_x_r = random.uniform(
                *self.policy.shadow_offset_x_ratio_to_font_size_range
            )
            s_off_y_r = random.uniform(
                *self.policy.shadow_offset_y_ratio_to_font_size_range
            )
            s_blur = random.randint(*self.policy.shadow_blur_radius_range)
            s_color = self._get_rgba_color(
                [self.policy.shadow_color], False, self.policy.shadow_opacity
            )
            if policy_is_sfx:
                s_off_x_r *= 1.5
                s_off_y_r *= 1.5
                s_blur = max(s_blur, 3)
                s_color = (
                    s_color[0],
                    s_color[1],
                    s_color[2],
                    min(255, int(s_color[3] * 1.5)),
                )
            shadow_params = (
                int(font.size * s_off_x_r),
                int(font.size * s_off_y_r),
                s_blur,
                s_color,
            )

        final_text = text_content
        text_align = random.choice(self.policy.text_align_options)
        line_spacing = int(
            font.size
            * (random.uniform(*self.policy.line_spacing_ratio_to_font_size_range) - 1.0)
        )

        _draw_dummy = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
        try:
            text_bbox = _draw_dummy.textbbox(
                (0, 0),
                final_text,
                font=font,
                stroke_width=stroke_width,
                spacing=line_spacing,
                align=text_align,
            )
        except TypeError:
            text_bbox = _draw_dummy.textbbox((0, 0), final_text, font=font)
            if stroke_width > 0:
                text_bbox = (
                    text_bbox[0] - stroke_width,
                    text_bbox[1] - stroke_width,
                    text_bbox[2] + stroke_width,
                    text_bbox[3] + stroke_width,
                )

        text_render_width = text_bbox[2] - text_bbox[0]
        text_render_height = text_bbox[3] - text_bbox[1]

        # *** 중요 수정: 변형을 위한 충분한 패딩 추가 ***
        # 예상 최대 변형 크기를 고려. 대각선 길이의 2배 정도 또는 고정값.
        # 회전 시 expand=True가 크기를 늘리지만, 다른 변형은 그렇지 않음.
        # 그림자, 외곽선, 회전, 기울이기, 원근왜곡 모두 고려.
        # 패딩을 텍스트 크기의 100% (양쪽으로 50%) 또는 그 이상으로 설정.
        padding_factor = 0.75  # 각 방향으로 75% 패딩 (총 150% 추가 크기)
        padding_x = int(text_render_width * padding_factor)
        padding_y = int(text_render_height * padding_factor)
        # 그림자나 스트로크가 매우 클 경우 패딩 추가
        padding_extra = (
            stroke_width
            + (abs(shadow_params[0]) + shadow_params[2] if shadow_params else 0)
            + (abs(shadow_params[1]) + shadow_params[2] if shadow_params else 0)
        )
        padding_x += padding_extra
        padding_y += padding_extra

        layer_width = int(max(1, text_render_width + padding_x * 2))
        layer_height = int(max(1, text_render_height + padding_y * 2))

        draw_origin_x = padding_x - text_bbox[0]
        draw_origin_y = padding_y - text_bbox[1]

        text_layer = Image.new("RGBA", (layer_width, layer_height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(text_layer)

        if shadow_params:
            s_offset_x, s_offset_y, s_blur_radius, s_color_rgba = shadow_params
            shadow_layer = Image.new("RGBA", text_layer.size, (0, 0, 0, 0))
            shadow_draw = ImageDraw.Draw(shadow_layer)
            shadow_draw.text(
                (draw_origin_x + s_offset_x, draw_origin_y + s_offset_y),
                final_text,
                font=font,
                fill=s_color_rgba,
                stroke_width=stroke_width,
                stroke_fill=stroke_fill,
                spacing=line_spacing,
                align=text_align,
            )
            if s_blur_radius > 0:
                shadow_layer = shadow_layer.filter(
                    ImageFilter.GaussianBlur(s_blur_radius)
                )
            text_layer.alpha_composite(shadow_layer)

        draw.text(
            (draw_origin_x, draw_origin_y),
            final_text,
            font=font,
            fill=text_color,
            stroke_width=stroke_width,
            stroke_fill=stroke_fill,
            spacing=line_spacing,
            align=text_align,
        )

        layer_to_transform = text_layer
        current_rotation_range = self.policy.rotation_angle_range
        if policy_is_sfx:
            current_rotation_range = (
                min(-90, self.policy.rotation_angle_range[0] * 2),
                max(90, self.policy.rotation_angle_range[1] * 2),
            )
        angle = random.uniform(*current_rotation_range)
        # 회전 시 expand=True 사용하면 레이어 크기가 변함. 이후 변형은 이 변한 크기 기준.
        transformed_layer_after_rotation = layer_to_transform.rotate(
            angle, expand=True, resample=Image.Resampling.BICUBIC
        )

        final_transformed_layer = transformed_layer_after_rotation
        if random.random() < self.policy.shear_apply_prob or policy_is_sfx:
            shear_x = random.uniform(*self.policy.shear_x_factor_range)
            shear_y = random.uniform(*self.policy.shear_y_factor_range)
            if policy_is_sfx:
                shear_x *= 1.5
                shear_y *= 1.5
            # Shear는 expand 옵션이 없음. 패딩된 레이어 내에서 변형됨.
            if abs(shear_x) > 0.01:
                final_transformed_layer = final_transformed_layer.transform(
                    final_transformed_layer.size,
                    Image.Transform.AFFINE,
                    (1, shear_x, 0, 0, 1, 0),
                    resample=Image.Resampling.BICUBIC,
                )
            elif abs(shear_y) > 0.01:
                final_transformed_layer = final_transformed_layer.transform(
                    final_transformed_layer.size,
                    Image.Transform.AFFINE,
                    (1, 0, 0, shear_y, 1, 0),
                    resample=Image.Resampling.BICUBIC,
                )

        if (
            random.random() < self.policy.perspective_transform_enabled_prob
            or policy_is_sfx
        ):
            current_layer_width, current_layer_height = final_transformed_layer.size
            perspective_coeffs = self._get_perspective_coeffs_numpy(
                current_layer_width,
                current_layer_height,
                self.policy.perspective_transform_strength_ratio_range,
            )
            if perspective_coeffs:
                try:
                    # Perspective도 expand 옵션 없음. 패딩된 레이어 내에서 변형.
                    final_transformed_layer = final_transformed_layer.transform(
                        final_transformed_layer.size,
                        Image.Transform.PERSPECTIVE,
                        perspective_coeffs,
                        resample=Image.Resampling.BICUBIC,
                    )
                except (ValueError, Exception):
                    pass

        final_bbox_in_layer = final_transformed_layer.getbbox()
        if final_bbox_in_layer is None:
            return None, None

        cropped_layer = final_transformed_layer.crop(final_bbox_in_layer)
        # 반환값: 잘라낸 최종 텍스트 이미지, (0,0,width,height) 형태의 BBox (잘라낸 이미지 기준)
        return cropped_layer, (0, 0, cropped_layer.width, cropped_layer.height)

    def generate_texted_image(
        self, original_pil_image: Image.Image, index: int
    ) -> Optional[TextedImage]:
        # original_pil_image는 이미 로드된 PIL 이미지라고 가정
        if not self.font_paths:
            print("Error: No font paths available.")
            return None

        orig_tensor = VTF.to_tensor(original_pil_image.convert("RGB"))
        timg_pil = original_pil_image.copy().convert("RGBA")
        mask_pil = Image.new("L", original_pil_image.size, 0)

        img_w, img_h = original_pil_image.size
        final_bboxes: List[BBox] = []
        num_texts_to_draw = random.randint(*self.policy.num_texts)

        for _ in range(num_texts_to_draw):
            text_content = self.get_random_text_content(
                self.setting.min_text_len, self.setting.max_text_len
            )
            if not text_content:
                continue

            is_sfx_style = random.random() < self.policy.sfx_style_prob
            font_size_ratio = random.uniform(
                *self.policy.font_size_ratio_to_image_height_range
            )
            if is_sfx_style:
                font_size_ratio = random.uniform(
                    self.policy.font_size_ratio_to_image_height_range[1],
                    min(
                        0.3, self.policy.font_size_ratio_to_image_height_range[1] * 2.5
                    ),
                )
            font_size_px = max(12, int(img_h * font_size_ratio))

            # 수정: _get_font 메소드를 통해 폰트 객체 가져오기
            font = self._get_font(font_size_px)
            if not font:
                # print(f"Warning: Could not get font for size {font_size_px}. Skipping text.")
                continue

            wrapped_text_content = text_content
            if random.random() < self.policy.multiline_prob:
                max_textbox_width_px = int(
                    img_w
                    * random.uniform(
                        *self.policy.textbox_width_ratio_to_image_width_range
                    )
                )
                wrapped_text_content = self._wrap_text_pil(
                    text_content, font, max_textbox_width_px
                )
                if not wrapped_text_content.strip():
                    continue

            rendered_text_img, bbox_in_rendered_img = self._render_text_layer(
                wrapped_text_content, font, is_sfx_style
            )
            if rendered_text_img is None or bbox_in_rendered_img is None:
                continue

            text_w, text_h = rendered_text_img.size
            if text_w == 0 or text_h == 0:
                continue

            # *** 중요 수정: 텍스트 배치 및 경계 처리 ***
            max_paste_x = img_w - text_w
            max_paste_y = img_h - text_h

            if max_paste_x < 0 or max_paste_y < 0:
                # print(f"Info: Rendered text ({text_w}x{text_h}) too large for image ({img_w}x{img_h}). Skipping text.")
                continue  # 텍스트가 이미지보다 크면 건너뜀

            paste_x = random.randint(0, max_paste_x)
            paste_y = random.randint(0, max_paste_y)

            timg_pil.paste(rendered_text_img, (paste_x, paste_y), rendered_text_img)

            # *** 중요 수정: 마스크 합성 방식 변경 ***
            text_alpha_for_mask = rendered_text_img.getchannel("A")
            # 임시 마스크 레이어 만들어서 현재 텍스트의 마스크만 포함하도록 함
            current_text_mask_on_full_image = Image.new("L", mask_pil.size, 0)
            current_text_mask_on_full_image.paste(
                text_alpha_for_mask, (paste_x, paste_y)
            )

            # 기존 마스크와 새 텍스트 마스크를 'lighter' 연산으로 합침 (픽셀 단위 max)
            mask_pil = ImageChops.lighter(mask_pil, current_text_mask_on_full_image)

            # *** 중요 수정: 최종 마스크 이진화 (0 또는 255) ***
            # 픽셀 값이 0보다 크면 255(흰색), 아니면 0(검은색)으로 만듦
            # Pillow의 point 함수를 사용하여 임계값 적용
            # lambda p: 255 if p > threshold else 0
            # 여기서는 threshold를 0으로 설정 (0보다 크면 모두 텍스트 영역으로 간주)
            binary_mask_pil = mask_pil.point(lambda p: 255 if p > 0 else 0, mode="L")
            # 또는 특정 임계값 (예: 128)을 사용할 수도 있습니다.
            # binary_mask_pil = mask_pil.point(lambda p: 255 if p > 128 else 0, mode='L')
            # BBox 좌표 계산 및 이미지 경계 내로 클리핑
            final_x1 = float(paste_x)
            final_y1 = float(paste_y)
            final_x2 = float(paste_x + text_w)
            final_y2 = float(paste_y + text_h)

            # 클리핑 (이미지 경계를 벗어나지 않도록)
            final_x1 = max(0.0, final_x1)
            final_y1 = max(0.0, final_y1)
            final_x2 = min(
                float(img_w), final_x2
            )  # img_w (exclusive) or img_w-1 (inclusive)
            final_y2 = min(float(img_h), final_y2)

            if (
                final_x2 > final_x1 and final_y2 > final_y1
            ):  # 유효한 BBox인 경우에만 추가
                final_bboxes.append(BBox(final_x1, final_y1, final_x2, final_y2))

        timg_tensor = VTF.to_tensor(timg_pil.convert("RGB"))
        mask_tensor = VTF.to_tensor(binary_mask_pil)

        return TextedImage(
            orig=orig_tensor,
            timg=timg_tensor,
            mask=mask_tensor,
            bboxes=final_bboxes,
            index=index,
        )


import matplotlib.pyplot as plt


def visualize_texted_image(
    texted_image_obj: Optional[TextedImage],
    save_path: str,
    show_bboxes: bool = True,
    figsize: tuple[int, int] = (15, 5),  # 그림 크기 조절
) -> None:
    """
    TextedImage 객체의 원본 이미지, 텍스트가 추가된 이미지, 마스크, BBox를 시각화하고 저장합니다.

    Args:
        texted_image_obj: 시각화할 TextedImage 객체. None이면 빈 이미지를 저장합니다.
        save_path: 이미지를 저장할 경로 (예: "sample_visualization.png").
        show_bboxes: 텍스트 이미지에 BBox를 표시할지 여부.
        figsize: matplotlib figure의 크기.
    """
    if texted_image_obj is None:
        print(f"TextedImage object is None. Saving a blank image to {save_path}")
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.text(
            0.5,
            0.5,
            "No TextedImage Data",
            ha="center",
            va="center",
            fontsize=16,
            color="red",
        )
        ax.set_xticks([])
        ax.set_yticks([])
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close(fig)
        return

    # 텐서를 PIL 이미지로 변환하는 함수
    def tensor_to_pil(tensor_img: torch.Tensor) -> Image.Image:
        # (C, H, W) -> (H, W, C)
        numpy_img = tensor_img.cpu().numpy().transpose(1, 2, 0)
        # 0-1 범위를 0-255 범위로 변환하고 uint8로 캐스팅
        if numpy_img.max() <= 1.0:  # 정규화된 텐서 (0~1)라고 가정
            numpy_img = (numpy_img * 255).astype(np.uint8)
        else:  # 이미 0-255 범위라고 가정 (만약 그렇다면)
            numpy_img = numpy_img.astype(np.uint8)

        if numpy_img.shape[2] == 1:  # 마스크의 경우 (H, W, 1) -> (H, W)
            return Image.fromarray(numpy_img.squeeze(), mode="L")  # 'L' (grayscale)
        return Image.fromarray(numpy_img, mode="RGB")

    orig_pil = tensor_to_pil(texted_image_obj.orig)
    timg_pil = tensor_to_pil(texted_image_obj.timg)
    mask_pil = tensor_to_pil(texted_image_obj.mask)  # 마스크는 단일 채널

    fig, axes = plt.subplots(1, 3, figsize=figsize)  # 원본, 텍스트, 마스크

    # 1. 원본 이미지
    axes[0].imshow(orig_pil)
    axes[0].set_title(f"Original (Index: {texted_image_obj.index})")
    axes[0].axis("off")

    # 2. 텍스트가 추가된 이미지 (BBox 포함 가능)
    axes[1].imshow(timg_pil)
    axes[1].set_title("Texted Image")
    axes[1].axis("off")

    if show_bboxes and texted_image_obj.bboxes:
        for bbox in texted_image_obj.bboxes:
            # BBox 좌표는 (x1, y1, x2, y2)
            rect = plt.Rectangle(
                (bbox.x1, bbox.y1),  # 좌상단 좌표
                bbox.width,  # 너비
                bbox.height,  # 높이
                linewidth=1.5,
                edgecolor="lime",  # 밝은 녹색
                facecolor="none",  # 채우기 없음
            )
            axes[1].add_patch(rect)

    # 3. 마스크 이미지
    axes[2].imshow(mask_pil, cmap="gray")  # 마스크는 흑백으로 표시
    axes[2].set_title("Text Mask")
    axes[2].axis("off")

    plt.tight_layout()  # 서브플롯 간 간격 자동 조절
    plt.savefig(save_path)
    print(f"Visualization saved to {save_path}")
    plt.close(fig)  # 메모리 해제
