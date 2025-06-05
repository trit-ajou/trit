import os
import random
import numpy as np
import torch
import torchvision.transforms.functional as VTF
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from typing import List, Tuple, Optional, Dict
from tqdm import tqdm
import multiprocessing

from .Utils import BBox, Lang, UNICODE_RANGES
from .TextedImage import TextedImage
from ..Utils import PipelineSetting, ImagePolicy


class ImageLoader:
    def __init__(
        self,
        setting: PipelineSetting,
        policy: ImagePolicy,
    ):
        self.setting = setting
        self.policy = policy
        # font path 미리 저장
        self.font_paths = [
            os.path.join(self.setting.font_dir, filename)
            for filename in os.listdir(self.setting.font_dir)
            if filename.lower().endswith((".ttf", ".otf", ".ttc"))
        ]
        if not self.font_paths:
            raise ValueError("[ImageLoader] 폰트를 추가해라 휴먼")
        # font cache 정의: key=(path, size)
        self.font_cache: Dict[Tuple[str, int], ImageFont.FreeTypeFont] = {}

    def load_images(
        self, num_images: int, dir: str, max_text_size: tuple[int, int]
    ) -> List[TextedImage]:
        clear_pils: list[Image.Image] = []
        # 노이즈 이미지 사용 안하는 경우
        if not self.setting.use_noise:
            _paths = [
                os.path.join(dir, filename)
                for filename in os.listdir(dir)
                if filename.lower().endswith((".png", ".jpg", ".jpeg"))
            ]
            # 랜덤 비복원 추출. 이미지가 num_images보다 부족하면 있는 만큼만 로드 후 나머지는 noise로 대체.
            if num_images < len(_paths):
                _paths = random.sample(_paths, k=num_images)
            print(f"[ImageLoader] 로딩 가능 이미지 {len(_paths)}/{num_images}")
            for _path in _paths:
                clear_pils.append(Image.open(_path).convert("RGB"))
        # 클리어 이미지 부족한 경우 또는 노이즈 이미지 사용 옵션이 켜져 있어 list가 비어 있는 경우
        num_noise_imgs = num_images - len(clear_pils)
        if num_noise_imgs > 0:
            h, w = self.setting.model1_input_size
            noise_imgs = np.random.randint(
                0, 256, (num_images, h, w, 3), dtype=np.uint8
            )
            for noise_img in noise_imgs:
                clear_pils.append(Image.fromarray(noise_img, "RGB"))
        # apply_async 사용해서 병렬로 TextedImage 생성
        print(
            f"[ImageLoader] Redering TextedImages with {self.setting.num_workers} workers"
        )
        texted_images = []
        with multiprocessing.Pool(self.setting.num_workers) as pool:
            results = [
                pool.apply_async(self.pil_to_texted_image, (clear_pil, max_text_size))
                for clear_pil in clear_pils
            ]
            for result in tqdm(results, total=len(results), leave=False):
                texted_images.append(result.get())
        return texted_images

    def pil_to_texted_image(self, pil: Image.Image, max_text_size: tuple[int, int]):
        w, h = pil.size
        orig = VTF.to_tensor(pil.convert("RGB")).to(self.setting.device)
        timg = orig.clone()
        mask = torch.zeros((1, h, w), device=self.setting.device)
        bboxes: List[BBox] = []
        
        # 텍스트 생성 루프
        num_texts = random.randint(*self.policy.num_texts)
        for i in range(num_texts):
            # 폰트 크기 설정 (기존 코드)
            is_sfx_style = random.random() < self.policy.sfx_style_prob
            font_size_ratio = random.uniform(*self.policy.font_size_ratio_to_image_height_range)
            if is_sfx_style:
                font_size_ratio = random.uniform(
                    self.policy.font_size_ratio_to_image_height_range[1],
                    min(0.3, self.policy.font_size_ratio_to_image_height_range[1] * 2.5),
                )
            font_size = max(12, int(h * font_size_ratio))
            
            # 폰트 선택 (기존 코드)
            font = self._get_random_font(font_size)
            if not font:
                continue
                
            # 텍스트 내용 생성 (기존 코드)
            _content = self._get_random_text_content()
            wrapped_text_content = _content
            if random.random() < self.policy.multiline_prob:
                max_textbox_width_px = int(w * random.uniform(*self.policy.textbox_width_ratio_to_image_width_range))
                wrapped_text_content = self._wrap_text_pil(_content, font, max_textbox_width_px)
                
            # 텍스트 렌더링 (기존 코드)
            text_pil = self._render_text_layer(wrapped_text_content, font, is_sfx_style)
            if text_pil is None:
                continue
                
            # 텍스트 크기 확인 및 클리핑 적용
            text_w, text_h = text_pil.size
            if text_w > max_text_size[0] or text_h > max_text_size[1]:
                # 클리핑 적용
                clip_width = min(text_w, max_text_size[0])
                clip_height = min(text_h, max_text_size[1])
                text_pil = text_pil.crop((0, 0, clip_width, clip_height))
                text_w, text_h = text_pil.size
                
            # 텍스트 위치 결정 (기존 코드)
            max_x = w - text_w
            max_y = h - text_h
            
            # 텍스트가 이미지보다 크면 (이 부분은 이제 클리핑으로 처리되므로 필요 없음)
            if w < text_w or h < text_h:
                raise ValueError("[ImageLoader] 텍스트가 이미지 크기를 초과했습니다. policy를 조절하십시오 인간.")
            
            # 텍스트가 max_text_size보다 크면 (이 부분도 클리핑으로 처리되므로 필요 없음)
            if text_w > max_text_size[0] or text_h > max_text_size[1]:
                raise ValueError("[ImageLoader] 텍스트가 최대 크기를 초과했습니다. policy를 조절하십시오 인간.")
            
            # 나머지 코드는 그대로 유지
            x = random.randint(0, max(0, max_x))
            y = random.randint(0, max(0, max_y))
            bbox = BBox(x, y, x + text_w, y + text_h)
            bboxes.append(bbox)
            
            _rgba = VTF.to_tensor(text_pil).to(self.setting.device)
            _rgb = _rgba[:3, :, :]
            _alpha = _rgba[3:4, :, :]
            timg = TextedImage._alpha_blend(timg, bbox, _rgb, _alpha)
            _mask = (_alpha > 0).float()
            mask[bbox.slice] = torch.maximum(mask[bbox.slice], _mask)
        
        return TextedImage(orig, timg, mask, bboxes)

    def _get_random_font(self, size: int) -> Optional[ImageFont.FreeTypeFont]:
        # 하나 선택
        _path = random.choice(self.font_paths)
        # 캐시에 있으면 리턴
        if (_path, size) in self.font_cache:
            return self.font_cache[(_path, size)]
        # 캐시에 없으면 캐싱 후 리턴
        try:
            font = ImageFont.truetype(_path, size)
            self.font_cache[(_path, size)] = font
            return font
        # 폰트 로딩 중 오류 발생 시 None으로 표시. 해당 폰트는 더 이상 로딩 안됨.
        except:
            print(f"[ImageLoader] {_path} 폰트 로딩 중 에러")
            self.font_cache[(_path, size)] = None
            return None

    @staticmethod
    def _get_random_char(lang: Lang) -> str:
        if lang not in UNICODE_RANGES or not UNICODE_RANGES[lang]:
            return "?"
        block = random.choice(UNICODE_RANGES[lang])
        return chr(random.randint(block[0], block[1]))

    def _get_random_text_content(self) -> str:
        content = ""
        length = random.randint(*self.policy.text_length_range)
        for _ in range(length):
            content += self._get_random_char(random.choice(list(Lang)))
        return content

    def _get_random_rgba(self):
        opacity = random.randint(*self.policy.opacity_range)
        color_rgb = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255),
        )
        return (*color_rgb, opacity)

    @staticmethod
    def _wrap_text_pil(text: str, font: ImageFont.FreeTypeFont, max_width: int) -> str:
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

    @staticmethod
    def _get_perspective_coeffs_numpy(
        width: int, height: int, strength_ratio_range: Tuple[float, float]
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
    ) -> Optional[Image.Image]:
        # 텍스트 색 설정
        if self.policy.text_color_is_random:
            text_color = self._get_random_rgba()
        else:
            text_color = random.choice(self.policy.fixed_text_color_options)
        # 외곽선 두께 색 설정
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
            if self.policy.stroke_color_is_random:
                stroke_fill = self._get_random_rgba()
            else:
                stroke_fill = random.choice(self.policy.fixed_stroke_color_options)
        # 그림자 설정
        shadow_params = None
        if random.random() < self.policy.shadow_prob or policy_is_sfx:
            s_off_x_r = random.uniform(
                *self.policy.shadow_offset_x_ratio_to_font_size_range
            )
            s_off_y_r = random.uniform(
                *self.policy.shadow_offset_y_ratio_to_font_size_range
            )
            s_blur = random.randint(*self.policy.shadow_blur_radius_range)
            s_color = self._get_random_rgba()
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
        # 텍스트 정렬 설정
        text_align = random.choice(self.policy.text_align_options)
        line_spacing = int(
            font.size
            * (random.uniform(*self.policy.line_spacing_ratio_to_font_size_range) - 1.0)
        )

        _draw_dummy = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
        try:
            text_bbox = _draw_dummy.textbbox(
                (0, 0),
                text_content,
                font=font,
                stroke_width=stroke_width,
                spacing=line_spacing,
                align=text_align,
            )
        except TypeError:
            text_bbox = _draw_dummy.textbbox((0, 0), text_content, font=font)
            if stroke_width > 0:
                text_bbox = (
                    text_bbox[0] - stroke_width,
                    text_bbox[1] - stroke_width,
                    text_bbox[2] + stroke_width,
                    text_bbox[3] + stroke_width,
                )

        text_render_width = text_bbox[2] - text_bbox[0]
        text_render_height = text_bbox[3] - text_bbox[1]

        # *** 변형을 위한 충분한 패딩 추가 ***
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
                text_content,
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
            text_content,
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
            return None

        cropped_layer = final_transformed_layer.crop(final_bbox_in_layer)
        # 반환값: 잘라낸 최종 텍스트 이미지
        return cropped_layer
