import os
import random
import numpy as np
import torch
import torchvision.transforms.functional as VTF
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from typing import List, Tuple, Optional, Dict
from tqdm import tqdm

from .Utils import BBox, Lang, UNICODE_RANGES
from .TextedImage import TextedImage, CharInfo
from ..Utils import PipelineSetting, ImagePolicy
import cv2



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
            self, num_images: int, dir_path: str, generate_craft_gt: bool = True
    ) -> List[TextedImage]:
        clear_pils: list[Image.Image] = []
        if not self.setting.use_noise:
            _paths = [os.path.join(dir_path, fn) for fn in os.listdir(dir_path) if
                      fn.lower().endswith((".png", ".jpg", ".jpeg"))]
            if num_images < len(_paths): _paths = random.sample(_paths, k=num_images)
            # print(f"[ImageLoader] 로딩 가능 이미지 {len(_paths)}/{num_images}")
            for _path in _paths:
                try:
                    clear_pils.append(Image.open(_path).convert("RGB"))
                except Exception as e:
                    print(f"Error loading image {_path}: {e}")
        num_noise_imgs = num_images - len(clear_pils)
        if num_noise_imgs > 0:
            h_img, w_img = self.setting.model1_input_size
            noise_arrays = np.random.randint(0, 256, (num_noise_imgs, h_img, w_img, 3), dtype=np.uint8)
            for noise_array in noise_arrays: clear_pils.append(Image.fromarray(noise_array, "RGB"))

        texted_images_list: List[TextedImage] = []
        for clear_pil_img in tqdm(clear_pils, leave=False, desc="Generating Texted Images"):
            # generate_craft_gt 플래그 전달
            texted_images_list.append(
                self.pil_to_texted_image(clear_pil_img, generate_craft_gt)
            )
        return texted_images_list

    def pil_to_texted_image(
            self, pil_img: Image.Image, generate_craft_gt: bool = False, debug_block:bool = False
    ) -> TextedImage:
        img_w, img_h = pil_img.size
        # orig_tensor = VTF.to_tensor(pil_img.convert("RGB")).to(self.setting.device)
        # timg_tensor = orig_tensor.clone()
        # text_block_pixel_mask = torch.zeros((1, img_h, img_w), device=self.setting.device)
        orig_tensor = VTF.to_tensor(pil_img.convert("RGB"))  # .to(self.setting.device) 제거
        timg_tensor = orig_tensor.clone()
        text_block_pixel_mask = torch.zeros((1, img_h, img_w))  # .to(self.setting.device) 제거
        word_level_bboxes_list: List[BBox] = []

        # CRAFT GT 생성을 위한 변수 초기화
        all_char_infos_for_craft: Optional[List[CharInfo]] = None
        region_score_map_tensor: Optional[torch.Tensor] = None
        affinity_score_map_tensor: Optional[torch.Tensor] = None

        if generate_craft_gt:
            all_char_infos_for_craft = []

        num_render_texts = random.randint(*self.policy.num_texts)
        current_text_block_id_base = 0

        for i_block in range(num_render_texts):
            # 1. 텍스트 덩어리 생성 및 렌더링 준비 (기존 로직과 거의 동일)
            is_sfx = random.random() < self.policy.sfx_style_prob
            font_size_r = random.uniform(*self.policy.font_size_ratio_to_image_height_range)
            if is_sfx: font_size_r = random.uniform(self.policy.font_size_ratio_to_image_height_range[1], min(0.3,
                                                                                                              self.policy.font_size_ratio_to_image_height_range[
                                                                                                                  1] * 2.5))
            font_size_val = max(12, int(img_h * font_size_r))
            font_obj = self._get_random_font(font_size_val)
            if not font_obj: continue
            text_str_content = self._get_random_text_content()
            wrapped_text_str_multiline = text_str_content
            if random.random() < self.policy.multiline_prob:
                max_textbox_w_px = int(img_w * random.uniform(*self.policy.textbox_width_ratio_to_image_width_range))
                wrapped_text_str_multiline = self._wrap_text_pil(text_str_content, font_obj, max_textbox_w_px)

            # 2. 수정된 함수 호출 (이 함수는 아래에 새로 정의/수정됨)
            rendered_text_block_pil, char_infos_relative_to_block_pil = \
                self._render_text_block_and_get_char_infos(  # 함수 이름 변경 및 역할 확장
                    wrapped_text_str_multiline, font_obj, is_sfx, generate_craft_gt,f"block_{i_block}_{random.randint(1000,9999) if debug_block else None}"
                )

            if rendered_text_block_pil is None: continue

            # 3. 렌더링된 텍스트 덩어리 PIL 이미지 합성 (기존 로직과 거의 동일)
            text_pil_w, text_pil_h = rendered_text_block_pil.size
            max_abs_x = img_w - text_pil_w;
            max_abs_y = img_h - text_pil_h
            if max_abs_x < 0 or max_abs_y < 0: continue
            abs_x_on_img = random.randint(0, max_abs_x);
            abs_y_on_img = random.randint(0, max_abs_y)
            current_text_block_bbox = BBox(abs_x_on_img, abs_y_on_img,
                                           abs_x_on_img + text_pil_w, abs_y_on_img + text_pil_h)
            word_level_bboxes_list.append(current_text_block_bbox)

            rgba_pil_tensor = VTF.to_tensor(rendered_text_block_pil) #.to(self.setting.device)
            rgb_pil_tensor = rgba_pil_tensor[:3, :, :];
            alpha_pil_tensor = rgba_pil_tensor[3:4, :, :]
            # TextedImage._alpha_blend 호출은 기존과 동일하다고 가정 (내부 로직은 이전 답변 참고)
            timg_tensor = TextedImage._alpha_blend(timg_tensor, current_text_block_bbox, rgb_pil_tensor,
                                                   alpha_pil_tensor)
            current_text_mask_tensor = (alpha_pil_tensor > 0).float()
            # text_block_pixel_mask 업데이트도 기존과 동일
            text_block_pixel_mask[:, current_text_block_bbox.y1:current_text_block_bbox.y2,
            current_text_block_bbox.x1:current_text_block_bbox.x2] = \
                torch.maximum(text_block_pixel_mask[:, current_text_block_bbox.y1:current_text_block_bbox.y2,
                              current_text_block_bbox.x1:current_text_block_bbox.x2], current_text_mask_tensor)

            # 4. CharInfo 처리
            if generate_craft_gt and char_infos_relative_to_block_pil and all_char_infos_for_craft is not None:
                for char_info_rel in char_infos_relative_to_block_pil:
                    abs_polygon = char_info_rel.polygon + np.array([abs_x_on_img, abs_y_on_img])
                    final_word_id = current_text_block_id_base + char_info_rel.word_id
                    all_char_infos_for_craft.append(
                        CharInfo(abs_polygon, char_info_rel.char_content, final_word_id)
                    )

            if generate_craft_gt and all_char_infos_for_craft is not None:
                max_rel_word_id_in_block = -1
                if char_infos_relative_to_block_pil:
                    for ci_rel in char_infos_relative_to_block_pil:
                        if ci_rel.word_id > max_rel_word_id_in_block:
                            max_rel_word_id_in_block = ci_rel.word_id
                current_text_block_id_base += (max_rel_word_id_in_block + 1)

        # 5. Score map 생성 (이 함수는 아래에 새로 정의됨)
        if generate_craft_gt and all_char_infos_for_craft and len(all_char_infos_for_craft) > 0:
            target_h_map, target_w_map = img_h // 2, img_w // 2
            region_score_map_tensor, affinity_score_map_tensor = \
                self._create_craft_gt_maps(
                    all_char_infos_for_craft, (target_h_map, target_w_map)
                )

        return TextedImage(
            orig=orig_tensor.cpu(),  # 명시적으로 CPU로 전달 (만약 GPU에 있었다면)
            timg=timg_tensor.cpu(),  # 명시적으로 CPU로 전달
            mask=text_block_pixel_mask.cpu(),  # 명시적으로 CPU로 전달
            bboxes=word_level_bboxes_list,
            all_char_infos=all_char_infos_for_craft,
            region_score_map=region_score_map_tensor.cpu() if region_score_map_tensor is not None else None,
            affinity_score_map=affinity_score_map_tensor.cpu() if affinity_score_map_tensor is not None else None,
        )

    def _render_text_block_and_get_char_infos(  # 함수 시그니처 변경
            self, text_content_str: str, font_obj: ImageFont.FreeTypeFont,
            is_sfx_style_policy: bool, extract_char_info_flag: bool,
            debug_visualization_prefix: Optional[str] = None# 플래그 추가
    ) -> Tuple[Optional[Image.Image], Optional[List[CharInfo]]]:  # 반환 타입 변경

        # --- 1. 텍스트 덩어리 렌더링 준비 (기존 _render_text_layer 로직과 매우 유사) ---
        # 이 부분은 기존 _render_text_layer의 앞부분 (색상, 외곽선, 그림자, 정렬, 줄간격,
        # 패딩 계산, 초기 레이어 생성, 텍스트 그리기) 코드를 대부분 가져옵니다.
        # 주의: 이 과정에서 "변형 전" 각 문자의 위치를 계산하기 위한 정보들을
        # 잘 기록하거나 계산할 수 있는 형태로 유지해야 합니다.
        # (예: 각 라인의 y 시작점, 각 라인 내 문자들의 x 오프셋, 사용된 정렬 방식 등)
        text_color_val = self._get_random_rgba();
        stroke_w_val = 0;
        stroke_fill_val = None
        # ... (stroke, shadow 파라미터 계산 로직) ...
        if random.random() < self.policy.stroke_prob or is_sfx_style_policy:
            s_r = random.uniform(*self.policy.stroke_width_ratio_to_font_size_range);
            stroke_w_val = max(self.policy.stroke_width_limit_px[0],
                               min(int(font_obj.size * s_r), self.policy.stroke_width_limit_px[1]))
            if is_sfx_style_policy: stroke_w_val = min(max(stroke_w_val, int(font_obj.size * 0.15)),
                                                       self.policy.stroke_width_limit_px[
                                                           1] * 2); stroke_fill_val = self._get_random_rgba()
        shadow_params_val = None
        if random.random() < self.policy.shadow_prob or is_sfx_style_policy:
            s_off_x_r = random.uniform(*self.policy.shadow_offset_x_ratio_to_font_size_range);
            s_off_y_r = random.uniform(*self.policy.shadow_offset_y_ratio_to_font_size_range);
            s_blur = random.randint(*self.policy.shadow_blur_radius_range);
            s_color = self._get_random_rgba()
            if is_sfx_style_policy: s_off_x_r *= 1.5;s_off_y_r *= 1.5;s_blur = max(s_blur, 3);s_color = (s_color[0],
                                                                                                         s_color[1],
                                                                                                         s_color[2],
                                                                                                         min(255, int(
                                                                                                             s_color[
                                                                                                                 3] * 1.5)))
            shadow_params_val = (int(font_obj.size * s_off_x_r), int(font_obj.size * s_off_y_r), s_blur, s_color)
        final_text_str = text_content_str;
        text_align_opt = random.choice(self.policy.text_align_options)
        line_spacing_ratio = random.uniform(*self.policy.line_spacing_ratio_to_font_size_range);
        line_spacing_pixels = int(font_obj.size * (line_spacing_ratio - 1.0))
        _draw_dummy = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
        try:
            text_bbox_on_pil = _draw_dummy.textbbox((0, 0), final_text_str, font=font_obj, stroke_width=stroke_w_val,
                                                    spacing=line_spacing_pixels, align=text_align_opt)
        except TypeError:
            text_bbox_on_pil = _draw_dummy.textbbox((0, 0), final_text_str, font=font_obj); text_bbox_on_pil = (
                text_bbox_on_pil[0] - stroke_w_val, text_bbox_on_pil[1] - stroke_w_val,
                text_bbox_on_pil[2] + stroke_w_val,
                text_bbox_on_pil[3] + stroke_w_val) if stroke_w_val > 0 else text_bbox_on_pil
        text_render_w = text_bbox_on_pil[2] - text_bbox_on_pil[0];
        text_render_h = text_bbox_on_pil[3] - text_bbox_on_pil[1]
        pad_f = 0.75;
        pad_x = int(text_render_w * pad_f);
        pad_y = int(text_render_h * pad_f)
        pad_extra = stroke_w_val + (abs(shadow_params_val[0]) + shadow_params_val[2] if shadow_params_val else 0) + (
            abs(shadow_params_val[1]) + shadow_params_val[2] if shadow_params_val else 0)
        pad_x += pad_extra;
        pad_y += pad_extra
        initial_padded_layer_w = int(max(1, text_render_w + pad_x * 2));
        initial_padded_layer_h = int(max(1, text_render_h + pad_y * 2))
        draw_orig_x_in_padded = pad_x - text_bbox_on_pil[0];
        draw_orig_y_in_padded = pad_y - text_bbox_on_pil[1]
        text_layer_img_untransformed = Image.new("RGBA", (initial_padded_layer_w, initial_padded_layer_h), (0, 0, 0, 0))
        draw_on_layer = ImageDraw.Draw(text_layer_img_untransformed)
        if shadow_params_val:  # ... (그림자 그리기) ...
            s_off_x, s_offset_y, s_blur_radius, s_color_rgba = shadow_params_val
            shadow_layer_img = Image.new("RGBA", text_layer_img_untransformed.size, (0, 0, 0, 0));
            shadow_draw_obj = ImageDraw.Draw(shadow_layer_img)
            shadow_draw_obj.text((draw_orig_x_in_padded + s_off_x, draw_orig_y_in_padded + s_offset_y), final_text_str,
                                 font=font_obj, fill=s_color_rgba, stroke_width=stroke_w_val,
                                 stroke_fill=stroke_fill_val, spacing=line_spacing_pixels, align=text_align_opt)
            if s_blur_radius > 0: shadow_layer_img = shadow_layer_img.filter(ImageFilter.GaussianBlur(s_blur_radius))
            text_layer_img_untransformed.alpha_composite(shadow_layer_img)
        draw_on_layer.text((draw_orig_x_in_padded, draw_orig_y_in_padded), final_text_str, font=font_obj,
                           fill=text_color_val, stroke_width=stroke_w_val, stroke_fill=stroke_fill_val,
                           spacing=line_spacing_pixels, align=text_align_opt)

        # -------------수정 시작---------------
        # --- 2. 변형 전 각 문자의 폴리곤 계산 (char_polygons_untransformed_in_padded_layer) ---
        untransformed_char_polygons_with_info: List[Tuple[np.ndarray, str, int]] = []
        if extract_char_info_flag:
            current_relative_word_id_in_block = 0
            lines = final_text_str.split('\n')

            # Pillow의 textbbox는 전체 텍스트 블록의 bbox를 (spacing, align 고려하여) 반환.
            # 이 bbox의 top-left를 기준으로 각 라인, 각 문자의 상대 위치를 계산해야 함.
            # text_bbox_on_pil[0]과 text_bbox_on_pil[1]은 텍스트 블록의 좌상단 좌표.
            # draw_orig_x_in_padded와 draw_orig_y_in_padded는 이 좌상단이 패딩된 이미지에 그려지는 위치.

            # 각 라인의 높이를 정확히 알기 위해 font.getsize 또는 font.getmask 사용 필요.
            # Pillow의 text() 함수는 내부적으로 복잡한 로직으로 위치를 결정함.
            # 이를 정확히 모방하기보다, 그려진 결과를 바탕으로 추정하거나,
            # 문자별로 직접 렌더링하여 bbox를 얻는 것이 더 정확할 수 있음.
            # 여기서는 textbbox 결과를 바탕으로 최대한 근사.

            # 라인별 시작 y 좌표 계산을 위한 변수.
            # text_bbox_on_pil[1]은 이미 stroke와 spacing의 영향을 일부 받은 텍스트 블록의 시작 y.
            # draw_orig_y_in_padded가 패딩된 레이어에서 텍스트 블록이 시작되는 y.
            current_line_y_start_in_padded = draw_orig_y_in_padded

            for line_text_str in lines:
                if not line_text_str.strip():  # 빈 줄 처리
                    # 빈 줄의 높이를 대략적으로 계산 (예: 'A' 문자의 높이) + 줄 간격
                    try:
                        # Pillow 9.2.0+ .getmask().size, older .getsize()
                        empty_line_h = font_obj.getmask("A").size[1] if hasattr(font_obj.getmask("A"), 'size') else \
                        font_obj.getsize("A")[1]
                    except AttributeError:  # getsize도 없는 아주 오래된 버전 또는 이상한 폰트 대비
                        empty_line_h = font_obj.size  # 최후의 보루
                    current_line_y_start_in_padded += empty_line_h + line_spacing_pixels
                    current_relative_word_id_in_block += 1
                    continue

                # 현재 라인의 렌더링된 너비 계산 (정렬에 사용)
                # 주의: _draw_dummy.textbbox는 (0,0) 기준 bbox. stroke, spacing=0으로 하여 순수 텍스트 너비 계산
                try:
                    line_bbox_for_align = _draw_dummy.textbbox((0, 0), line_text_str, font=font_obj,
                                                               stroke_width=stroke_w_val, spacing=0)  # spacing=0 중요!
                    line_actual_render_width = line_bbox_for_align[2] - line_bbox_for_align[0]
                    # 라인 높이도 유사하게 (ascent + descent)
                    # font.getmetrics() 사용 가능 (Pillow 9.metrics[0] + metrics[1])
                    # 간단히는 getbbox로 y 범위.
                    line_actual_render_height = line_bbox_for_align[3] - line_bbox_for_align[1]

                except TypeError:  # Older Pillow
                    line_actual_render_width = font_obj.getsize(line_text_str)[0] if hasattr(font_obj, 'getsize') else 0
                    line_actual_render_height = font_obj.getsize("A")[1] if hasattr(font_obj,
                                                                                    'getsize') else font_obj.size

                # 라인 정렬에 따른 x 시작 위치 조정
                # draw_orig_x_in_padded는 전체 텍스트 블록의 시작 x.
                # text_render_w는 전체 텍스트 블록의 너비.
                current_line_x_start_in_padded = draw_orig_x_in_padded
                if text_align_opt == 'center':
                    current_line_x_start_in_padded += (text_render_w - line_actual_render_width) / 2.0
                elif text_align_opt == 'right':
                    current_line_x_start_in_padded += (text_render_w - line_actual_render_width)

                current_char_x_offset_in_line = 0  # 현재 라인 내에서 문자의 x축 누적 오프셋
                for char_val in line_text_str:
                    # 각 문자의 바운딩 박스 (폰트 원점 기준, stroke 고려)
                    # font.getbbox()는 (left, top, right, bottom)을 반환. top/bottom은 baseline 기준.
                    try:
                        # stroke_width를 getbbox에 전달하면 stroke 포함된 bbox 반환
                        char_font_bbox = font_obj.getbbox(char_val, stroke_width=stroke_w_val)
                        # 문자의 가로 전진 폭 (advance width). getlength 사용 (Pillow 9.2.0+)
                        char_advance = font_obj.getlength(char_val)  # stroke_width 고려 안함. 순수 문자 폭.
                    except AttributeError:  # Older Pillow
                        # getmask().getbbox()는 stroke_width 고려 안함.
                        mask = font_obj.getmask(char_val)
                        char_font_bbox = mask.getbbox() if mask else (0, 0, 0, 0)  # (x0,y0,x1,y1) 래스터 이미지의 bbox
                        char_advance = font_obj.getsize(char_val)[0] if hasattr(font_obj, 'getsize') else (
                                    char_font_bbox[2] - char_font_bbox[0])

                    # 패딩된 레이어 내에서의 절대 좌표 계산
                    # char_font_bbox[0]은 문자 원점(보통 글자 왼쪽)에서 bbox 왼쪽까지의 x 오프셋.
                    # char_font_bbox[1]은 baseline에서 bbox 상단까지의 y 오프셋 (음수일 수 있음).
                    # current_line_y_start_in_padded는 현재 라인의 상단 y좌표.
                    # Pillow의 text() 함수는 baseline에 맞춰 그리므로, y좌표 계산 시 주의.
                    # 여기서는 라인 상단 + char_font_bbox[1] (폰트 bbox의 top)으로 근사.
                    # 더 정확하려면 폰트의 ascent 값을 알아야 함.
                    # font_ascent, font_descent = font_obj.getmetrics() # Pillow 9+
                    # y_baseline_in_padded = current_line_y_start_in_padded + font_ascent (근사)

                    char_x1 = current_line_x_start_in_padded + current_char_x_offset_in_line + char_font_bbox[0]
                    char_y1 = current_line_y_start_in_padded + char_font_bbox[1]  # 라인 상단 기준 bbox top
                    char_x2 = current_line_x_start_in_padded + current_char_x_offset_in_line + char_font_bbox[2]
                    char_y2 = current_line_y_start_in_padded + char_font_bbox[3]  # 라인 상단 기준 bbox bottom

                    untransformed_poly = np.array(
                        [[char_x1, char_y1], [char_x2, char_y1], [char_x2, char_y2], [char_x1, char_y2]],
                        dtype=np.float32)
                    untransformed_char_polygons_with_info.append(
                        (untransformed_poly, char_val, current_relative_word_id_in_block))

                    # 다음 문자 x 시작 위치는 현재 문자의 advance 폭만큼 이동
                    current_char_x_offset_in_line += char_advance
                    if char_val == ' ' and char_advance == 0:  # 공백인데 advance가 0인 경우 (일부 폰트)
                        try:
                            char_advance = font_obj.getmask(" ").size[0] * 0.8  # 근사치
                        except:
                            char_advance = font_obj.size * 0.25
                        current_char_x_offset_in_line += char_advance

                # 다음 라인의 y 시작 위치 업데이트
                current_line_y_start_in_padded += line_actual_render_height + line_spacing_pixels
                current_relative_word_id_in_block += 1
        if debug_visualization_prefix and extract_char_info_flag and untransformed_char_polygons_with_info:
            img_vis_untransformed = text_layer_img_untransformed.copy()
            draw_vis_untransformed = ImageDraw.Draw(img_vis_untransformed)
            for poly, char_c, word_id in untransformed_char_polygons_with_info:
                # 폴리곤 그리기 (초록색)
                draw_vis_untransformed.polygon([tuple(p) for p in poly.tolist()], outline=(0, 255, 0, 128), width=1)
                # 중심점 계산 및 그리기 (빨간색 점)
                center_x = np.mean(poly[:, 0])
                center_y = np.mean(poly[:, 1])
                r = 2
                draw_vis_untransformed.ellipse((center_x - r, center_y - r, center_x + r, center_y + r),
                                               fill=(255, 0, 0, 200))

            # 시각화 이미지 저장 (경로는 적절히 수정 필요)
            vis_save_dir = self.setting.debug_dir
            if not os.path.exists(vis_save_dir): os.makedirs(vis_save_dir, exist_ok=True)
            img_vis_untransformed.save(
                os.path.join(vis_save_dir, f"{debug_visualization_prefix}_untransformed_chars.png"))
        # -------------수정 시작---------------
        # 이 부분은 이미지 변환과 좌표 변환을 동기화하는 핵심 로직입니다.
        # Pillow 이미지 변환 함수들은 변환 행렬을 직접 반환하지 않으므로,
        # 각 변환에 해당하는 3x3 동차 변환 행렬을 직접 구성하고,
        # 이를 문자 좌표에 순차적으로 적용해야 합니다.

        img_to_transform_pil = text_layer_img_untransformed
        # T_coords_total: 변형 전 좌표계(untransformed_char_polygons)에서
        # 최종 변형 후 좌표계(final_transformed_img의 crop 전 이미지)로의 누적 변환 행렬
        T_coords_total = np.eye(3, dtype=np.float64)  # 3x3 단위 행렬 (동차 좌표계용)

        # 3.1 회전 (Rotation)
        # Pillow의 rotate(expand=True)는 이미지 중심 기준 회전 후, 내용이 잘리지 않도록 이미지 크기를 확장.
        # 이와 동일한 변환을 좌표에 적용해야 함.
        current_rot_range = self.policy.rotation_angle_range
        if is_sfx_style_policy:
            current_rot_range = (min(-90, self.policy.rotation_angle_range[0] * 2),
                                 max(90, self.policy.rotation_angle_range[1] * 2))
        angle_deg_to_apply = random.uniform(*current_rot_range)

        W0, H0 = img_to_transform_pil.size  # 회전 전 이미지 크기
        center_x0, center_y0 = W0 / 2.0, H0 / 2.0  # 회전 중심 (이미지 중심)

        # Pillow 이미지 회전
        img_after_rotation_pil = img_to_transform_pil.rotate(
            angle_deg_to_apply, expand=True, resample=Image.Resampling.BICUBIC
        )
        W1, H1 = img_after_rotation_pil.size  # 회전 후 확장된 이미지 크기
        center_x1, center_y1 = W1 / 2.0, H1 / 2.0  # 새 이미지의 중심

        # 좌표 변환 행렬 계산:
        # 1. 원본 이미지 중심을 원점(0,0)으로 이동 (T_to_origin)
        # 2. 원점 기준 회전 (R_matrix)
        # 3. 확장된 새 이미지의 중심으로 이동 (T_to_new_center)
        # M_rotation_for_coords = T_to_new_center @ R_matrix @ T_to_origin
        T_to_origin = np.array([[1, 0, -center_x0],
                                [0, 1, -center_y0],
                                [0, 0, 1]], dtype=np.float64)
        angle_rad = np.radians(-angle_deg_to_apply)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        R_matrix = np.array([[cos_a, -sin_a, 0],
                             [sin_a, cos_a, 0],
                             [0, 0, 1]], dtype=np.float64)
        T_to_new_center = np.array([[1, 0, center_x1],
                                    [0, 1, center_y1],
                                    [0, 0, 1]], dtype=np.float64)
        M_rotation_for_coords = T_to_new_center @ R_matrix @ T_to_origin

        T_coords_total = M_rotation_for_coords @ T_coords_total
        img_to_transform_pil = img_after_rotation_pil  # 다음 변환을 위해 이미지 업데이트
        current_W, current_H = W1, H1  # 현재 이미지 크기 업데이트

        # 3.2 기울이기 (Shear / Affine Transform)
        M_shear_for_coords = np.eye(3, dtype=np.float64)  # 단위 행렬로 초기화
        if random.random() < self.policy.shear_apply_prob or is_sfx_style_policy:
            shear_x_factor = random.uniform(*self.policy.shear_x_factor_range)
            shear_y_factor = random.uniform(*self.policy.shear_y_factor_range)
            if is_sfx_style_policy:
                shear_x_factor *= 1.5
                shear_y_factor *= 1.5

            # Pillow의 affine transform 계수: (a, b, c, d, e, f)
            # x_new = a*x + b*y + c
            # y_new = d*x + e*y + f
            # M_shear_for_coords는 이에 해당하는 3x3 행렬:
            # [[a, b, c],
            #  [d, e, f],
            #  [0, 0, 1]]
            pil_affine_coeffs = None
            if abs(shear_x_factor) > 0.01:  # x축 방향 기울이기
                pil_affine_coeffs = (1, shear_x_factor, 0,  # a, b, c
                                     0, 1, 0)  # d, e, f
                M_shear_for_coords = np.array([[1, -shear_x_factor, 0],
                                               [0, 1, 0],
                                               [0, 0, 1]], dtype=np.float64)
            elif abs(shear_y_factor) > 0.01:  # y축 방향 기울이기
                pil_affine_coeffs = (1, 0, 0,  # a, b, c
                                     shear_y_factor, 1, 0)  # d, e, f
                M_shear_for_coords = np.array([[1, 0, 0],
                                               [-shear_y_factor, 1, 0],
                                               [0, 0, 1]], dtype=np.float64)

            if pil_affine_coeffs:
                img_to_transform_pil = img_to_transform_pil.transform(
                    (current_W, current_H),  # 현재 이미지 크기 사용
                    Image.Transform.AFFINE,
                    pil_affine_coeffs,
                    resample=Image.Resampling.BICUBIC
                )
                T_coords_total = M_shear_for_coords @ T_coords_total
                # 중요: Affine 변환은 이미지 크기를 변경하지 않음. current_W, current_H 유지.

        # 3.3 원근 변형 (Perspective Transform)
        M_perspective_for_coords = np.eye(3, dtype=np.float64)  # 단위 행렬로 초기화
        if random.random() < self.policy.perspective_transform_enabled_prob or is_sfx_style_policy:
            # _get_perspective_coeffs_numpy는 H_forward (src -> dst)의 8개 계수를 반환한다고 가정.
            # H = [[a,b,c],[d,e,f],[g,h,1]]
            # Pillow의 transform(PERSPECTIVE, data)는 H_inverse의 8개 계수를 data로 받음.
            H_forward_coeffs_tuple = self._get_perspective_coeffs_numpy(
                current_W, current_H, self.policy.perspective_transform_strength_ratio_range
            )
            if H_forward_coeffs_tuple:
                # 정방향 원근 변환 행렬 H_forward 구성
                M_perspective_for_coords = np.array([
                    [H_forward_coeffs_tuple[0], H_forward_coeffs_tuple[1], H_forward_coeffs_tuple[2]],
                    [H_forward_coeffs_tuple[3], H_forward_coeffs_tuple[4], H_forward_coeffs_tuple[5]],
                    [H_forward_coeffs_tuple[6], H_forward_coeffs_tuple[7], 1.0]  # H[2,2]는 보통 1
                ], dtype=np.float64)

                try:
                    # Pillow에 전달할 역방향 변환 행렬 H_inverse 계산
                    H_inverse_matrix = np.linalg.inv(M_perspective_for_coords)
                    pil_perspective_coeffs = tuple(H_inverse_matrix.flatten()[:8])

                    img_to_transform_pil = img_to_transform_pil.transform(
                        (current_W, current_H),  # 현재 이미지 크기 사용
                        Image.Transform.PERSPECTIVE,
                        pil_perspective_coeffs,
                        resample=Image.Resampling.BICUBIC
                    )
                    T_coords_total = M_perspective_for_coords @ T_coords_total
                    # 중요: Perspective 변환은 이미지 크기를 변경하지 않음. current_W, current_H 유지.
                except (np.linalg.LinAlgError, ValueError) as e:
                    # print(f"Perspective transform failed: {e}. Skipping.")
                    M_perspective_for_coords = np.eye(3, dtype=np.float64)  # 실패 시 변환 없음으로 처리
                    pass  # H_forward_coeffs_tuple이 유효했으나 역행렬 계산 실패 등

        final_transformed_img_pil = img_to_transform_pil  # 모든 변환이 적용된 최종 이미지
        final_bbox_in_transformed_img = final_transformed_img_pil.getbbox()

        if final_bbox_in_transformed_img is None:
            return None, None  # 변환 후 이미지가 비었으면 실패 처리

        cropped_final_text_img = final_transformed_img_pil.crop(final_bbox_in_transformed_img)
        if cropped_final_text_img.width == 0 or cropped_final_text_img.height == 0:
            return None, None  # 크롭 후 이미지 크기가 0이면 실패

        # --- 4. 변형된 문자 폴리곤 계산 및 CharInfo 생성 ---
        char_infos_list_relative_to_cropped: Optional[List[CharInfo]] = None
        if extract_char_info_flag and untransformed_char_polygons_with_info:
            char_infos_list_relative_to_cropped = []
            # 크롭 오프셋: final_bbox_in_transformed_img의 (x1, y1)
            crop_offset_x = final_bbox_in_transformed_img[0]
            crop_offset_y = final_bbox_in_transformed_img[1]

            for poly_untransformed, char_s, rel_word_id in untransformed_char_polygons_with_info:
                # poly_untransformed (4,2) -> homogeneous (4,3) (각 점에 1 추가)
                poly_h = np.hstack([poly_untransformed, np.ones((poly_untransformed.shape[0], 1))])

                # 누적된 변환 행렬 T_coords_total 적용: P'_h = T_total @ P_h.T  (P_h가 열벡터 형태일 때)
                # poly_h.T는 (3,4) 행렬, T_coords_total은 (3,3) 행렬. 결과는 (3,4)
                # 다시 .T 하여 (4,3) 형태로 변환
                transformed_poly_h = (T_coords_total @ poly_h.T).T

                # 동차 좌표를 데카르트 좌표로 변환 (w 성분으로 나누기)
                # transformed_poly_h의 마지막 열(w 성분)이 0에 가까우면 문제 발생 가능성 (무한대 좌표)
                # 이 경우 해당 문자는 건너뛰거나 다른 처리 필요.
                w_coords = transformed_poly_h[:, 2]
                if np.any(np.abs(w_coords) < 1e-7):  # w가 0에 매우 가까운 경우
                    # print(f"Warning: char '{char_s}' resulted in near-zero w after perspective. Skipping.")
                    continue  # 이 문자는 GT 생성에서 제외

                transformed_poly_cartesian = transformed_poly_h[:, :2] / w_coords[:, np.newaxis]

                # 최종 크롭 영역 기준 상대 좌표로 변환
                poly_relative_to_crop = transformed_poly_cartesian - np.array([crop_offset_x, crop_offset_y])

                char_infos_list_relative_to_cropped.append(
                    CharInfo(poly_relative_to_crop, char_s, rel_word_id)
                )

        return cropped_final_text_img, char_infos_list_relative_to_cropped

    def _create_craft_gt_maps(
            self,
            all_char_infos_abs_coords: List[CharInfo],  # CharInfo의 polygon은 이미지 전체 절대 좌표
            target_map_output_size: Tuple[int, int],  # (H_out, W_out), 예: (H/2, W/2)
            gaussian_variance_scale_factor: float = 0.25,  # 표준편차 = short_side * factor
            affinity_gaussian_scale_factor_major: float = 0.3,  # 주축 표준편차 = dist * factor
            affinity_gaussian_scale_factor_minor: float = 0.15,  # 부축 표준편차 = avg_h * factor
            min_char_size_on_map: int = 2  # 1/2 스케일 맵 기준 최소 문자 크기
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        주어진 문자 정보(절대 좌표 폴리곤) 리스트로부터
        Region Score Map과 Affinity Score Map을 생성합니다.
        맵 크기는 target_map_output_size (일반적으로 원본 이미지의 1/2) 입니다.
        """
        target_h_out, target_w_out = target_map_output_size
        region_map_np = np.zeros((target_h_out, target_w_out), dtype=np.float32)
        affinity_map_np = np.zeros((target_h_out, target_w_out), dtype=np.float32)

        # 메쉬그리드는 한 번만 생성하여 재사용
        x_coords_mg = np.arange(target_w_out)
        y_coords_mg = np.arange(target_h_out)
        xx_mg, yy_mg = np.meshgrid(x_coords_mg, y_coords_mg)

        # 1. Region Score Map 생성 (회전된 가우시안 사용)
        for char_info in all_char_infos_abs_coords:
            if char_info.polygon is None or char_info.polygon.shape[0] < 3:
                continue
            poly_on_map = char_info.polygon / 2.0
            poly_on_map_int = np.array(poly_on_map, dtype=np.int32)

            try:
                rect_min_area_char = cv2.minAreaRect(poly_on_map_int.reshape(-1, 1, 2))
            except cv2.error:
                continue

            (cx_on_map, cy_on_map), (rect_w_on_map, rect_h_on_map), angle_on_map_cv = rect_min_area_char

            std_dev_x_axis_aligned = (rect_w_on_map / 2.0) * gaussian_variance_scale_factor
            std_dev_y_axis_aligned = (rect_h_on_map / 2.0) * gaussian_variance_scale_factor
            std_dev_x_axis_aligned = max(std_dev_x_axis_aligned, 0.5)
            std_dev_y_axis_aligned = max(std_dev_y_axis_aligned, 0.5)
            variance_x_sq = std_dev_x_axis_aligned ** 2 + 1e-6
            variance_y_sq = std_dev_y_axis_aligned ** 2 + 1e-6
            angle_rad_char_coords = np.radians(-angle_on_map_cv)
            cos_a = np.cos(angle_rad_char_coords)
            sin_a = np.sin(angle_rad_char_coords)

            # Affinity Map에서 사용된 것과 동일한 xx_mg, yy_mg를 사용합니다.
            # 이 xx_mg, yy_mg는 루프 밖에서 (target_h_out, target_w_out) shape으로 정의되어 있습니다.
            x_coords_shifted = xx_mg - cx_on_map  # Shape: (target_h_out, target_w_out)
            y_coords_shifted = yy_mg - cy_on_map  # Shape: (target_h_out, target_w_out)

            # 이 연산은 (H,W) shape 배열 간의 요소별 연산이므로 문제가 없어야 합니다.
            x_coords_rotated = x_coords_shifted * cos_a - y_coords_shifted * sin_a
            y_coords_rotated = x_coords_shifted * sin_a + y_coords_shifted * cos_a

            gaussian_char_map_values = np.exp(
                -((x_coords_rotated ** 2 / (2 * variance_x_sq)) +
                  (y_coords_rotated ** 2 / (2 * variance_y_sq)))
            )

            char_fill_mask_on_map = np.zeros_like(region_map_np, dtype=np.uint8)
            if poly_on_map_int.shape[0] > 0:
                cv2.fillPoly(char_fill_mask_on_map, [poly_on_map_int.reshape(-1, 2)], 1)

            region_map_np = np.maximum(region_map_np, gaussian_char_map_values * char_fill_mask_on_map)

        # 2. Affinity Score Map 생성
        words_data_dict: Dict[int, List[CharInfo]] = {}
        for char_info in all_char_infos_abs_coords:
            if char_info.word_id not in words_data_dict:
                words_data_dict[char_info.word_id] = []
            words_data_dict[char_info.word_id].append(char_info)

        for word_id_val, chars_list_in_word in words_data_dict.items():
            if len(chars_list_in_word) < 2:  # 단어 내 문자가 2개 이상일 때만 어피니티 존재
                continue

            # 문자 순서 정렬 (일반적으로 왼쪽에서 오른쪽, 위에서 아래 순서)
            # 여기서는 각 문자의 첫 번째 꼭짓점의 x좌표, 그 다음 y좌표 기준으로 정렬
            sorted_chars_in_word_list = sorted(
                chars_list_in_word, key=lambda ci: (ci.polygon[0, 0], ci.polygon[0, 1])
            )

            for i in range(len(sorted_chars_in_word_list) - 1):
                char1_info_obj = sorted_chars_in_word_list[i]
                char2_info_obj = sorted_chars_in_word_list[i + 1]

                # 두 문자의 폴리곤 (1/2 스케일)
                poly1_on_map = char1_info_obj.polygon / 2.0
                poly2_on_map = char2_info_obj.polygon / 2.0

                # 어피니티 박스(가우시안의 중심과 방향, 크기) 정의
                # CRAFT 논문 Figure 3(c) 및 Figure 4 참고

                # 1. 두 문자 폴리곤의 중심점 계산
                center1_on_map = np.mean(poly1_on_map, axis=0)
                center2_on_map = np.mean(poly2_on_map, axis=0)

                # 2. 어피니티 가우시안의 중심: 두 문자 중심의 중간점
                affinity_center_x_on_map = (center1_on_map[0] + center2_on_map[0]) / 2.0
                affinity_center_y_on_map = (center1_on_map[1] + center2_on_map[1]) / 2.0

                # 3. 어피니티 가우시안의 주축 방향 (두 문자 중심을 잇는 벡터의 각도)
                delta_x_map = center2_on_map[0] - center1_on_map[0]
                delta_y_map = center2_on_map[1] - center1_on_map[1]
                # angle_rad_affinity는 x축 양의 방향에서 두 중심을 잇는 벡터까지의 각도 (반시계 방향)
                angle_rad_affinity = np.arctan2(delta_y_map, delta_x_map)

                # 4. 어피니티 가우시안의 크기 (표준편차)
                #    주축(문자 연결 방향) 표준편차: 두 문자 중심 간 거리의 일정 비율
                #    부축(문자 연결에 수직 방향) 표준편차: 평균 문자 높이의 일정 비율
                distance_between_centers_on_map = np.linalg.norm(center1_on_map - center2_on_map)
                if distance_between_centers_on_map < 1e-3:  # 중심이 거의 같으면 어피니티 없음 (오류 방지)
                    continue

                std_dev_affinity_major_axis = distance_between_centers_on_map * affinity_gaussian_scale_factor_major

                # 각 문자의 높이 근사 (폴리곤의 y범위)
                h1_on_map = np.max(poly1_on_map[:, 1]) - np.min(poly1_on_map[:, 1])
                h2_on_map = np.max(poly2_on_map[:, 1]) - np.min(poly2_on_map[:, 1])
                avg_char_height_on_map = (h1_on_map + h2_on_map) / 2.0
                std_dev_affinity_minor_axis = avg_char_height_on_map * affinity_gaussian_scale_factor_minor

                # 표준편차 최소값 설정
                std_dev_affinity_major_axis = max(std_dev_affinity_major_axis, 0.5)
                std_dev_affinity_minor_axis = max(std_dev_affinity_minor_axis, 0.5)

                variance_affinity_major_sq = std_dev_affinity_major_axis ** 2 + 1e-6
                variance_affinity_minor_sq = std_dev_affinity_minor_axis ** 2 + 1e-6

                # 5. 회전된 2D 가우시안 생성
                # 좌표계를 (affinity_center_x_on_map, affinity_center_y_on_map)으로 이동시키고,
                # -angle_rad_affinity 만큼 회전시켜서 가우시안 주축이 x축과 정렬되도록 함.
                cos_a_aff = np.cos(-angle_rad_affinity)  # 반대 방향 회전
                sin_a_aff = np.sin(-angle_rad_affinity)

                x_coords_shifted = xx_mg - affinity_center_x_on_map
                y_coords_shifted = yy_mg - affinity_center_y_on_map

                x_coords_rotated_for_affinity = x_coords_shifted * cos_a_aff - y_coords_shifted * sin_a_aff
                y_coords_rotated_for_affinity = x_coords_shifted * sin_a_aff + y_coords_shifted * cos_a_aff

                gaussian_affinity_values = np.exp(
                    -((x_coords_rotated_for_affinity ** 2 / (2 * variance_affinity_major_sq)) +
                      (y_coords_rotated_for_affinity ** 2 / (2 * variance_affinity_minor_sq)))
                )

                # 어피니티 가우시안을 적용할 마스크 (두 문자 폴리곤을 포함하는 convex hull)
                # 또는 두 문자 폴리곤 사이를 채우는 더 정교한 모양 (사다리꼴 등)
                affinity_fill_mask_on_map = np.zeros_like(affinity_map_np, dtype=np.uint8)
                # 두 폴리곤의 모든 점을 합쳐서 convex hull을 만들고 채움
                combined_poly_for_affinity = np.concatenate(
                    (np.array(poly1_on_map, dtype=np.int32),
                     np.array(poly2_on_map, dtype=np.int32)),
                    axis=0
                )
                try:
                    hull_indices_affinity = cv2.convexHull(combined_poly_for_affinity, returnPoints=False)
                    # hull_indices가 Nx1 형태이므로, Nx2 형태로 만들기 위해 combined_poly_for_affinity에서 인덱싱
                    hull_points_affinity = combined_poly_for_affinity[hull_indices_affinity.flatten()]
                    cv2.fillPoly(affinity_fill_mask_on_map, [hull_points_affinity], 1)
                except cv2.error:  # 점이 너무 적거나 일직선상에 있을 때 convexHull 에러 발생 가능
                    # print(f"Warning: cv2.convexHull failed for affinity between chars. Using line mask.")
                    # 대신 두 중심을 잇는 두꺼운 선으로 마스크 생성 시도
                    line_thickness_affinity = int(max(1, avg_char_height_on_map * 0.2))  # 두께는 평균 문자 높이의 20%
                    cv2.line(affinity_fill_mask_on_map,
                             tuple(np.int32(center1_on_map)),
                             tuple(np.int32(center2_on_map)),
                             1, thickness=line_thickness_affinity)

                affinity_map_np = np.maximum(affinity_map_np, gaussian_affinity_values * affinity_fill_mask_on_map)

        # 최종 맵 값 [0, 1] 범위로 클리핑
        region_map_np = np.clip(region_map_np, 0, 1)
        affinity_map_np = np.clip(affinity_map_np, 0, 1)

        # PyTorch 텐서로 변환 (C, H, W)
        region_map_torch_tensor = torch.from_numpy(region_map_np).unsqueeze(0) #.to(self.setting.device)
        affinity_map_torch_tensor = torch.from_numpy(affinity_map_np).unsqueeze(0) #.to(self.setting.device)

        return region_map_torch_tensor, affinity_map_torch_tensor

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
        text_color = self._get_random_rgba()
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
            stroke_fill = self._get_random_rgba()

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
            return None

        cropped_layer = final_transformed_layer.crop(final_bbox_in_layer)
        # 반환값: 잘라낸 최종 텍스트 이미지
        return cropped_layer
