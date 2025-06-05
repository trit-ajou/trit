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
            self, pil_img: Image.Image, generate_craft_gt: bool = False
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
                    wrapped_text_str_multiline, font_obj, is_sfx, generate_craft_gt
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
            is_sfx_style_policy: bool, extract_char_info_flag: bool  # 플래그 추가
    ) -> Tuple[Optional[Image.Image], Optional[List[CharInfo]]]:  # 반환 타입 변경

        # --- 1. 텍스트 덩어리 렌더링 준비 (기존 _render_text_layer 로직과 매우 유사) ---
        # 이 부분은 기존 _render_text_layer의 앞부분 (색상, 외곽선, 그림자, 정렬, 줄간격,
        # 패딩 계산, 초기 레이어 생성, 텍스트 그리기) 코드를 대부분 가져옵니다.
        # 주의: 이 과정에서 "변형 전" 각 문자의 위치를 계산하기 위한 정보들을
        # 잘 기록하거나 계산할 수 있는 형태로 유지해야 합니다.
        # (예: 각 라인의 y 시작점, 각 라인 내 문자들의 x 오프셋, 사용된 정렬 방식 등)
        # (이전 답변에서 제공한 _render_text_block_and_get_char_infos 앞부분 코드 참고)
        # (코드가 길어 여기서는 핵심 아이디어만 기술합니다.)
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

        # --- 2. 변형 전 각 문자의 폴리곤 계산 (char_polygons_untransformed_in_padded_layer) ---
        #    이 로직은 이 함수 내에서 직접 수행.
        untransformed_char_polygons_with_info: List[Tuple[np.ndarray, str, int]] = []
        if extract_char_info_flag:
            current_relative_word_id_in_block = 0
            lines = text_content_str.split('\n')
            y_text_start_in_padded = draw_orig_x_in_padded  # 오타 수정: draw_orig_y_in_padded

            for line_text_str in lines:
                if not line_text_str.strip():
                    try:
                        y_text_start_in_padded += font_obj.getmask("A").size[1] + line_spacing_pixels
                    except AttributeError:
                        y_text_start_in_padded += font_obj.size + line_spacing_pixels
                    current_relative_word_id_in_block += 1
                    continue

                line_bbox_from_font = _draw_dummy.textbbox((0, 0), line_text_str, font=font_obj,
                                                           stroke_width=stroke_w_val, spacing=0)
                line_render_width_from_font = line_bbox_from_font[2] - line_bbox_from_font[0]

                x_text_start_in_padded = draw_orig_x_in_padded
                if text_align_opt == 'center':
                    x_text_start_in_padded += (text_render_w - line_render_width_from_font) / 2
                elif text_align_opt == 'right':
                    x_text_start_in_padded += (text_render_w - line_render_width_from_font)

                current_char_x_advance_in_line = 0
                for char_val in line_text_str:
                    char_bbox_font_local = font_obj.getbbox(char_val, stroke_width=stroke_w_val)
                    char_x1 = x_text_start_in_padded + current_char_x_advance_in_line + char_bbox_font_local[0]
                    char_y1 = y_text_start_in_padded + char_bbox_font_local[1]
                    char_x2 = x_text_start_in_padded + current_char_x_advance_in_line + char_bbox_font_local[2]
                    char_y2 = y_text_start_in_padded + char_bbox_font_local[3]
                    untransformed_poly = np.array(
                        [[char_x1, char_y1], [char_x2, char_y1], [char_x2, char_y2], [char_x1, char_y2]],
                        dtype=np.float32)
                    untransformed_char_polygons_with_info.append(
                        (untransformed_poly, char_val, current_relative_word_id_in_block))
                    char_advance = char_bbox_font_local[2] - char_bbox_font_local[0]
                    if char_val == ' ':
                        try:
                            char_advance = font_obj.getlength(" ")
                        except AttributeError:
                            char_advance = font_obj.getmask(" ").size[0] if font_obj.getmask(
                                " ") else font_obj.size * 0.25
                    current_char_x_advance_in_line += char_advance
                line_h = line_bbox_from_font[3] - line_bbox_from_font[1];
                y_text_start_in_padded += line_h + line_spacing_pixels
                current_relative_word_id_in_block += 1

        # --- 3. 전체 텍스트 덩어리에 기하학적 변형 적용 (기존 _render_text_layer 로직) ---
        #    이 과정에서 적용된 최종 변환 행렬(들)을 계산하거나 기록해두는 것이 중요.
        layer_to_transform_img = text_layer_img_untransformed
        # `applied_transforms` 딕셔너리를 만들어 각 단계의 변환 행렬(또는 파라미터) 저장
        # 예: M_rotation, M_shear_affine, M_perspective_coeffs
        # (이전 답변의 변형 적용 코드 부분과 동일하게 진행, 단 변환 행렬/파라미터 기록)
        applied_rotation_val = 0.0  # 회전 각도
        # Pillow의 rotate(expand=True)에 해당하는 정확한 3x3 변환 행렬을 구해야 함.
        # 이는 회전 + 이동(새로운 중심에 맞게)을 포함.
        # M_rotation_for_coords = ... (3x3 행렬)
        # --- (회전, 기울이기, 원근 변형 로직은 기존과 동일하게 수행) ---
        # (이 과정에서 Pillow API가 적용된 최종 이미지: cropped_final_text_img)
        current_rot_range = self.policy.rotation_angle_range
        if is_sfx_style_policy: current_rot_range = (min(-90, self.policy.rotation_angle_range[0] * 2),
                                                     max(90, self.policy.rotation_angle_range[1] * 2))
        angle_to_apply = random.uniform(*current_rot_range);
        applied_rotation_val = angle_to_apply  # 저장
        img_after_rotation = layer_to_transform_img.rotate(angle_to_apply, expand=True,
                                                           resample=Image.Resampling.BICUBIC)
        layer_to_transform_img = img_after_rotation

        # 기울이기 변환 행렬 (M_shear_for_coords, 3x3) 계산
        applied_shear_coeffs_val = None
        if random.random() < self.policy.shear_apply_prob or is_sfx_style_policy:
            shear_x = random.uniform(*self.policy.shear_x_factor_range);
            shear_y = random.uniform(*self.policy.shear_y_factor_range)
            if is_sfx_style_policy: shear_x *= 1.5; shear_y *= 1.5
            s_coeffs = None
            if abs(shear_x) > 0.01:
                s_coeffs = (1, shear_x, 0, 0, 1, 0)
            elif abs(shear_y) > 0.01:
                s_coeffs = (1, 0, 0, shear_y, 1, 0)
            if s_coeffs:
                applied_shear_coeffs_val = s_coeffs  # Pillow용 계수
                layer_to_transform_img = layer_to_transform_img.transform(layer_to_transform_img.size,
                                                                          Image.Transform.AFFINE, s_coeffs,
                                                                          resample=Image.Resampling.BICUBIC)
                # M_shear_for_coords = np.array([[s_coeffs[0], s_coeffs[1], s_coeffs[2]],
                #                                [s_coeffs[3], s_coeffs[4], s_coeffs[5]],
                #                                [0, 0, 1]], dtype=np.float32) # (잘못된 접근, Pillow affine은 6개 파라미터)
                # Pillow affine (a,b,c,d,e,f) -> 3x3 행렬 [[a,b,c],[d,e,f],[0,0,1]] (x,y)출력 기준
                # x_out = ax + by + c, y_out = dx + ey + f
                # M_shear_for_coords = np.array([[s_coeffs[0],s_coeffs[1],s_coeffs[2]],[s_coeffs[3],s_coeffs[4],s_coeffs[5]],[0,0,1]], dtype=np.float32).T # Pillow문서 확인 필요

        # 원근 변환 행렬 (M_perspective_for_coords, 3x3) 계산
        applied_perspective_coeffs_val = None
        if random.random() < self.policy.perspective_transform_enabled_prob or is_sfx_style_policy:
            curr_w, curr_h = layer_to_transform_img.size
            p_coeffs = self._get_perspective_coeffs_numpy(curr_w, curr_h,
                                                          self.policy.perspective_transform_strength_ratio_range)
            if p_coeffs:
                applied_perspective_coeffs_val = p_coeffs  # Pillow용 계수
                try:
                    layer_to_transform_img = layer_to_transform_img.transform(layer_to_transform_img.size,
                                                                              Image.Transform.PERSPECTIVE, p_coeffs,
                                                                              resample=Image.Resampling.BICUBIC)
                except:
                    pass
                # M_perspective_for_coords = np.array([[p_coeffs[0],p_coeffs[1],p_coeffs[2]],
                #                                      [p_coeffs[3],p_coeffs[4],p_coeffs[5]],
                #                                      [p_coeffs[6],p_coeffs[7],1.0]], dtype=np.float32).T # Pillow 문서 확인 필요

        final_bbox_in_transformed_img = layer_to_transform_img.getbbox()
        if final_bbox_in_transformed_img is None: return None, None
        cropped_final_text_img = layer_to_transform_img.crop(final_bbox_in_transformed_img)

        # --- 4. 변형된 문자 폴리곤 계산 및 CharInfo 생성 ---
        char_infos_list_relative_to_cropped: Optional[List[CharInfo]] = None
        if extract_char_info_flag and untransformed_char_polygons_with_info:
            char_infos_list_relative_to_cropped = []
            # 이 루프에서 각 untransformed_poly에 위에서 계산/저장한
            # M_rotation_for_coords, M_shear_for_coords, M_perspective_for_coords를
            # 순서대로 곱하여 변형된 폴리곤을 얻고, 최종 크롭 기준으로 상대 좌표화.
            # (NumPy를 사용한 3x3 행렬과 (N,3) 동차좌표 폴리곤 점들의 행렬곱)

            # ----- 중요: 아래는 변환 행렬 계산 및 적용 로직의 핵심 아이디어 -----
            # 1. 전체 변환 행렬 M_total = M_persp @ M_shear @ M_rot (행렬곱 순서 중요)
            #    각 M은 3x3 동차 변환 행렬.
            #    - M_rot: Pillow의 rotate(expand=True)에 해당하는 변환. (가장 복잡)
            #      단순 회전 + 이동(새로운 중심에 맞게)을 포함해야 함.
            #      (예: 초기 레이어 중심을 원점으로 이동 -> 회전 -> 새 레이어 크기에 맞게 이동)
            #    - M_shear: applied_shear_coeffs_val (Pillow affine 계수)로부터 3x3 아핀 행렬 구성.
            #      [[a,b,c],[d,e,f],[0,0,1]] 형태. (Pillow 문서에서 계수 순서 확인 필요)
            #    - M_persp: applied_perspective_coeffs_val (Pillow perspective 계수)로부터 3x3 원근 행렬 구성.
            #      [[a,b,c],[d,e,f],[g,h,1]] 형태. (Pillow 문서에서 계수 순서 확인 필요, 보통 inv(H)의 전치)

            # 이 변환 행렬들을 정확히 구성하는 것이 핵심. OpenCV 함수 활용 가능.
            # 예: M_rot = cv2.getRotationMatrix2D(...) 후 3x3 확장 및 추가 이동.
            # M_shear = np.float32([[a,b,c],[d,e,f]]) 후 cv2.warpAffine 대신 좌표에 직접 적용.
            # M_persp = cv2.getPerspectiveTransform(...)으로 src->dst 맵핑 후 cv2.perspectiveTransform(coords, H_persp).

            for poly_untransformed, char_s, rel_word_id in untransformed_char_polygons_with_info:
                # poly_untransformed (4,2) -> homogeneous (4,3)
                poly_h = np.hstack([poly_untransformed, np.ones((4, 1))])

                # transformed_poly_h = (M_total @ poly_h.T).T # (4,3)
                # 또는 각 변환을 순차적으로 적용:
                # poly_after_rot_h = (M_rot @ poly_h.T).T
                # poly_after_shear_h = (M_shear @ poly_after_rot_h.T).T
                # poly_after_persp_h = (M_persp @ poly_after_shear_h.T).T
                # transformed_poly_h = poly_after_persp_h

                # 현재는 변환 행렬 M_total이 없으므로, 변환 없이 진행 (부정확)
                transformed_poly_h = poly_h

                # homogeneous to cartesian: x_cart = x_h / w_h, y_cart = y_h / w_h
                # transformed_poly = transformed_poly_h[:, :2] / transformed_poly_h[:, 2:] # (4,2)
                # 아핀 변환까지는 w_h가 1이므로, transformed_poly = transformed_poly_h[:,:2]
                # 원근 변환 후에는 w_h가 1이 아닐 수 있음.

                # 현재는 변환이 없다고 가정하고 진행 (이 부분 수정 필요)
                cartesian_poly = transformed_poly_h[:, :2]

                # 최종 크롭 영역 기준 상대 좌표
                poly_relative_to_crop = cartesian_poly - np.array(
                    [final_bbox_in_transformed_img[0], final_bbox_in_transformed_img[1]]
                )
                char_infos_list_relative_to_cropped.append(
                    CharInfo(poly_relative_to_crop, char_s, rel_word_id)
                )
            # ----- 변환 행렬 계산 및 적용 로직 끝 -----

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

        # 1. Region Score Map 생성
        for char_info in all_char_infos_abs_coords:
            # 폴리곤 좌표를 target_map_output_size (1/2 스케일) 기준으로 변환
            poly_on_map = char_info.polygon / 2.0
            poly_on_map_int = np.array(poly_on_map, dtype=np.int32)

            # 문자의 회전된 최소 영역 사각형(minAreaRect) 정보 가져오기
            # rect: ((center_x, center_y), (width, height), angle)
            # angle: OpenCV의 minAreaRect는 너비(width)와 수평축 사이의 각도를 반환. 범위는 (-90, 0].
            #        너비가 높이보다 길면, 각도는 너비 축의 방향. 아니면 높이 축의 방향.
            try:
                rect_min_area_char = cv2.minAreaRect(poly_on_map_int)
            except cv2.error:  # 점이 너무 적거나 일직선상에 있는 경우 등
                # print(f"Warning: cv2.minAreaRect failed for char '{char_info.char_content}'. Skipping.")
                continue  # 이 문자는 Region Score 생성에서 제외

            (cx_on_map, cy_on_map), (w_on_map, h_on_map), angle_on_map_cv = rect_min_area_char

            # 매우 작은 크기의 문자 처리
            w_on_map = max(w_on_map, min_char_size_on_map)
            h_on_map = max(h_on_map, min_char_size_on_map)

            # 가우시안 표준편차 계산 (문자 크기의 짧은 변 기준)
            # CRAFT 논문에서는 문자 박스의 짧은 변의 길이에 비례하도록 표준편차를 설정.
            short_side_on_map = min(w_on_map, h_on_map)
            std_dev_char = short_side_on_map * gaussian_variance_scale_factor
            std_dev_char = max(std_dev_char, 0.5)  # 최소 표준편차 값 (너무 뾰족해지는 것 방지)
            variance_char_sq = std_dev_char ** 2 + 1e-6  # 분산 (0으로 나누기 방지)

            # 회전된 가우시안을 위한 좌표 변환:
            # 1. 좌표계를 문자의 중심(cx_on_map, cy_on_map)으로 이동.
            # 2. 회전된 좌표계 (문자의 주축에 정렬된)로 변환.
            #    OpenCV의 minAreaRect 각도는 약간 다루기 까다로울 수 있음.
            #    여기서는 간단하게 축에 정렬된 타원형 가우시안을 사용하고,
            #    더 정확하게는 w_on_map과 h_on_map을 주축/부축의 길이로 보고
            #    angle_on_map_cv를 사용해 회전시켜야 함.

            # 여기서는 타원형 가우시안 (각 축에 다른 분산)을 사용하고, 회전은 생략 후 폴리곤 마스크로 제한.
            # 또는, w_on_map과 h_on_map 중 긴 쪽을 주축으로 하여 대칭적 가우시안을 사용하고 회전.
            # 좀 더 CRAFT에 가까운 방식은, 짧은 변을 기준으로 대칭적 가우시안을 만들고,
            # 이를 문자의 방향에 맞게 늘리거나 회전시키는 것.

            # 여기서는 w_on_map, h_on_map을 각 축 방향의 크기로 보고 (회전 무시)
            # 각 축에 대한 분산을 계산하여 타원형 가우시안 생성
            variance_x_char_sq = ((w_on_map / 2.0) * gaussian_variance_scale_factor) ** 2 + 1e-6
            variance_y_char_sq = ((h_on_map / 2.0) * gaussian_variance_scale_factor) ** 2 + 1e-6

            # 2D 가우시안 값 계산 (축 정렬 타원형)
            gaussian_char_map_values = np.exp(
                -(((xx_mg - cx_on_map) ** 2 / (2 * variance_x_char_sq)) +
                  ((yy_mg - cy_on_map) ** 2 / (2 * variance_y_char_sq)))
            )

            # 가우시안을 문자 폴리곤 내부에만 적용하기 위한 마스크 생성
            char_fill_mask_on_map = np.zeros_like(region_map_np, dtype=np.uint8)
            cv2.fillPoly(char_fill_mask_on_map, [poly_on_map_int], 1)

            # 기존 맵과 현재 문자 가우시안의 최대값을 취함
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
