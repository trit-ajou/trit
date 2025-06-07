import os
import random
import numpy as np
import torch
import torchvision.transforms.functional as VTF
import threading
import time
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from typing import List, Tuple, Optional, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
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
        # 멀티스레딩 관련
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.future = None
        self.total_tasks = 0
        self.completed_tasks = 0
        self.lock = threading.Lock()
    def start_loading_async(
        self, num_images: int, dir: str, max_text_size: tuple[int, int]
    ):
        if self.future and not self.future.done():
            raise RuntimeError("[ImageLoader] 이미 진행 중인 로딩 작업이 있습니다.")
        print("[ImageLoader] 이미지 로딩 작업을 시작합니다.")
        self.future = self.executor.submit(
            self.load_images, num_images, dir, max_text_size
        )

    def get_loaded_images(self) -> List[TextedImage]:
        if not self.future:
            raise ValueError("[ImageLoader] 오류: 시작된 로딩 작업이 없습니다.")
        # 작업이 이미 완료되었다면 바로 결과 반환
        if self.future.done():
            try:
                return self.future.result()
            except Exception as e:
                raise ValueError(f"[ImageLoader] 이미지 로딩 중 오류 발생: {e}")
        # 작업이 진행 중이라면 tqdm 사용하여 대기
        pbar = None
        try:
            while not self.future.done():
                # pbar 생성
                if pbar is None:
                    with self.lock:
                        if self.total_tasks > 0:
                            pbar = tqdm(
                                total=self.total_tasks,
                                desc="이미지 렌더링 중",
                                leave=False,
                            )
                            pbar.update(self.completed_tasks)
                            last_completed = self.completed_tasks
                # pbar 진행률 업데이트
                if pbar:
                    current_completed = self.completed_tasks
                    update_val = current_completed - last_completed
                    if update_val > 0:
                        pbar.update(update_val)
                        last_completed = current_completed
                time.sleep(1)
        except Exception as e:
            raise ValueError(f"[ImageLoader] 이미지 로딩 중 오류 발생: {e}")
        # 작업 끝
        if pbar:
            pbar.close()
        return self.future.result()

    def shutdown(self):
        """진행 중인 작업을 멈추고 executer 안전하게 종료."""
        self.executor.shutdown()

    def load_images(
            self, num_images: int, dir_path: str,max_text_size: tuple[int, int], generate_craft_gt: bool = True
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

            h, w = self.setting.model1_input_size
            noise_imgs = np.random.randint(
                0, 256, (num_images, h, w, 3), dtype=np.uint8
            )
            for noise_img in noise_imgs:
                clear_pils.append(Image.fromarray(noise_img, "RGB"))
        # 작업 시작 전 총 작업량 초기화
        with self.lock:
            self.total_tasks = len(clear_pils)
            self.completed_tasks = 0
        # 병렬로 TextedImage 생성
        print(
            f"[ImageLoader] Redering TextedImages with {self.setting.num_workers} workers"
        )
        texted_images = []
        with ThreadPoolExecutor(self.setting.num_workers) as pool:
            futures = [
                pool.submit(self.pil_to_texted_image, clear_pil, max_text_size,generate_craft_gt)
                for clear_pil in clear_pils
            ]
            for future in as_completed(futures):
                texted_images.append(future.result())
                with self.lock:
                    self.completed_tasks += 1
        return texted_images
    def pil_to_texted_image(
            self,
            pil_img: Image.Image,
            max_text_size: tuple[int, int],
            generate_craft_gt: bool = True,
            debug_block: bool = False
    ) -> 'TextedImage':
        """
        단일 PIL 이미지에 텍스트를 합성하고, 필요시 CRAFT용 GT 데이터를 생성하여
        하나의 TextedImage 객체로 반환합니다.

        Args:
            pil_img (Image.Image): 텍스트를 합성할 원본 PIL 이미지.
            generate_craft_gt (bool): CRAFT 학습을 위한 GT 데이터(스코어맵 등) 생성 여부.
            debug_block (bool): True일 경우, 각 텍스트 블록의 렌더링 과정을 디버그 이미지로 저장.

        Returns:
            TextedImage: 모든 정보가 포함된 TextedImage 객체.
        """
        # --- 1. 초기 텐서 및 변수 준비 ---
        img_w, img_h = pil_img.size

        # 원본, 텍스트 합성본, 마스크 텐서를 CPU에 생성
        orig_tensor = VTF.to_tensor(pil_img.convert("RGB"))
        timg_tensor = orig_tensor.clone()
        text_block_pixel_mask = torch.zeros((1, img_h, img_w))

        # 텍스트 블록 단위의 바운딩 박스 리스트
        word_level_bboxes_list: List[BBox] = []

        # CRAFT GT 데이터 생성을 위한 변수
        all_char_infos_for_craft: Optional[List[CharInfo]] = [] if generate_craft_gt else None
        region_score_map_tensor: Optional[torch.Tensor] = None
        affinity_score_map_tensor: Optional[torch.Tensor] = None

        # 텍스트 블록(단어) ID를 고유하게 관리하기 위한 변수
        current_text_block_id_base = 0

        # --- 2. 텍스트 블록 생성 및 합성 루프 ---
        num_render_texts = random.randint(*self.policy.num_texts)
        for i_block in range(num_render_texts):

            # 2-1. 텍스트 스타일 및 내용 랜덤 생성
            is_sfx = random.random() < self.policy.sfx_style_prob
            font_size_r = random.uniform(*self.policy.font_size_ratio_to_image_height_range)
            if is_sfx:
                font_size_r = random.uniform(
                    self.policy.font_size_ratio_to_image_height_range[1],
                    min(0.3, self.policy.font_size_ratio_to_image_height_range[1] * 2.5)
                )
            font_size_val = max(12, int(img_h * font_size_r))
            font_obj = self._get_random_font(font_size_val)
            if not font_obj: continue

            text_str_content = self._get_random_text_content()
            if not text_str_content: continue

            wrapped_text_str_multiline = text_str_content
            if random.random() < self.policy.multiline_prob:
                max_textbox_w_px = int(img_w * random.uniform(*self.policy.textbox_width_ratio_to_image_width_range))
                wrapped_text_str_multiline = self._wrap_text_pil(text_str_content, font_obj, max(1, max_textbox_w_px))

            # 2-2. 텍스트 블록 렌더링 및 문자 정보 추출
            debug_prefix = f"block_{i_block}_{random.randint(1000, 9999)}" if debug_block else None
            rendered_text_block_pil, char_infos_relative_to_block_pil = self._render_text_block_and_get_char_infos(
                wrapped_text_str_multiline,
                font_obj,
                is_sfx,
                debug_visualization_prefix=debug_prefix
            )
            if rendered_text_block_pil is None: continue

            # 2-3. 렌더링된 블록을 이미지에 합성
            text_pil_w, text_pil_h = rendered_text_block_pil.size
            if img_w < text_pil_w or img_h < text_pil_h: raise ValueError("[ImageLoader] 텍스트가 이미지 크기를 초과했습니다. policy를 조절하십시오 인간.")
            # 텍스트가 max_text_size보다 크면 (이 부분도 클리핑으로 처리되므로 필요 없음)
            if text_pil_w > max_text_size[0] or text_pil_h > max_text_size[1]:
                raise ValueError("[ImageLoader] 텍스트가 최대 크기를 초과했습니다. policy를 조절하십시오 인간.")
            abs_x_on_img = random.randint(0, img_w - text_pil_w)
            abs_y_on_img = random.randint(0, img_h - text_pil_h)
            current_text_block_bbox = BBox(abs_x_on_img, abs_y_on_img, abs_x_on_img + text_pil_w,
                                           abs_y_on_img + text_pil_h)
            word_level_bboxes_list.append(current_text_block_bbox)

            rgba_pil_tensor = VTF.to_tensor(rendered_text_block_pil)
            rgb_pil_tensor = rgba_pil_tensor[:3, :, :]
            alpha_pil_tensor = rgba_pil_tensor[3:4, :, :]

            timg_tensor = TextedImage._alpha_blend(timg_tensor, current_text_block_bbox, rgb_pil_tensor,
                                                   alpha_pil_tensor)

            current_text_mask_tensor = (alpha_pil_tensor > 0).float()
            text_block_pixel_mask[:, current_text_block_bbox.y1:current_text_block_bbox.y2,
            current_text_block_bbox.x1:current_text_block_bbox.x2] = \
                torch.maximum(text_block_pixel_mask[:, current_text_block_bbox.y1:current_text_block_bbox.y2,
                              current_text_block_bbox.x1:current_text_block_bbox.x2], current_text_mask_tensor)

            # 2-4. 문자 정보(CharInfo) 처리
            if generate_craft_gt and char_infos_relative_to_block_pil:
                for char_info_rel in char_infos_relative_to_block_pil:
                    abs_polygon = char_info_rel.polygon + np.array([abs_x_on_img, abs_y_on_img])
                    final_word_id = current_text_block_id_base + char_info_rel.word_id
                    all_char_infos_for_craft.append(
                        CharInfo(abs_polygon, char_info_rel.char_content, final_word_id)
                    )

                # 다음 텍스트 블록의 word_id가 겹치지 않도록 base 업데이트
                max_rel_word_id = max(ci.word_id for ci in char_infos_relative_to_block_pil)
                current_text_block_id_base += (max_rel_word_id + 1)

        # --- 3. CRAFT Ground Truth Score Map 생성 ---
        if generate_craft_gt:
            target_h_map, target_w_map = img_h // 2, img_w // 2
            region_score_map_tensor, affinity_score_map_tensor = self._create_craft_gt_maps(
                all_char_infos_for_craft,
                (target_h_map, target_w_map)
            )

        # --- 4. 최종 TextedImage 객체 생성 및 반환 ---
        return TextedImage(
            orig=orig_tensor,
            timg=timg_tensor,
            mask=text_block_pixel_mask,
            bboxes=word_level_bboxes_list,
            all_char_infos=all_char_infos_for_craft,
            region_score_map=region_score_map_tensor,
            affinity_score_map=affinity_score_map_tensor,
        )
    def _render_text_block_and_get_char_infos(
            self,
            text_content_str: str,
            font_obj: ImageFont.FreeTypeFont,
            is_sfx_style_policy: bool,
            # extract_char_info_flag는 CRAFT GT 생성을 위해 항상 True라고 가정하고,
            # 코드 내에서 명시적으로 사용하기보다 로직에 통합합니다.
            debug_visualization_prefix: Optional[str] = None
    ) -> Tuple[Optional[Image.Image], Optional[List[CharInfo]]]:
        """
        하나의 텍스트 블록을 렌더링하고, 변형을 적용하며, 각 문자의 폴리곤 정보를 추출합니다.

        이 함수는 다음 단계를 거칩니다:
        1. 텍스트 스타일(색상, 외곽선, 그림자 등)을 결정하고, 충분한 패딩을 가진 초기 레이어에 텍스트를 렌더링합니다.
        2. 렌더링된 초기 레이어 상에서 각 문자의 바운딩 박스(폴리곤)를 계산합니다.
        3. 기하학적 변형(회전, 기울이기, 원근)을 이미지와 좌표 변환 행렬에 동기화하여 적용합니다.
        4. 최종적으로 변형되고 크롭된 텍스트 이미지와, 그에 맞춰 변환된 문자 폴리곤 리스트를 반환합니다.

        Args:
            text_content_str (str): 렌더링할 텍스트 문자열.
            font_obj (ImageFont.FreeTypeFont): 사용할 폰트 객체.
            is_sfx_style_policy (bool): 특수 효과 스타일 적용 여부.
            debug_visualization_prefix (Optional[str]): 디버그 시각화 이미지 저장 시 사용할 파일명 접두사.

        Returns:
            Tuple[Optional[Image.Image], Optional[List[CharInfo]]]:
                - 변형 및 크롭된 최종 텍스트 PIL 이미지.
                - 크롭된 이미지 기준의 상대 좌표를 가진 CharInfo 객체 리스트.
                - 처리 중 오류 발생 시 (None, None)을 반환할 수 있습니다.
        """
        # --- 1. 텍스트 스타일 설정 및 초기 렌더링 ---

        # 1-1. 렌더링 스타일 파라미터 결정
        if self.policy.text_color_is_random:
            text_color = self._get_random_rgba()
        else:
            text_color = random.choice(self.policy.fixed_text_color_options)
        stroke_w_val, stroke_fill = 0, None
        if random.random() < self.policy.stroke_prob or is_sfx_style_policy:
            s_r = random.uniform(*self.policy.stroke_width_ratio_to_font_size_range)
            stroke_w_val = max(self.policy.stroke_width_limit_px[0],
                               min(int(font_obj.size * s_r), self.policy.stroke_width_limit_px[1]))
            if is_sfx_style_policy:
                stroke_w_val = min(max(stroke_w_val, int(font_obj.size * 0.15)),
                                   self.policy.stroke_width_limit_px[1] * 2)
                if self.policy.stroke_color_is_random:
                    stroke_fill = self._get_random_rgba()
                else:
                    stroke_fill = random.choice(self.policy.fixed_stroke_color_options)

        shadow_params_val = None
        if random.random() < self.policy.shadow_prob or is_sfx_style_policy:
            s_off_x_r = random.uniform(*self.policy.shadow_offset_x_ratio_to_font_size_range)
            s_off_y_r = random.uniform(*self.policy.shadow_offset_y_ratio_to_font_size_range)
            s_blur = random.randint(*self.policy.shadow_blur_radius_range)
            s_color = self._get_random_rgba()
            if is_sfx_style_policy:
                s_off_x_r *= 1.5;
                s_off_y_r *= 1.5;
                s_blur = max(s_blur, 3)
                s_color = (s_color[0], s_color[1], s_color[2], min(255, int(s_color[3] * 1.5)))
            shadow_params_val = (int(font_obj.size * s_off_x_r), int(font_obj.size * s_off_y_r), s_blur, s_color)

        final_text_str = text_content_str
        text_align_opt = random.choice(self.policy.text_align_options)
        line_spacing_pixels = int(
            font_obj.size * (random.uniform(*self.policy.line_spacing_ratio_to_font_size_range) - 1.0))

        # 1-2. 렌더링될 텍스트의 크기 계산 및 패딩이 적용된 레이어 생성
        _draw_dummy = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
        try:
            text_bbox_on_pil = _draw_dummy.textbbox((0, 0), final_text_str, font=font_obj, stroke_width=stroke_w_val,
                                                    spacing=line_spacing_pixels, align=text_align_opt)
        except TypeError:  # 구버전 Pillow 호환
            text_bbox_on_pil = _draw_dummy.textbbox((0, 0), final_text_str, font=font_obj)
            if stroke_w_val > 0:
                text_bbox_on_pil = (text_bbox_on_pil[0] - stroke_w_val, text_bbox_on_pil[1] - stroke_w_val,
                                    text_bbox_on_pil[2] + stroke_w_val, text_bbox_on_pil[3] + stroke_w_val)

        text_render_w, text_render_h = text_bbox_on_pil[2] - text_bbox_on_pil[0], text_bbox_on_pil[3] - \
                                       text_bbox_on_pil[1]
        if text_render_w <= 0 or text_render_h <= 0: return None, None

        padding_factor = 1.0  # 변형 시 잘림 방지를 위해 넉넉한 패딩
        pad_x = int(text_render_w * padding_factor) + stroke_w_val + (
            abs(shadow_params_val[0]) + shadow_params_val[2] if shadow_params_val else 0)
        pad_y = int(text_render_h * padding_factor) + stroke_w_val + (
            abs(shadow_params_val[1]) + shadow_params_val[2] if shadow_params_val else 0)

        layer_w, layer_h = int(max(1, text_render_w + pad_x * 2)), int(max(1, text_render_h + pad_y * 2))
        draw_x_origin, draw_y_origin = pad_x - text_bbox_on_pil[0], pad_y - text_bbox_on_pil[1]

        # 1-3. 텍스트를 초기 레이어에 렌더링
        text_layer_img_untransformed = Image.new("RGBA", (layer_w, layer_h), (0, 0, 0, 0))
        draw_on_layer = ImageDraw.Draw(text_layer_img_untransformed)
        if shadow_params_val:
            s_off_x, s_off_y, s_blur_r, s_color = shadow_params_val
            shadow_layer = Image.new("RGBA", text_layer_img_untransformed.size, (0, 0, 0, 0))
            shadow_draw = ImageDraw.Draw(shadow_layer)
            try:
                shadow_draw.text((draw_x_origin + s_off_x, draw_y_origin + s_off_y), final_text_str, font=font_obj,
                                 fill=s_color, stroke_width=stroke_w_val, stroke_fill=stroke_fill,
                                 spacing=line_spacing_pixels, align=text_align_opt)
            except TypeError:
                shadow_draw.text((draw_x_origin + s_off_x, draw_y_origin + s_off_y), final_text_str, font=font_obj,
                                 fill=s_color)
            if s_blur_r > 0: shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(s_blur_r))
            text_layer_img_untransformed.alpha_composite(shadow_layer)

        try:
            draw_on_layer.text((draw_x_origin, draw_y_origin), final_text_str, font=font_obj, fill=text_color,
                               stroke_width=stroke_w_val, stroke_fill=stroke_fill, spacing=line_spacing_pixels,
                               align=text_align_opt)
        except TypeError:
            draw_on_layer.text((draw_x_origin, draw_y_origin), final_text_str, font=font_obj, fill=text_color)

        # --- 2. 변형 전 문자 폴리곤 계산 ---
        untransformed_char_polygons_with_info: List[Tuple[np.ndarray, str, int]] = []

        current_relative_word_id_in_block = 0
        lines = final_text_str.split('\n')
        current_line_y_start_in_padded = draw_y_origin

        for line_text_str in lines:
            if not line_text_str.strip():
                try:
                    empty_line_h = font_obj.getmask("A").size[1] if hasattr(font_obj.getmask("A"), 'size') else \
                    font_obj.getsize("A")[1]
                except AttributeError:
                    empty_line_h = font_obj.size
                current_line_y_start_in_padded += empty_line_h + line_spacing_pixels
                current_relative_word_id_in_block += 1
                continue

            try:
                line_bbox_for_align = _draw_dummy.textbbox((0, 0), line_text_str, font=font_obj,
                                                           stroke_width=stroke_w_val, spacing=0)
                line_actual_render_width = line_bbox_for_align[2] - line_bbox_for_align[0]
                line_actual_render_height = line_bbox_for_align[3] - line_bbox_for_align[1]
            except TypeError:
                line_actual_render_width = font_obj.getsize(line_text_str)[0] if hasattr(font_obj, 'getsize') else 0
                line_actual_render_height = font_obj.getsize("A")[1] if hasattr(font_obj, 'getsize') else font_obj.size

            current_line_x_start_in_padded = draw_x_origin
            if text_align_opt == 'center':
                current_line_x_start_in_padded += (text_render_w - line_actual_render_width) / 2.0
            elif text_align_opt == 'right':
                current_line_x_start_in_padded += (text_render_w - line_actual_render_width)

            current_char_x_offset_in_line = 0
            for char_val in line_text_str:
                try:
                    char_font_bbox = font_obj.getbbox(char_val, stroke_width=stroke_w_val)
                    char_advance = font_obj.getlength(char_val)
                except AttributeError:
                    mask = font_obj.getmask(char_val)
                    char_font_bbox = mask.getbbox() if mask else (0, 0, 0, 0)
                    char_advance = font_obj.getsize(char_val)[0] if hasattr(font_obj, 'getsize') else (
                                char_font_bbox[2] - char_font_bbox[0])

                char_x1 = current_line_x_start_in_padded + current_char_x_offset_in_line + char_font_bbox[0]
                char_y1 = current_line_y_start_in_padded + char_font_bbox[1]
                char_x2 = current_line_x_start_in_padded + current_char_x_offset_in_line + char_font_bbox[2]
                char_y2 = current_line_y_start_in_padded + char_font_bbox[3]

                untransformed_poly = np.array(
                    [[char_x1, char_y1], [char_x2, char_y1], [char_x2, char_y2], [char_x1, char_y2]], dtype=np.float32)
                untransformed_char_polygons_with_info.append(
                    (untransformed_poly, char_val, current_relative_word_id_in_block))

                current_char_x_offset_in_line += char_advance
                if char_val == ' ' and char_advance == 0:
                    try:
                        char_advance_space = font_obj.getmask(" ").size[0] * 0.8
                    except:
                        char_advance_space = font_obj.size * 0.25
                    current_char_x_offset_in_line += char_advance_space

            current_line_y_start_in_padded += line_actual_render_height + line_spacing_pixels
            current_relative_word_id_in_block += 1

        # --- 디버그 시각화 (변형 전) ---
        if debug_visualization_prefix and untransformed_char_polygons_with_info:
            img_vis_untransformed = text_layer_img_untransformed.copy()
            draw_vis_untransformed = ImageDraw.Draw(img_vis_untransformed)
            for poly, _, _ in untransformed_char_polygons_with_info:
                draw_vis_untransformed.polygon([tuple(p) for p in poly.tolist()], outline=(0, 255, 0, 128), width=1)
                center_x, center_y = np.mean(poly[:, 0]), np.mean(poly[:, 1])
                r = 2
                draw_vis_untransformed.ellipse((center_x - r, center_y - r, center_x + r, center_y + r),
                                               fill=(255, 0, 0, 200))

            vis_save_dir = self.setting.debug_dir
            os.makedirs(vis_save_dir, exist_ok=True)
            img_vis_untransformed.save(
                os.path.join(vis_save_dir, f"{debug_visualization_prefix}_untransformed_chars.png"))

        # --- 3. 기하학적 변형 및 좌표 변환 동기화 ---
        img_to_transform_pil = text_layer_img_untransformed
        T_coords_total = np.eye(3, dtype=np.float64)

        # 3-1. 회전(Rotation)
        current_rot_range = self.policy.rotation_angle_range
        if is_sfx_style_policy: current_rot_range = (min(-90, current_rot_range[0] * 2),
                                                     max(90, current_rot_range[1] * 2))
        angle_deg_to_apply = random.uniform(*current_rot_range)

        W0, H0 = img_to_transform_pil.size
        img_after_rotation_pil = img_to_transform_pil.rotate(angle_deg_to_apply, expand=True,
                                                             resample=Image.Resampling.BICUBIC)
        W1, H1 = img_after_rotation_pil.size

        center_x0, center_y0 = W0 / 2.0, H0 / 2.0
        center_x1, center_y1 = W1 / 2.0, H1 / 2.0
        T_to_origin = np.array([[1, 0, -center_x0], [0, 1, -center_y0], [0, 0, 1]], dtype=np.float64)
        angle_rad_for_coords = np.radians(-angle_deg_to_apply)
        cos_a, sin_a = np.cos(angle_rad_for_coords), np.sin(angle_rad_for_coords)
        R_matrix = np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]], dtype=np.float64)
        T_to_new_center = np.array([[1, 0, center_x1], [0, 1, center_y1], [0, 0, 1]], dtype=np.float64)
        M_rotation = T_to_new_center @ R_matrix @ T_to_origin

        T_coords_total = M_rotation @ T_coords_total
        img_to_transform_pil = img_after_rotation_pil

        # 3-2. 기울이기(Shear)
        if random.random() < self.policy.shear_apply_prob or is_sfx_style_policy:
            shear_x_factor = random.uniform(*self.policy.shear_x_factor_range)
            shear_y_factor = random.uniform(*self.policy.shear_y_factor_range)
            if is_sfx_style_policy: shear_x_factor *= 1.5; shear_y_factor *= 1.5

            pil_affine_coeffs, M_shear = None, np.eye(3, dtype=np.float64)
            if abs(shear_x_factor) > 0.01:
                pil_affine_coeffs = (1, shear_x_factor, 0, 0, 1, 0)
                M_shear = np.array([[1, -shear_x_factor, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
            elif abs(shear_y_factor) > 0.01:
                pil_affine_coeffs = (1, 0, 0, shear_y_factor, 1, 0)
                M_shear = np.array([[1, 0, 0], [-shear_y_factor, 1, 0], [0, 0, 1]], dtype=np.float64)

            if pil_affine_coeffs:
                img_to_transform_pil = img_to_transform_pil.transform(img_to_transform_pil.size, Image.Transform.AFFINE,
                                                                      pil_affine_coeffs,
                                                                      resample=Image.Resampling.BICUBIC)
                T_coords_total = M_shear @ T_coords_total

        # 3-3. 원근(Perspective)
        if random.random() < self.policy.perspective_transform_enabled_prob or is_sfx_style_policy:
            H_forward_coeffs = self._get_perspective_coeffs_numpy(img_to_transform_pil.size[0],
                                                                  img_to_transform_pil.size[1],
                                                                  self.policy.perspective_transform_strength_ratio_range)
            if H_forward_coeffs:
                M_perspective_forward = np.array([[H_forward_coeffs[0], H_forward_coeffs[1], H_forward_coeffs[2]],
                                                  [H_forward_coeffs[3], H_forward_coeffs[4], H_forward_coeffs[5]],
                                                  [H_forward_coeffs[6], H_forward_coeffs[7], 1.0]], dtype=np.float64)
                try:
                    M_perspective_inverse = np.linalg.inv(M_perspective_forward)
                    pil_perspective_coeffs = tuple(M_perspective_inverse.flatten()[:8])
                    img_to_transform_pil = img_to_transform_pil.transform(img_to_transform_pil.size,
                                                                          Image.Transform.PERSPECTIVE,
                                                                          pil_perspective_coeffs,
                                                                          resample=Image.Resampling.BICUBIC)
                    T_coords_total = M_perspective_forward @ T_coords_total
                except (np.linalg.LinAlgError, ValueError):
                    pass

        final_transformed_img_pil = img_to_transform_pil

        # --- 디버그 시각화 (변형 후) ---
        if debug_visualization_prefix and untransformed_char_polygons_with_info:
            img_vis_transformed = final_transformed_img_pil.copy()
            draw_vis_transformed = ImageDraw.Draw(img_vis_transformed)
            for poly_untransformed, _, _ in untransformed_char_polygons_with_info:
                poly_h = np.hstack([poly_untransformed, np.ones((poly_untransformed.shape[0], 1))])
                transformed_poly_h = (T_coords_total @ poly_h.T).T
                w_coords = transformed_poly_h[:, 2]
                if np.any(np.abs(w_coords) < 1e-7): continue
                transformed_poly = transformed_poly_h[:, :2] / w_coords[:, np.newaxis]

                draw_vis_transformed.polygon([tuple(p) for p in transformed_poly.tolist()], outline=(0, 0, 255, 128),
                                             width=1)
                center_x, center_y = np.mean(transformed_poly[:, 0]), np.mean(transformed_poly[:, 1])
                r = 2
                draw_vis_transformed.ellipse((center_x - r, center_y - r, center_x + r, center_y + r),
                                             fill=(255, 255, 0, 200))

            vis_save_dir = self.setting.debug_dir
            img_vis_transformed.save(os.path.join(vis_save_dir, f"{debug_visualization_prefix}_transformed_chars.png"))

        # --- 4. 최종 이미지 크롭 및 변환된 폴리곤 계산 ---
        final_bbox = final_transformed_img_pil.getbbox()
        if final_bbox is None: return None, None

        cropped_final_img_pil = final_transformed_img_pil.crop(final_bbox)
        if cropped_final_img_pil.width == 0 or cropped_final_img_pil.height == 0: return None, None

        char_infos_list_relative_to_cropped: Optional[List[CharInfo]] = None
        if untransformed_char_polygons_with_info:
            char_infos_list_relative_to_cropped = []
            crop_offset_x, crop_offset_y = final_bbox[0], final_bbox[1]

            for poly_untransformed, char_s, rel_word_id in untransformed_char_polygons_with_info:
                poly_h = np.hstack([poly_untransformed, np.ones((poly_untransformed.shape[0], 1))])
                transformed_poly_h = (T_coords_total @ poly_h.T).T

                w_coords = transformed_poly_h[:, 2]
                if np.any(np.abs(w_coords) < 1e-7): continue

                transformed_poly_cartesian = transformed_poly_h[:, :2] / w_coords[:, np.newaxis]
                poly_relative_to_crop = transformed_poly_cartesian - np.array([crop_offset_x, crop_offset_y])

                char_infos_list_relative_to_cropped.append(CharInfo(poly_relative_to_crop, char_s, rel_word_id))

        return cropped_final_img_pil, char_infos_list_relative_to_cropped

    def _create_craft_gt_maps(
              self,
              all_char_infos_abs_coords: List[CharInfo],
              target_map_output_size: Tuple[int, int],
              gaussian_variance_scale_factor: float = 0.25,
              affinity_gaussian_scale_factor_major: float = 0.3,
              affinity_gaussian_scale_factor_minor: float = 0.15,
              min_char_size_on_map: int = 2
      ) -> Tuple[torch.Tensor, torch.Tensor]:
          """
          주어진 문자 정보 리스트로부터 Region 및 Affinity Score Map을 생성합니다.
          이 함수는 원본 CRAFT 구현의 GT 생성 방식을 따릅니다.
          텍스트가 없는 이미지에 대해서는 0으로 채워진 맵을 반환합니다.

          Args:
              all_char_infos_abs_coords (List[CharInfo]): 이미지 전체 좌표 기준의 문자 정보 리스트.
              target_map_output_size (Tuple[int, int]): 생성할 GT 맵의 (높이, 너비).
              ... (다른 파라미터들)

          Returns:
              Tuple[torch.Tensor, torch.Tensor]: (Region Score Map, Affinity Score Map) 텐서 튜플.
          """
          # --- 1. 초기화 ---
          target_h, target_w = target_map_output_size
          region_map_np = np.zeros((target_h, target_w), dtype=np.float32)
          affinity_map_np = np.zeros((target_h, target_w), dtype=np.float32)

          # 텍스트가 없는 경우(all_char_infos가 비어있음), 0으로 채워진 맵을 즉시 반환
          if not all_char_infos_abs_coords:
              region_map_tensor = torch.from_numpy(region_map_np).unsqueeze(0)
              affinity_map_tensor = torch.from_numpy(affinity_map_np).unsqueeze(0)
              return region_map_tensor, affinity_map_tensor

          # 전체 맵에 대한 좌표 그리드 생성 (한 번만)
          xx_mesh, yy_mesh = np.meshgrid(np.arange(target_w), np.arange(target_h))

          # --- 2. Region Score Map 생성 ---
          for char_info in all_char_infos_abs_coords:
              poly_on_map = char_info.polygon / 2.0
              poly_on_map_int = poly_on_map.astype(np.int32)
              if poly_on_map_int.shape[0] < 3: continue

              try:
                  rect = cv2.minAreaRect(poly_on_map_int.reshape(-1, 1, 2))
              except cv2.error:
                  continue

              (center_x, center_y), (width, height), angle_cv = rect
              width = max(width, min_char_size_on_map)
              height = max(height, min_char_size_on_map)

              std_dev_x = (width / 2.0) * gaussian_variance_scale_factor
              std_dev_y = (height / 2.0) * gaussian_variance_scale_factor
              variance_x_sq = max(std_dev_x, 0.5) ** 2 + 1e-6
              variance_y_sq = max(std_dev_y, 0.5) ** 2 + 1e-6

              angle_rad = np.radians(-angle_cv)
              cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

              shifted_x, shifted_y = xx_mesh - center_x, yy_mesh - center_y
              rotated_x = shifted_x * cos_a - shifted_y * sin_a
              rotated_y = shifted_x * sin_a + shifted_y * cos_a

              gaussian_values = np.exp(-((rotated_x ** 2 / (2 * variance_x_sq)) + (rotated_y ** 2 / (2 * variance_y_sq))))

              mask = np.zeros_like(region_map_np, dtype=np.uint8)
              cv2.fillPoly(mask, [poly_on_map_int], 1)
              region_map_np = np.maximum(region_map_np, gaussian_values * mask)

          # --- 3. Affinity Score Map 생성 ---
          # 가. 문자를 단어(word_id)별로 그룹화
          words_data = {}
          for char_info in all_char_infos_abs_coords:
              words_data.setdefault(char_info.word_id, []).append(char_info)

          # 나. 각 단어 내 인접 문자 쌍에 대해 어피니티 계산
          for _, chars_in_word in words_data.items():
              if len(chars_in_word) < 2: continue

              # 문자를 x좌표 기준으로 정렬하여 처리 순서 결정
              sorted_chars = sorted(chars_in_word, key=lambda ci: (ci.polygon[0, 0], ci.polygon[0, 1]))

              for i in range(len(sorted_chars) - 1):
                  char1, char2 = sorted_chars[i], sorted_chars[i + 1]
                  poly1, poly2 = char1.polygon / 2.0, char2.polygon / 2.0

                  # 어피니티 가우시안 중심 및 방향 계산
                  center1, center2 = np.mean(poly1, axis=0), np.mean(poly2, axis=0)
                  aff_center_x, aff_center_y = (center1[0] + center2[0]) / 2.0, (center1[1] + center2[1]) / 2.0
                  angle_rad = np.arctan2(center2[1] - center1[1], center2[0] - center1[0])

                  # 어피니티 가우시안 크기(표준편차) 계산
                  dist = np.linalg.norm(center1 - center2)
                  if dist < 1e-3: continue

                  h1 = np.max(poly1[:, 1]) - np.min(poly1[:, 1]) if poly1.shape[0] > 0 else 0
                  h2 = np.max(poly2[:, 1]) - np.min(poly2[:, 1]) if poly2.shape[0] > 0 else 0
                  avg_h = max((h1 + h2) / 2.0, 1.0)  # 0으로 나누기 방지

                  std_major = dist * affinity_gaussian_scale_factor_major
                  std_minor = avg_h * affinity_gaussian_scale_factor_minor
                  var_major_sq = max(std_major, 0.5) ** 2 + 1e-6
                  var_minor_sq = max(std_minor, 0.5) ** 2 + 1e-6

                  # 회전된 가우시안 값 계산
                  cos_a, sin_a = np.cos(-angle_rad), np.sin(-angle_rad)
                  shifted_x, shifted_y = xx_mesh - aff_center_x, yy_mesh - aff_center_y
                  rotated_x = shifted_x * cos_a - shifted_y * sin_a
                  rotated_y = shifted_x * sin_a + shifted_y * cos_a

                  gaussian_values = np.exp(
                      -((rotated_x ** 2 / (2 * var_major_sq)) + (rotated_y ** 2 / (2 * var_minor_sq))))

                  # 두 문자를 잇는 볼록 다각형(convex hull)에만 가우시안 적용
                  mask = np.zeros_like(affinity_map_np, dtype=np.uint8)
                  combined_poly = np.concatenate((poly1.astype(np.int32), poly2.astype(np.int32)), axis=0)
                  if combined_poly.shape[0] >= 3:
                      try:
                          hull = cv2.convexHull(combined_poly)
                          cv2.fillPoly(mask, [hull], 1)
                      except cv2.error:  # 점이 너무 적거나 일직선상에 있어 hull을 만들 수 없는 경우
                          cv2.line(mask, tuple(center1.astype(np.int32)), tuple(center2.astype(np.int32)), 1,
                                   thickness=int(max(1, avg_h * 0.2)))
                  elif combined_poly.shape[0] > 1:  # 점이 2개일 때
                      cv2.line(mask, tuple(center1.astype(np.int32)), tuple(center2.astype(np.int32)), 1,
                               thickness=int(max(1, avg_h * 0.2)))

                  affinity_map_np = np.maximum(affinity_map_np, gaussian_values * mask)

          # --- 4. 최종 결과 반환 ---
          region_map_tensor = torch.from_numpy(np.clip(region_map_np, 0, 1)).unsqueeze(0)
          affinity_map_tensor = torch.from_numpy(np.clip(affinity_map_np, 0, 1)).unsqueeze(0)

          return region_map_tensor, affinity_map_tensor
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
        if self.policy.stroke_color_is_random:
            stroke_fill = self._get_random_rgba()
        else:
            stroke_fill = random.choice(self.policy.fixed_stroke_color_options)
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
