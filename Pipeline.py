from peft import LoraConfig
import torch
from tqdm import tqdm
from copy import deepcopy  # Changed from copy to deepcopy for TextedImage objects
from torch.utils.data import DataLoader, random_split

import os  # Added for path joining and directory creation
import json  # Added for JSON file reading
import numpy as np  # Added for array operations
import torchvision.transforms.functional as VTF  # Added for visualization
from PIL import ImageDraw, Image  # Added for visualization and image loading
from .models.model1_util.test import test_net
from .datas.ImageLoader import ImageLoader
from .datas.TextedImage import TextedImage
from .datas.textedImage_save_utils import save_timgs, load_timgs
from .datas.Dataset import (
    MangaDataset1,
    MangaDataset2,
    MangaDataset3,
)  # Added MangaDataset3
from .datas.Utils import BBox
from .models.Utils import ModelMode, tensor_rgb_to_cv2
from .models.Model1 import Model1
from .models.Model2 import Model2
from .models.Model3 import Model3  # Added Model3
from .Utils import PipelineSetting, ImagePolicy, TimgGeneration
import matplotlib.pyplot as plt
import os, json, glob
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from torchvision.transforms.functional import to_pil_image, to_tensor
from PIL import Image
from .datas.visualize_gt import visualize_craft_gt_components
import torch.nn.functional as F
from .models.model1_util import imgproc
import cv2
from torch.autograd import Variable
from .models.model1_util.model1_train import train_one_epoch_model1

import torchvision.transforms.functional as VTF
import os

from .datas.ImageLoader import ImageLoader
from .datas.TextedImage import TextedImage
from .datas.Dataset import MangaDataset1
from .models.Utils import ModelMode
from .models.Model1 import Model1
from .models.Model3 import Model3
from .models.Model3_pretrained import Model3_pretrained
from .Utils import PipelineSetting, ImagePolicy


class PipelineMgr:
    def __init__(self, setting: PipelineSetting, policy: ImagePolicy):
        self.setting = setting
        self.imageloader = ImageLoader(setting, policy)
        self.texted_images = None
        os.makedirs(self.setting.ckpt_dir, exist_ok=True)
        os.makedirs(self.setting.font_dir, exist_ok=True)
        os.makedirs(self.setting.clear_img_dir, exist_ok=True)
        os.makedirs(self.setting.output_img_dir, exist_ok=True)

    def _load_preprocessed_data(self) -> list[TextedImage]:
        """
        trit/datas/images/preprocess/ 폴더에서 전처리된 데이터를 로드하여 TextedImage 객체들을 생성
        """
        preprocess_dir = "trit/datas/images/preprocess"
        if not os.path.exists(preprocess_dir):
            raise FileNotFoundError(f"Preprocess directory not found: {preprocess_dir}")

        # JSON 파일들 찾기
        json_files = [f for f in os.listdir(preprocess_dir) if f.endswith('.json')]
        json_files.sort()  # 파일명 순서대로 정렬

        if not json_files:
            raise FileNotFoundError(f"No JSON files found in {preprocess_dir}")

        print(f"[Pipeline] Loading {len(json_files)} preprocessed files from {preprocess_dir}")

        texted_images = []

        for json_file in json_files:
            json_path = os.path.join(preprocess_dir, json_file)

            try:
                # JSON 파일 읽기
                with open(json_path, 'r') as f:
                    metadata = json.load(f)

                # 필요한 파일 경로들
                orig_path = os.path.join(preprocess_dir, metadata["orig_file"])
                timg_path = os.path.join(preprocess_dir, metadata["timg_file"])
                mask_path = os.path.join(preprocess_dir, metadata["mask_file"])

                # 이미지 로드 및 텐서 변환
                orig_img = Image.open(orig_path).convert('RGB')
                timg_img = Image.open(timg_path).convert('RGB')
                mask_img = Image.open(mask_path).convert('L')  # 그레이스케일

                print(f"[Pipeline] Original PIL image sizes:")
                print(f"  orig_img: {orig_img.size} (W x H)")
                print(f"  timg_img: {timg_img.size} (W x H)")
                print(f"  mask_img: {mask_img.size} (W x H)")

                # PIL to Tensor 변환 (0-1 범위)
                orig_tensor = torch.from_numpy(np.array(orig_img)).permute(2, 0, 1).float() / 255.0
                timg_tensor = torch.from_numpy(np.array(timg_img)).permute(2, 0, 1).float() / 255.0
                mask_tensor = torch.from_numpy(np.array(mask_img)).unsqueeze(0).float() / 255.0

                print(f"[Pipeline] Tensor shapes before resize:")
                print(f"  orig_tensor: {orig_tensor.shape} (C x H x W)")
                print(f"  timg_tensor: {timg_tensor.shape} (C x H x W)")
                print(f"  mask_tensor: {mask_tensor.shape} (C x H x W)")

                # 8의 배수 확인 및 동적 조정
                _, h, w = orig_tensor.shape
                print(f"[Pipeline] Divisibility check:")
                print(f"  Height {h}: {h} % 8 = {h % 8} ({'OK' if h % 8 == 0 else 'NEEDS RESIZE'})")
                print(f"  Width {w}: {w} % 8 = {w % 8} ({'OK' if w % 8 == 0 else 'NEEDS RESIZE'})")

                # 동적으로 8의 배수로 조정
                orig_tensor = self._resize_to_multiple_of_8(orig_tensor)
                timg_tensor = self._resize_to_multiple_of_8(timg_tensor)
                mask_tensor = self._resize_to_multiple_of_8(mask_tensor)

                # 조정 후 크기 확인
                _, final_h, final_w = orig_tensor.shape
                print(f"[Pipeline] After resize:")
                print(f"  Final size: {final_h}x{final_w}")
                print(f"  Height {final_h}: {final_h} % 8 = {final_h % 8} ({'OK' if final_h % 8 == 0 else 'ERROR!'})")
                print(f"  Width {final_w}: {final_w} % 8 = {final_w % 8} ({'OK' if final_w % 8 == 0 else 'ERROR!'})")

                # BBox 객체들 생성
                bboxes = []
                for bbox_coords in metadata["bboxes"]:
                    if len(bbox_coords) == 4:
                        x1, y1, x2, y2 = bbox_coords
                        bboxes.append(BBox(x1, y1, x2, y2))

                # TextedImage 객체 생성
                texted_image = TextedImage(
                    orig=orig_tensor,
                    timg=timg_tensor,
                    mask=mask_tensor,
                    bboxes=bboxes
                )

                texted_images.append(texted_image)
                print(f"[Pipeline] Loaded {json_file}: {len(bboxes)} bboxes")

            except Exception as e:
                print(f"[Pipeline] Error loading {json_file}: {e}")
                continue

        if not texted_images:
            raise RuntimeError("No valid preprocessed data could be loaded")

        print(f"[Pipeline] Successfully loaded {len(texted_images)} preprocessed images")
        return texted_images

    def _resize_to_multiple_of_8(self, image_tensor):
        """
        동적으로 이미지를 8의 배수 크기로 리사이즈
        어떤 크기의 입력이든 Stable Diffusion VAE 요구사항에 맞게 조정
        """
        _, h, w = image_tensor.shape
        new_h = ((h + 7) // 8) * 8  # 8의 배수로 올림
        new_w = ((w + 7) // 8) * 8  # 8의 배수로 올림

        if h != new_h or w != new_w:
            print(f"[Pipeline] Resizing {h}x{w} → {new_h}x{new_w} (8의 배수 맞춤)")

            # 고품질 리샘플링으로 리사이즈
            from torchvision import transforms
            pil_img = transforms.ToPILImage()(image_tensor)
            resized_img = pil_img.resize((new_w, new_h), Image.LANCZOS)
            return transforms.ToTensor()(resized_img)

        print(f"[Pipeline] Size {h}x{w} already compatible (no resize needed)")
        return image_tensor

    def _run_model3_pretrained_final(self):
        """
        PRETRAINED_FINAL 모드: 전처리된 데이터로 Model3_pretrained 추론만 실행
        """
        print("[Pipeline] Running Model3_pretrained Final Inference")

        # Model3_pretrained 설정 (기존 설정과 동일)
        model_pretrained_config = {
            "model_id": "stabilityai/stable-diffusion-2-inpainting",
            "prompts": "pure black and white manga style image with no color tint, absolute grayscale, contextual manga style, remove lettering, remove text, remove logo, remove watermark, consistent with surrounding",
            "negative_prompt": "photo, realistic, color, colorful, purple, violet, sepia, any color tint, blurry",
            "lora_path": self.setting.lora_weight_path,
            "epochs": self.setting.epochs,
            "batch_size": self.setting.batch_size,
            "guidance_scale": 7.5,
            "inference_steps": 28,
            "input_size": self.setting.model3_input_size,
            "lora_rank": self.setting.lora_rank,
            "lora_alpha": self.setting.lora_alpha,
            "output_dir": "trit/datas/images/output",
            "mask_weight": self.setting.mask_weight,
        }

        # texted_images_for_model3 생성 (전처리된 데이터 그대로 사용)
        texted_images_for_model3 = [
            deepcopy(texted_image) for texted_image in self.texted_images
        ]

        print(f"[Pipeline] Final check before Model3 inference:")
        for i, texted_image in enumerate(texted_images_for_model3):
            print(f"  Image {i+1}:")
            print(f"    orig shape: {texted_image.orig.shape} (C x H x W)")
            print(f"    timg shape: {texted_image.timg.shape} (C x H x W)")
            print(f"    mask shape: {texted_image.mask.shape} (C x H x W)")
            _, h, w = texted_image.orig.shape
            print(f"    Size check: {h}x{w}, H%8={h%8}, W%8={w%8}")

        # Model3_pretrained 추론 실행
        model3 = Model3_pretrained(model_pretrained_config)
        outputs = model3.inference(texted_images_for_model3)

        # 결과를 원본 이미지에 병합
        i = 0
        for texted_image in self.texted_images:
            texted_image.merge_cropped(
                outputs[i : i + len(texted_image.bboxes)]
            )
            i += len(texted_image.bboxes)

        # 최종 합성 결과 저장
        output_dir = "trit/datas/images/output"
        os.makedirs(output_dir, exist_ok=True)

        print(f"[Pipeline] Saving final composite results to {output_dir}")

        from torchvision import transforms
        for i, texted_image in enumerate(self.texted_images):
            try:
                # 최종 합성된 이미지를 PIL로 변환
                final_result = transforms.ToPILImage()(texted_image.orig.cpu())

                # 최종 결과 저장
                final_path = f"{output_dir}/final_result_{i:03d}.png"
                final_result.save(final_path)
                print(f"[Pipeline] Saved final result: {final_path}")

                # 원본 이미지도 비교용으로 저장 (timg - 텍스트가 있는 버전)
                original_with_text = transforms.ToPILImage()(texted_image.timg.cpu())
                original_path = f"{output_dir}/original_with_text_{i:03d}.png"
                original_with_text.save(original_path)
                print(f"[Pipeline] Saved original (with text): {original_path}")

            except Exception as e:
                print(f"[Pipeline] Error saving final result {i}: {e}")
                continue

        print(f"[Pipeline] Model3_pretrained Final Inference completed")
        print(f"[Pipeline] Final results saved: {len(self.texted_images)} images")

    def run(self):
        # PRETRAINED_FINAL 모드는 전처리된 데이터를 직접 로드
        if self.setting.model3_mode == ModelMode.PRETRAINED_FINAL:
            print("[Pipeline] PRETRAINED_FINAL mode: Loading preprocessed data")
            self.texted_images = self._load_preprocessed_data()
            # Change device to GPU
            self.setting.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            print(f"[Pipeline] Using Device: {self.setting.device}")

            # Model1, Model2 건너뛰고 Model3_pretrained 추론만 실행
            self._run_model3_pretrained_final()
            return

        # 기존 파이프라인 (다른 모드들)
        self._run_imageloader()
        print(f"[Pipeline] Merging bboxes with margin {self.setting.margin}")
        for texted_image in self.texted_images:
            texted_image.merge_bboxes_with_margin(self.setting.margin)
        # Change device to GPU
        self.setting.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(f"[Pipeline] Using Device: {self.setting.device}")
        # model 1
        if self.setting.model1_mode != ModelMode.SKIP:
            texted_images_for_model1 = [
                deepcopy(texted_image) for texted_image in self.texted_images
            ]
            if self.setting.model1_mode == ModelMode.TRAIN:
                print("[Pipeline] Training Model 1")
                self._model1_train(texted_images_for_model1)
            elif self.setting.model1_mode == ModelMode.INFERENCE:
                print("[Pipeline] Inference Model 1")
                self._model1_inference(texted_images_for_model1)
        else:
            print("[Pipeline] Skipping Model 1")
        # model 2
        if self.setting.model2_mode != ModelMode.SKIP:
            texted_images_for_model2 = [
                _splitted
                for texted_image in self.texted_images
                for _splitted in texted_image.split_margin_crop(self.setting.margin)
            ]
            if self.setting.model2_mode == ModelMode.TRAIN:
                print("[Pipeline] Training Model 2")
                self._model2_train(texted_images_for_model2)
            elif self.setting.model2_mode == ModelMode.INFERENCE:
                print("[Pipeline] Inference Model 2")
                self._model2_inference(texted_images_for_model2)
        else:
            print("[Pipeline] Skipping Model 2")
        # model 3
        if self.setting.model3_mode != ModelMode.SKIP:
            model_config = {
                "model_id": "stabilityai/stable-diffusion-3.5-medium",
                "prompts": "pure black and white manga style image with no color tint, absolute grayscale, contextual manga style",
                "negative_prompt": "photo, realistic, color, colorful, purple, violet, sepia, any color tint, blurry",
                "lora_path": self.setting.lora_weight_path,
                "epochs": self.setting.epochs,
                "batch_size": self.setting.batch_size,
                "inference_steps": 10,  # 기본값 : 10
                "input_size": self.setting.model3_input_size,
                "gradient_accumulation_steps": 8,  # 조절 가능 기본값 : 4
                "guidance_scale": 7.5,  #  기본값 : 7.5
                "lora_rank": self.setting.lora_rank,  # LoRA rank 값 - 작은 값으로 조정
                "lora_alpha": self.setting.lora_alpha,  # LoRA alpha 값 - 보통 rank * 2가 적당
                "output_dir": "trit/datas/images/output",  # 학습 중 시각화 결과 저장 경로
                "mask_weight": self.setting.mask_weight,
            }
            model_pretrained_config = {
                "model_id": "stabilityai/stable-diffusion-2-inpainting",
                "prompts": "pure black and white manga style image with no color tint, absolute grayscale, contextual manga style, remove lettering, remove text, remove logo, remove watermark, consistent with surrounding",
                "negative_prompt": "photo, realistic, color, colorful, purple, violet, sepia, any color tint, blurry",
                "lora_path": self.setting.lora_weight_path,
                "epochs": self.setting.epochs,
                "batch_size": self.setting.batch_size,
                "guidance_scale": 7.5,  #  기본값 : 7.5
                "inference_steps": 28,  # 기본값 : 28
                "input_size": self.setting.model3_input_size,
                "lora_rank": self.setting.lora_rank,  # LoRA rank 값
                "lora_alpha": self.setting.lora_alpha,  # LoRA alpha 값
                "output_dir": "trit/datas/images/output",  # 학습 중 시각화 결과 저장 경로
                "mask_weight": self.setting.mask_weight,
            }

            texted_images_for_model3 = [
                deepcopy(texted_image) for texted_image in self.texted_images
            ]
            if self.setting.model3_mode == ModelMode.TRAIN:
                print("[Pipeline] Training Model 3")
                model3 = Model3(model_config)
                print("[Pipeline] Calling model3.lora_train...")
                model3.lora_train(texted_images_for_model3)

            elif self.setting.model3_mode == ModelMode.INFERENCE:
                print("[Pipeline] Running Model 3 Inference")
                model3 = Model3(model_config)
                # 패치들을 인페인팅
                outputs = model3.inference(texted_images_for_model3)
                # merge_cropped
                i = 0
                for texted_image in self.texted_images:
                    texted_image.merge_cropped(
                        outputs[i : i + len(texted_image.bboxes)]
                    )
                    i += len(texted_image.bboxes)

            # 사전 훈련 모델 훈련
            elif self.setting.model3_mode == ModelMode.PRETRAINED_TRAIN:
                print("[Pipeline] Training Model 3 Pretrained")
                model3 = Model3_pretrained(model_pretrained_config)
                print("[Pipeline] Calling model3_pretrained.lora_train...")
                model3.lora_train(texted_images_for_model3)

            # 사전 훈련 모델 사용
            elif self.setting.model3_mode == ModelMode.PRETRAINED_INFERENCE:
                print("[Pipeline] Running Model 3 Pretrained Inference")
                model3 = Model3_pretrained(model_pretrained_config)
                # 패치들을 인페인팅
                outputs = model3.inference(texted_images_for_model3)
                # merge_cropped
                i = 0
                for texted_image in self.texted_images:
                    texted_image.merge_cropped(
                        outputs[i : i + len(texted_image.bboxes)]
                    )
                    i += len(texted_image.bboxes)

    def _run_imageloader(self):
        print("[Pipeline] Loading Images")
        if self.setting.timg_generation == TimgGeneration.generate_only:
            print("[Pipeline] Generating TextedImages")
            self.imageloader.start_gen_async(
                num_images=self.setting.num_images,
                dir=self.setting.clear_img_dir,
                max_text_size=self.setting.model3_input_size,
            )
            self.texted_images = self.imageloader.get_gen_images()
        elif self.setting.timg_generation == TimgGeneration.generate_save:
            print(
                "[Pipeline] Generating TextedImages and Save",
                self.setting.model3_input_size,
            )
            self.imageloader.start_gen_async(
                num_images=self.setting.num_images,
                dir=self.setting.clear_img_dir,
                max_text_size=self.setting.model3_input_size,
            )
            self.texted_images = self.imageloader.get_gen_images()
            save_timgs(self.texted_images, self.setting.texted_img_dir)
            num_viz_samples = getattr(
                self.setting, "num_gt_viz_samples", 3
            )  # 설정 또는 기본값
            num_viz_samples = min(num_viz_samples, len(self.texted_images))

            # 시각화 이미지 저장 디렉토리 (output_img_dir 내부에 생성)
            viz_save_dir = os.path.join(
                self.setting.output_img_dir, "generated_score_map_samples"
            )
            for i in range(num_viz_samples):
                if i < len(self.texted_images):  # 유효한 인덱스인지 확인
                    try:
                        # 위에서 정의한 visualize_craft_gt_components 함수 호출
                        visualize_craft_gt_components(
                            self.texted_images[i], i, viz_save_dir
                        )
                    except Exception as e:
                        print(
                            f"[Pipeline] Error during Score Map GT visualization for sample {i}: {e}"
                        )

        elif self.setting.timg_generation == TimgGeneration.use_saved:
            print("[Pipeline] Loading Saved TextedImages")
            self.texted_images = load_timgs(
                self.setting.texted_img_dir,
                self.setting.device,
                max_num=self.setting.num_images,
            )

        elif self.setting.timg_generation == TimgGeneration.test:
            """
            todo: 테스트 모드일때 test에서 이미지 로드하기
            """

    def _model1_train(self, texted_images: list[TextedImage]):
        if self.setting.model1_mode != ModelMode.SKIP:
            if self.setting.model1_mode == ModelMode.TRAIN:
                print("[Pipeline] Training Model 1")
                model1 = Model1()
                train_dataset = MangaDataset1(texted_images)
                train_loader = DataLoader(
                    train_dataset,
                    batch_size=self.setting.batch_size,
                    shuffle=True,
                    num_workers=self.setting.num_workers,
                    persistent_workers=(self.setting.num_workers > 0),
                    # pin_memory=True,
                )

                optimizer = torch.optim.Adam(
                    model1.parameters(),
                    lr=self.setting.lr,
                    weight_decay=self.setting.weight_decay,
                )

                model1_ckpt_path = os.path.join(self.setting.ckpt_dir, "model1.pth")
                start_epoch = model1.load_checkpoint(model1_ckpt_path, optimizer)

                print(f"[Pipeline] Starting Model 1 training from epoch {start_epoch}")
                for epoch in range(start_epoch, self.setting.epochs):
                    # 분리된 함수를 호출하여 한 에폭 학습 진행
                    avg_epoch_loss = train_one_epoch_model1(
                        model=model1,
                        train_loader=train_loader,
                        optimizer=optimizer,
                        device=model1.device,  # 모델이 있는 디바이스 사용
                        epoch=epoch,
                        num_epochs=self.setting.epochs,
                        vis_save_dir=self.setting.output_img_dir,
                    )

                    # 에폭 결과 출력
                    print(
                        f"[Pipeline] Epoch {epoch + 1}/{self.setting.epochs} - Model 1 Average Training Loss: {avg_epoch_loss:.4f}"
                    )

                    if (epoch + 1) % self.setting.ckpt_interval == 0:
                        model1.save_checkpoint(
                            model1_ckpt_path, epoch, optimizer.state_dict()
                        )

    def _model1_inference(self, texted_images: list[TextedImage]):
        print("[Pipeline] Running Model 1 Inference")
        model1 = Model1()
        model1_ckpt_path = os.path.join(self.setting.ckpt_dir, "model1.pth")
        model1.load_checkpoint(model1_ckpt_path)  # Optimizer not needed for inference
        model1.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        model1.eval()

        for idx, texted_image_copy in enumerate(
            tqdm(texted_images, desc="Model1 Inference")
        ):
            cv_img = tensor_rgb_to_cv2(texted_image_copy.timg)
            bboxes, polys, score_text = test_net(
                model1,  # = CRAFT network
                cv_img,  # 변환한 이미지
                cuda=(self.setting.device != "cpu"),
                poly=False,
            )

            def to_rect(pts):
                if pts is None or len(pts) == 0:
                    return None
                xs, ys = pts[:, 0], pts[:, 1]
                return BBox(int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))

            rects = [to_rect(b) for b in bboxes]  # (4,2) → (x1,y1,x2,y2)
            self.texted_images[idx].bboxes = rects

    def _model2_train(self, texted_images: list[TextedImage]):
        if self.setting.model2_mode != ModelMode.SKIP:
            if self.setting.model2_mode == ModelMode.TRAIN:
                print("[Pipeline] Training Model 2")
                model2 = Model2(n_channels=3, n_classes=1, device=self.setting.device)
                train_dataset = MangaDataset2(
                    texted_images,
                    self.setting.model2_input_size,
                )
                train_loader = DataLoader(
                    train_dataset,
                    batch_size=self.setting.batch_size,
                    shuffle=True,
                    num_workers=self.setting.num_workers,
                )
                optimizer = torch.optim.Adam(model2.parameters(), lr=self.setting.lr)
                criterion = torch.nn.BCEWithLogitsLoss()
                model2_ckpt_path = os.path.join(self.setting.ckpt_dir, "model2.pth")
                start_epoch = model2.load_checkpoint(model2_ckpt_path, optimizer)

                print(f"[Pipeline] Starting Model 2 training from epoch {start_epoch}")
                for epoch in range(start_epoch, self.setting.epochs):
                    model2.train()
                    epoch_loss = 0.0
                    batch_iterator = tqdm(
                        train_loader,
                        desc=f"Epoch {epoch+1}/{self.setting.epochs} - Model2 Train",
                        leave=False,
                    )
                    for images, true_masks in batch_iterator:
                        images = images.to(self.setting.device)
                        true_masks = true_masks.to(self.setting.device)
                        pred_mask_logits = model2(images)
                        loss = criterion(pred_mask_logits, true_masks)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        epoch_loss += loss.item()
                        batch_iterator.set_postfix(loss=f"{loss.item():.4f}")
                    avg_epoch_loss = epoch_loss / len(train_loader)
                    print(
                        f"[Pipeline] Epoch {epoch+1}/{self.setting.epochs} - Model 2 Average Training Loss: {avg_epoch_loss:.4f}"
                    )

                    if (epoch + 1) % self.setting.ckpt_interval == 0:
                        model2.save_checkpoint(
                            model2_ckpt_path, epoch, optimizer.state_dict()
                        )

    def _model2_inference(self, texted_images: list[TextedImage]):
        print("[Pipeline] Running Model 2 Inference")
        model2 = Model2(n_channels=3, n_classes=1, device=self.setting.device)
        model2_ckpt_path = os.path.join(self.setting.ckpt_dir, "model2.pth")
        model2.load_checkpoint(model2_ckpt_path)  # Optimizer not needed for inference
        model2.eval()
        outputs: list[torch.Tensor] = []
        # process
        dataset = MangaDataset2(
            texted_images,
            self.setting.model2_input_size,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=self.setting.batch_size,
            shuffle=True,
            num_workers=self.setting.num_workers,
        )
        for images, _ in tqdm(dataloader, leave=False):
            images = images.to(self.setting.device)
            with torch.no_grad():
                pred_logits = model2(images)
            pred_probabilities = torch.sigmoid(pred_logits)
            pred_mask_batch = (pred_probabilities > 0.5).float().cpu()
            # 배치 차원 flatten
            pred_mask_flattened = list(pred_mask_batch)
            outputs.extend(pred_mask_flattened)
        # 모델2 결과 적용
        for texted_image, output in zip(texted_images, outputs):
            texted_image.mask = output
        # merge_cropped
        i = 0
        for texted_image in self.texted_images:
            texted_image.merge_cropped(texted_images[i : i + len(texted_image.bboxes)])
            i += len(texted_image.bboxes)
