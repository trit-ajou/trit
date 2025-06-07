import gc
import glob
from peft import LoraConfig
import torch
import time
import os
from tqdm import tqdm
from copy import copy, deepcopy
from torch.utils.data import DataLoader, random_split, Dataset
from typing import Optional # Changed from copy to deepcopy for TextedImage objects
from torch.utils.data import DataLoader, random_split

import os  # Added for path joining and directory creation
import torchvision.transforms.functional as VTF  # Added for visualization
from PIL import ImageDraw  # Added for visualization
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
from .models.Utils import ModelMode
from .models.Model1 import Model1
from .models.Model2 import Model2
from .models.Model3 import Model3
from .models.Model3_pretrained import Model3_pretrained
from .Utils import PipelineSetting, ImagePolicy


class PipelineMgr:
    def __init__(self, setting: PipelineSetting, policy: ImagePolicy):
        self.setting = setting
        self.imageloader = ImageLoader(setting, policy)
        self.model2: Optional[Model2] = None
        self.texted_images: list[TextedImage] = []
        os.makedirs(self.setting.ckpt_dir, exist_ok=True)
        os.makedirs(self.setting.font_dir, exist_ok=True)
        os.makedirs(self.setting.clear_img_dir, exist_ok=True)
        os.makedirs(self.setting.output_img_dir, exist_ok=True)
        
    def _load_and_preprocess_base_images(self) -> bool:
        print("[Pipeline] Loading base images...")
        self.imageloader.start_loading_async(
            num_images=self.setting.num_images_per_load,
            dir=self.setting.clear_img_dir,
            max_text_size=self.setting.model3_input_size,
        )
        loaded_raw_images = self.imageloader.get_loaded_images()
        if not loaded_raw_images:
            print("[Error] ImageLoader returned no images.")
            self.texted_images = []
            return False
        self.texted_images = loaded_raw_images
        print(f"[Pipeline] Loaded {len(self.texted_images)} base images.")
        print(f"[Pipeline] Merging bboxes with margin {self.setting.margin} for loaded images.")

    def run(self):
        ################################################### Step 1: Load Images ##############################################
        print("[Pipeline] Loading Images")
#         # 이미지로더 사용 방법 예시(NEW)
#         self.imageloader.start_loading_async(
#             num_images=self.setting.num_images,
#             dir=self.setting.clear_img_dir,
#             max_text_size=self.setting.model3_input_size,
#         )
#         # 할일 하기
#         time.sleep(5)
#         # 로딩된 이미지 불러오기(덜끝났으면 끝날 때까지 대기)
#         self.texted_images = self.imageloader.get_loaded_images()
#         # 프로그램 종료 시
#         self.imageloader.shutdown()

        if self.setting.timg_generation == TimgGeneration.generate_only:
            print("[Pipeline] Generating TextedImages")
#             self.texted_images = self.imageloader.load_images(
#                 self.setting.num_images, self.setting.clear_img_dir, self.setting.model3_input_size)
            self.imageloader.start_loading_async(
						num_images=self.setting.num_images,
						dir=self.setting.clear_img_dir,
						max_text_size=self.setting.model3_input_size,
					)
            self.texted_images = self.imageloader.get_loaded_images()
        elif self.setting.timg_generation == TimgGeneration.generate_save:
            print("[Pipeline] Generating TextedImages and Save", self.setting.model3_input_size)
            self.imageloader.start_loading_async(
						num_images=self.setting.num_images,
						dir=self.setting.clear_img_dir,
						max_text_size=self.setting.model3_input_size,
					)
            self.texted_images = self.imageloader.get_loaded_images()
            save_timgs(self.texted_images, self.setting.texted_img_dir)
            num_viz_samples = getattr(self.setting, 'num_gt_viz_samples', 3)  # 설정 또는 기본값
            num_viz_samples = min(num_viz_samples, len(self.texted_images))

            # 시각화 이미지 저장 디렉토리 (output_img_dir 내부에 생성)
            viz_save_dir = os.path.join(self.setting.output_img_dir, "generated_score_map_samples")
            for i in range(num_viz_samples):
                if i < len(self.texted_images):  # 유효한 인덱스인지 확인
                    try:
                        # 위에서 정의한 visualize_craft_gt_components 함수 호출
                        visualize_craft_gt_components(self.texted_images[i], i, viz_save_dir)
                    except Exception as e:
                        print(f"[Pipeline] Error during Score Map GT visualization for sample {i}: {e}")

        elif self.setting.timg_generation == TimgGeneration.use_saved:
            print("[Pipeline] Loading Saved TextedImages")
            self.texted_images = load_timgs(self.setting.texted_img_dir, self.setting.device, max_num = self.setting.num_images)

        elif self.setting.timg_generation == TimgGeneration.test:
            '''
                todo: 테스트 모드일때 test에서 이미지 로드하기
            '''


        ################################################### Step 2: BBox Merge ###############################################
        print(f"[Pipeline] Merging bboxes with margin {self.setting.margin}")
        for texted_image in self.texted_images:
            texted_image.merge_bboxes_with_margin(self.setting.margin)
        return True

    def _prepare_data_for_model2(self, base_texted_images: list[TextedImage]) -> list[TextedImage]:
        texted_images_for_model2 = []
        if not base_texted_images:
            return []
        print("[Pipeline] Preparing data for Model 2: Splitting and Resizing...")
        for texted_image in base_texted_images:
            try:
                splitted_images = texted_image.split_margin_crop(self.setting.margin)
                texted_images_for_model2.extend(splitted_images)
            except Exception as e:
                print(f"[Warning] Error processing image for split_margin_crop: {e}")
                continue
        if not texted_images_for_model2:
            print("[Warning] No images after split_margin_crop for Model 2.")
            return []
        print(
            f"[Pipeline] Resizing {len(texted_images_for_model2)} images to {self.setting.model2_input_size} for Model 2."
        )
        resized_images = []
        for i, ti_crop in enumerate(texted_images_for_model2):
            try:
                ti_crop._resize(self.setting.model2_input_size)
                resized_images.append(ti_crop)
            except ValueError as e:
                print(f"[Warning] Skipping image {i} during resize for Model 2: {e}")
                continue
        print(f"[Pipeline] Successfully prepared {len(resized_images)} images for Model 2.")
        return resized_images

    def run(self):
        if not self._load_and_preprocess_base_images():
            print("[Error] Initial image loading failed. Exiting.")
            self.imageloader.shutdown()
            return

        self.setting.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Pipeline] Using Device: {self.setting.device}")

        ################################################### Step 3: Model 1 ##################################################
        if self.setting.model1_mode != ModelMode.SKIP:
            # This list is already created by the existing code structure if we assume
            # texted_images are loaded before this check.
            # For Model1 training, we'll use a copy of the loaded self.texted_images
            # to avoid altering the original list that might be used by other models.
            texted_images_for_model1 = [
                deepcopy(texted_image) for texted_image in self.texted_images
            ]
            if self.setting.model1_mode == ModelMode.TRAIN:
                print("[Pipeline] Training Model 1")
                model1 = Model1()
                # model1.to(self.setting.device) # This is handled by BaseModel's __init__

                # Consider splitting texted_images_for_model1 into train/val sets for robust evaluation.
                # For now, using the full list for training as per initial plan.
                train_dataset = MangaDataset1(texted_images_for_model1)
                train_loader = DataLoader(
                    train_dataset,
                    batch_size=self.setting.batch_size,
                    shuffle=True,
                    num_workers=self.setting.num_workers,
                    collate_fn=None, # 또는 명시적으로 default_collate 사용 (보통 None이면 알아서 default_collate)
                    persistent_workers= (self.setting.num_workers > 0),
                    pin_memory = True,
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

            elif self.setting.model1_mode == ModelMode.INFERENCE:
                print("[Pipeline] Running Model 1 Inference")

                model1 = Model1()
                model1_ckpt_path = os.path.join(self.setting.ckpt_dir, "model1.pth")
                model1.load_checkpoint(
                    model1_ckpt_path
                )  # Optimizer not needed for inference
                model1.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
                model1.eval()

                print(
                    f"[Pipeline] Applying Model 1 predictions to {len(texted_images_for_model1)} images..."
                )
                for idx, texted_image_copy in enumerate(
                    tqdm(texted_images_for_model1, desc="Model1 Inference")
                ):

                    cv_img = tensor_rgb_to_cv2(texted_image_copy.timg)
                    bboxes, polys, score_text = test_net(
                        model1,  # = CRAFT network
                        cv_img,  # 변환한 이미지
                        cuda=(self.setting.device != "cpu"),
                        poly=False,

                    )
                    def to_rect(pts):
                        """
                        pts : np.ndarray([[x, y], ...])     ─ shape (4,2) or (N,2)
                        return : [x1, y1, x2, y2]  (좌상, 우하)
                        """
                        if pts is None or len(pts) == 0:
                            return None
                        xs, ys = pts[:, 0], pts[:, 1]
                        return BBox(int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))

                    rects = [to_rect(b) for b in bboxes]  # (4,2) → (x1,y1,x2,y2)

                    self.texted_images[idx].bboxes = rects

                print(
                    "[Pipeline] Model 1 Inference complete. BBoxes in self.texted_images updated."
                )

                # Visualization for a few samples (e.g., first 3)
                num_viz_samples = min(3, len(self.texted_images))
                if num_viz_samples > 0:
                    print(
                        f"[Pipeline] Visualizing first {num_viz_samples} Model 1 inference results..."
                    )
                for i in range(num_viz_samples):
                    # Use a deepcopy of the *updated* self.texted_images[i] for visualization
                    img_to_viz = deepcopy(self.texted_images[i])

                    try:
                        # Convert timg to PIL to draw on it
                        pil_timg = VTF.to_pil_image(img_to_viz.timg.cpu())
                        draw = ImageDraw.Draw(pil_timg)

                        # Update timg of the copied object with the image that has boxes drawn
                        img_to_viz.timg = VTF.to_tensor(pil_timg).to(
                            self.setting.device
                        )

                        viz_filename = f"model1_inference_viz_sample{i}.png"
                        # The visualize method of TextedImage shows orig, timg, and mask.
                        # By updating img_to_viz.timg, the timg panel in the visualization will show the predictions.
                        img_to_viz.visualize(
                            dir=self.setting.output_img_dir, filename=viz_filename
                        )
                        print(
                            f"Saved Model 1 inference visualization to {os.path.join(self.setting.output_img_dir, viz_filename)}"
                        )
                    except Exception as e:
                        print(
                            f"[Pipeline] Error during Model 1 inference visualization for sample {i}: {e}"
                        )

        else:
            print("[Pipeline] Skipping Model 1")

        ################################################### Step 4: Model 1 output apply #####################################
        # This step is now effectively handled within the INFERENCE block above,
        # as self.texted_images[idx].bboxes are updated directly.
        # No separate application step is needed here for Model1 if inference was run.
        if self.setting.model1_mode == ModelMode.INFERENCE:
            print(
                "[Pipeline] Model 1 output (bboxes) has been applied to self.texted_images."
            )
            pass  # Placeholder for any further common logic if needed, otherwise can be removed.

        ################################################### Step 5: Model 2 ##################################################
        
        if self.setting.model2_mode != ModelMode.SKIP:
            if self.setting.model2_mode == ModelMode.TRAIN:
                print("[Pipeline] Starting Model 2 Training with periodic data reloading.")
                initial_data_for_model2 = self._prepare_data_for_model2(self.texted_images)
                if initial_data_for_model2:
                    self.train_model2(initial_data_for_model2)
                else:
                    print("[Error] No data prepared for Model 2 initial training. Skipping Model 2 training.")
            elif self.setting.model2_mode == ModelMode.INFERENCE:
                print("[Pipeline] Preparing data for Model 2 Inference.")
                data_for_model2_inference = self._prepare_data_for_model2(self.texted_images)
                if data_for_model2_inference:
                    print("[Pipeline] Applying Model 2 results")
                    if self.model2 is None:
                        self.model2 = Model2(num_classes=2, pretrained=True).to(self.setting.device)
                    self.inference_model2(data_for_model2_inference)
                else:
                    print("[Error] No data prepared for Model 2 inference. Skipping.")
        else:
            print("[Pipeline] Skipping Model 2")

        ################################################### Step 7: Model 3 ##################################################
        if self.setting.model3_mode != ModelMode.SKIP:

            model_config = {
                    "model_id" : "stabilityai/stable-diffusion-3.5-medium",
                    "prompts" : "pure black and white manga style image with no color tint, absolute grayscale, contextual manga style",
                    "negative_prompt" : "photo, realistic, color, colorful, purple, violet, sepia, any color tint, blurry",
                    "lora_path" : self.setting.lora_weight_path,
                    "epochs": self.setting.epochs,
                    "batch_size": self.setting.batch_size,
                    "inference_steps" : 10, # 기본값 : 10
                    "input_size": self.setting.model3_input_size,
                    "gradient_accumulation_steps": 8, # 조절 가능 기본값 : 4
                    "guidance_scale": 7.5, #  기본값 : 7.5
                    "lora_rank": self.setting.lora_rank, # LoRA rank 값 - 작은 값으로 조정
                    "lora_alpha": self.setting.lora_alpha, # LoRA alpha 값 - 보통 rank * 2가 적당
                    "output_dir": "trit/datas/images/output", # 학습 중 시각화 결과 저장 경로
                    "mask_weight": self.setting.mask_weight
                    }
            
            model_pretrained_config = {
                "model_id" : "stabilityai/stable-diffusion-2-inpainting",
                "prompts" : "pure black and white manga style image with no color tint, absolute grayscale, contextual manga style, remove lettering, remove text, remove logo, remove watermark, consistent with surrounding",
                "negative_prompt" : "photo, realistic, color, colorful, purple, violet, sepia, any color tint, blurry",
                "lora_path" : self.setting.lora_weight_path,
                "epochs": self.setting.epochs,
                "batch_size": self.setting.batch_size,
                "guidance_scale": 7.5, #  기본값 : 7.5
                "inference_steps" : 28, # 기본값 : 28
                "input_size": self.setting.model3_input_size,
                "lora_rank": self.setting.lora_rank, # LoRA rank 값
                "lora_alpha": self.setting.lora_alpha, # LoRA alpha 값
                "output_dir": "trit/datas/images/output", # 학습 중 시각화 결과 저장 경로
                "mask_weight": self.setting.mask_weight
                }
            
            if self.setting.model3_mode == ModelMode.TRAIN:

                # 학습시에만 사용용
                texted_images_for_model3 = [
                _splitted
                for texted_image in self.texted_images
                for _splitted in texted_image.split_center_crop(
                    self.setting.model3_input_size
                )]
                
                print("[Pipeline] Training Model 3")
                model3 = Model3(model_config)
                print("[Pipeline] Calling model3.lora_train...")
                model3.lora_train(texted_images_for_model3)

            elif self.setting.model3_mode == ModelMode.INFERENCE:
                print("[Pipeline] Running Model 3 Inference")
                
                model3 = Model3(model_config)

                # 각 원본 이미지에 대해 center crop으로 패치 생성
                texted_images_for_model3 = [
                    _splitted
                    for texted_image in self.texted_images
                    for _splitted in texted_image.split_center_crop(
                        self.setting.model3_input_size
                    )
                ]
                
                # 출력 디렉토리 생성
                output_dir = "trit/datas/images/output"
                os.makedirs(output_dir, exist_ok=True)

                for i, texted_image in enumerate(texted_images_for_model3):
                    texted_image.visualize(dir=output_dir, filename=f"before_inpainting{i}.png")
                
                # 패치들을 인페인팅
                inpainted_patches = model3.inference(texted_images_for_model3)

                # # 인페인팅된 패치들을 원본 이미지에 다시 합성
                # # 패치들을 원본 이미지별로 그룹화하여 합성
                patch_idx = 0
                for texted_image in self.texted_images:
                    # 현재 이미지의 bbox 개수만큼 패치 가져오기
                    num_bboxes = len(texted_image.bboxes)
                    current_patches = inpainted_patches[patch_idx:patch_idx + num_bboxes]

                    # 패치들을 원본 이미지에 합성
                    texted_image.merge_cropped(current_patches)

                    patch_idx += num_bboxes

                # 결과 이미지 저장
                output_dir = "trit/datas/images/output"
                os.makedirs(output_dir, exist_ok=True)
                for i, texted_image in enumerate(self.texted_images):
                    texted_image.visualize(dir=output_dir, filename=f"final_inpainted_{i}.png")

                print(f"[Pipeline] Inpainting completed. Results saved to {output_dir}")

            # 사전 훈련 모델 훈련
            elif self.setting.model3_mode == ModelMode.PRETRAINED_TRAIN:

                # 학습시에만 사용
                texted_images_for_model3 = [
                _splitted
                for texted_image in self.texted_images
                for _splitted in texted_image.split_center_crop(
                    self.setting.model3_input_size
                )]

                print("[Pipeline] Training Model 3 Pretrained")
                model3 = Model3_pretrained(model_pretrained_config)
                print("[Pipeline] Calling model3_pretrained.lora_train...")
                model3.lora_train(texted_images_for_model3)

            # 사전 훈련 모델 사용
            elif self.setting.model3_mode == ModelMode.PRETRAINED:
                                
                print("[Pipeline] Running Model 3 Pretrained Inference")
                model3 = Model3_pretrained(model_pretrained_config)    

                # 각 원본 이미지에 대해 center crop으로 패치 생성
                texted_images_for_model3 = [
                    _splitted
                    for texted_image in self.texted_images
                    for _splitted in texted_image.split_center_crop(
                        self.setting.model3_input_size
                    )
                ]

                # 출력 디렉토리 생성
                output_dir = "trit/datas/images/output"
                os.makedirs(output_dir, exist_ok=True)

                for i, texted_image in enumerate(texted_images_for_model3):
                    texted_image.visualize(dir=output_dir, filename=f"before_inpainting{i}.png")
                
                # 패치들을 인페인팅
                inpainted_patches = model3.inference(texted_images_for_model3)

                # # 인페인팅된 패치들을 원본 이미지에 다시 합성
                # # 패치들을 원본 이미지별로 그룹화하여 합성
                patch_idx = 0
                for texted_image in self.texted_images:
                    # 현재 이미지의 bbox 개수만큼 패치 가져오기
                    num_bboxes = len(texted_image.bboxes)
                    current_patches = inpainted_patches[patch_idx:patch_idx + num_bboxes]

                    # 패치들을 원본 이미지에 합성
                    texted_image.merge_cropped(current_patches)

                    patch_idx += num_bboxes

                # 결과 이미지 저장
                output_dir = "trit/datas/images/output"
                os.makedirs(output_dir, exist_ok=True)
                for i, texted_image in enumerate(self.texted_images):
                    texted_image.visualize(dir=output_dir, filename=f"final_inpainted_{i}.png")

                print(f"[Pipeline] Inpainting completed. Results saved to {output_dir}")

        else:
            print("[Pipeline] Skipping Model 3")

        ################################################### Step 8: Model 3 output apply #####################################

        if self.setting.model3_mode == ModelMode.INFERENCE or self.setting.model3_mode == ModelMode.PRETRAINED:
            pass

    @staticmethod
    def _create_dataloaders(
        processed_data_list: list[TextedImage],
        batch_size: int,
        num_workers: int,
        train_valid_split: float,
        device_type: str,
        persistent_workers: bool,
    ):
        if not processed_data_list:
            print("[DataLoader] No processed data provided.")
            return None, None
        dataset = MangaDataset2(processed_data_list, transform=True)
        if len(dataset) == 0:
            print("[DataLoader] Created dataset is empty.")
            return None, None
        train_dataset: Optional[Dataset] = None
        val_dataset: Optional[Dataset] = None
        if len(dataset) == 1:
            print("[DataLoader] Dataset size is 1. Using for training, no validation split.")
            train_dataset = dataset
        elif train_valid_split <= 0.0:
            print("[DataLoader] train_valid_split <= 0.0. Using all data for validation.")
            val_dataset = dataset
        elif train_valid_split >= 1.0:
            print("[DataLoader] train_valid_split >= 1.0. Using all data for training.")
            train_dataset = dataset
        else:
            train_size = int((1.0 - train_valid_split) * len(dataset))
            val_size = len(dataset) - train_size
            if train_size == 0 and val_size > 0:
                if val_size == len(dataset) and len(dataset) > 1:
                    train_size = 1
                    val_size = len(dataset) - 1
            elif val_size == 0 and train_size > 0:
                if train_size == len(dataset) and len(dataset) > 1:
                    val_size = 1
                    train_size = len(dataset) - 1
            if train_size > 0 and val_size > 0:
                train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
            elif train_size > 0:
                train_dataset = dataset
                print(
                    f"[DataLoader] Validation split resulted in 0 samples. Using all {len(dataset)} samples for training."
                )
            elif val_size > 0:
                val_dataset = dataset
                print(
                    f"[DataLoader] Training split resulted in 0 samples. Using all {len(dataset)} samples for validation."
                )
            else:
                print("[Error DataLoader] Both train and val splits are zero. Using all for train.")
                train_dataset = dataset
        train_loader = None
        if train_dataset and len(train_dataset) > 0:
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                drop_last=True,
                pin_memory=(device_type == "cuda"),
            )
        val_loader = None
        if val_dataset and len(val_dataset) > 0:
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                drop_last=True,
                pin_memory=(device_type == "cuda"),
            )
        if train_loader:
            print(f"[DataLoader] Train loader created with {len(train_dataset)} samples.")
        if val_loader:
            print(f"[DataLoader] Validation loader created with {len(val_dataset)} samples.")
        if not train_loader and not val_loader:
            print("[DataLoader] Failed to create any loaders.")
        return train_loader, val_loader

    def train_model2(self, initial_processed_data: list[TextedImage]):
        current_processed_data = initial_processed_data
        if self.model2 is None:
            self.model2 = Model2(num_classes=2, pretrained=True).to(self.setting.device)
        optimizer = torch.optim.Adam(
            self.model2.parameters(), lr=self.setting.lr, weight_decay=self.setting.weight_decay
        )
        criterion = torch.nn.BCELoss()
        scaler = torch.amp.GradScaler("cuda", enabled=self.setting.use_amp)
        start_epoch = 0
        final_checkpoint_path = f"{self.setting.ckpt_dir}/model2_final.pth"
        latest_epoch_checkpoint_path = None
        if os.path.exists(final_checkpoint_path):
            latest_epoch_checkpoint_path = final_checkpoint_path
        else:
            checkpoint_files = glob.glob(f"{self.setting.ckpt_dir}/model2_epoch_*.pth")
            if checkpoint_files:
                latest_epoch_checkpoint_path = max(checkpoint_files, key=lambda x: int(x.split("_")[-1].split(".")[0]))
        if latest_epoch_checkpoint_path:
            print(f"[Model2 Train] Loading checkpoint from: {latest_epoch_checkpoint_path}")
            checkpoint = torch.load(latest_epoch_checkpoint_path, map_location=self.setting.device)
            self.model2.load_state_dict(checkpoint["model_state_dict"])
            if "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if "epoch" in checkpoint:
                start_epoch = checkpoint["epoch"]  # Corrected: epoch in checkpoint is the completed epoch
            if "scaler_state_dict" in checkpoint and self.setting.use_amp:
                scaler.load_state_dict(checkpoint["scaler_state_dict"])
            print(f"[Model2 Train] Resuming from epoch {start_epoch}.")  # Will start training epoch `start_epoch`
        else:
            print("[Model2 Train] No checkpoint found. Starting fresh training.")

        train_loader, val_loader = self._create_dataloaders(
            current_processed_data,
            self.setting.batch_size,
            self.setting.num_workers,
            self.setting.train_valid_split,
            self.setting.device.type,
            self.setting.num_workers > 0,
        )
        if not train_loader:
            print("[Error] Could not create a train_loader with initial data. Aborting Model 2 training.")
            return

        for epoch in range(start_epoch, self.setting.epochs):
            actual_epoch_num = epoch + 1  # For display and saving
            epoch_start_time = time.time()
            if epoch > start_epoch and epoch % self.setting.reload_data_interval == 0:
                print(f"[Model2 Train] Epoch {actual_epoch_num}: Reloading training data...")

                # Step 1: Explicitly delete old DataLoaders and related data, then clear GPU cache
                # to free up memory BEFORE loading new data.
                if "train_loader" in locals() and train_loader is not None:
                    print("[Model2 Train] Deleting old train_loader.")
                    del train_loader
                    train_loader = None  # Ensure it's None so it's re-created or training is skipped
                if "val_loader" in locals() and val_loader is not None:
                    print("[Model2 Train] Deleting old val_loader.")
                    del val_loader
                    val_loader = None  # Ensure it's None

                if "current_processed_data" in locals() and current_processed_data is not None:
                    print("[Model2 Train] Clearing current_processed_data list.")
                    del current_processed_data
                    current_processed_data = None

                # Explicitly clear the instance variable self.texted_images
                # which holds TextedImage objects (with GPU tensors) from the *previous* load.
                if hasattr(self, "texted_images") and self.texted_images:
                    print(f"[Model2 Train] Clearing self.texted_images (contains {len(self.texted_images)} items).")
                    del self.texted_images
                    self.texted_images = []  # Re-initialize to an empty list to ensure old list is dereferenced

                print("[Model2 Train] Calling torch.cuda.empty_cache() and gc.collect().")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

                time.sleep(5)

                # Step 2: Load new data
                # _load_and_preprocess_base_images loads data to self.texted_images (potentially on GPU)
                if self._load_and_preprocess_base_images():
                    # _prepare_data_for_model2 processes self.texted_images (potentially on GPU)
                    new_processed_data_for_epoch = self._prepare_data_for_model2(self.texted_images)
                    if new_processed_data_for_epoch:
                        # Update current_processed_data for the new DataLoaders
                        current_processed_data = new_processed_data_for_epoch
                        print("[Model2 Train] Recreating DataLoaders with new data.")
                        train_loader, val_loader = self._create_dataloaders(
                            current_processed_data,
                            self.setting.batch_size,
                            self.setting.num_workers,
                            self.setting.train_valid_split,
                            self.setting.device.type,
                            self.setting.num_workers > 0,
                        )
                        if not train_loader:
                            print(
                                "[Error] Failed to create train_loader after data reload. Training for this epoch will be skipped."
                            )
                            # train_loader remains None, will be caught by the check below
                    else:
                        print(
                            "[Model2 Train] Failed to process reloaded images. Training for this epoch will be skipped as no data."
                        )
                        train_loader = None  # Ensure train_loader is None to skip training phase
                        val_loader = None
                else:
                    print(
                        "[Model2 Train] Failed to load new base images. Training for this epoch will be skipped as no data."
                    )
                    train_loader = None  # Ensure train_loader is None to skip training phase
                    val_loader = None

            # Check if train_loader is available for the current epoch's training
            if not train_loader:
                print(
                    f"[Model2 Train] Epoch {actual_epoch_num}: No train_loader available. Skipping training phase for this epoch."
                )
                # If checkpoints are saved based on epoch interval, you might want to skip saving
                # or handle this state appropriately if it persists.
                if actual_epoch_num % self.setting.ckpt_interval == 0:
                    print(
                        f"[Model2 Train] Epoch {actual_epoch_num}: Skipping checkpoint save due to no training data/loader."
                    )
                # Adding a small delay can prevent a tight loop if data loading continuously fails.
                time.sleep(1)
                continue  # Go to the next epoch

            self.model2.train()

            running_train_loss = 0.0
            train_pbar = tqdm(train_loader, desc=f"Epoch {actual_epoch_num}/{self.setting.epochs} Trn")
            for timg, mask in train_pbar:
                timg = timg.to(self.setting.device, non_blocking=True)
                mask = mask.to(self.setting.device, non_blocking=True)
                if mask.dim() == 3:
                    mask = mask.unsqueeze(1)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=self.setting.use_amp):
                    output = self.model2(timg)
                    if isinstance(output, dict):
                        output_for_loss = output["out"][:, 1:2, :, :]
                        loss = criterion(output_for_loss, mask)
                        if "aux" in output and output["aux"] is not None:
                            aux_output_for_loss = output["aux"][:, 1:2, :, :]
                            loss += 0.4 * criterion(aux_output_for_loss, mask)
                    else:
                        output_for_loss = output[:, 1:2, :, :] if output.shape[1] == 2 else output
                        loss = criterion(output_for_loss, mask)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                running_train_loss += loss.item()
                train_pbar.set_postfix(loss=f"{loss.item():.4f}")
            avg_train_loss = running_train_loss / len(train_loader) if len(train_loader) > 0 else 0.0

            avg_val_loss = 0.0
            if val_loader and len(val_loader) > 0 and actual_epoch_num % self.setting.vis_interval == 0:
                self.model2.eval()
                running_val_loss = 0.0
                val_pbar = tqdm(val_loader, desc=f"Epoch {actual_epoch_num}/{self.setting.epochs} Val")
                with torch.no_grad():
                    for timg, mask in val_pbar:
                        timg = timg.to(self.setting.device, non_blocking=True)
                        mask = mask.to(self.setting.device, non_blocking=True)
                        if mask.dim() == 3:
                            mask = mask.unsqueeze(1)
                        with torch.amp.autocast("cuda", enabled=self.setting.use_amp):
                            output = self.model2(timg)
                            if isinstance(output, dict):
                                # FIX: Use the sliced output for criterion
                                main_output_for_loss = output["out"][:, 1:2, :, :]
                                loss = criterion(main_output_for_loss, mask)
                                if (
                                    "aux" in output and output["aux"] is not None
                                ):  # Consider aux loss in validation if desired
                                    aux_output_for_loss = output["aux"][:, 1:2, :, :]
                                    loss += 0.4 * criterion(aux_output_for_loss, mask)
                            else:
                                # FIX: Use the sliced output for criterion
                                if output.shape[1] == 2:
                                    simple_output_for_loss = output[:, 1:2, :, :]
                                    loss = criterion(simple_output_for_loss, mask)
                                else:  # Assumes output is already [B, 1, H, W]
                                    loss = criterion(output, mask)
                        running_val_loss += loss.item()
                        val_pbar.set_postfix(loss=f"{loss.item():.4f}")
                avg_val_loss = running_val_loss / len(val_loader) if len(val_loader) > 0 else 0.0
                print(
                    f"Epoch {actual_epoch_num}/{self.setting.epochs} | Time: {time.time() - epoch_start_time:.2f}s | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}"
                )
            else:
                print(
                    f"Epoch {actual_epoch_num}/{self.setting.epochs} | Time: {time.time() - epoch_start_time:.2f}s | Train Loss: {avg_train_loss:.4f}"
                )

            if actual_epoch_num % self.setting.ckpt_interval == 0:
                os.makedirs(self.setting.ckpt_dir, exist_ok=True)
                ckpt_path = f"{self.setting.ckpt_dir}/model2_epoch_{actual_epoch_num}.pth"
                save_obj = {
                    "epoch": actual_epoch_num,
                    "model_state_dict": self.model2.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg_train_loss,
                }
                if self.setting.use_amp:
                    save_obj["scaler_state_dict"] = scaler.state_dict()
                torch.save(save_obj, ckpt_path)
                print(f"Checkpoint saved to {ckpt_path}")

        os.makedirs(self.setting.ckpt_dir, exist_ok=True)
        final_model_path = f"{self.setting.ckpt_dir}/model2_final.pth"
        final_save_obj = {
            "epoch": self.setting.epochs,
            "model_state_dict": self.model2.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_train_loss,
        }
        if self.setting.use_amp:
            final_save_obj["scaler_state_dict"] = scaler.state_dict()
        torch.save(final_save_obj, final_model_path)
        print(f"[Model2 Train] Training completed. Final model saved to {final_model_path}")

    def inference_model2(self, texted_images_for_model2: list[TextedImage]):
        """Run Model2 inference for pixel-wise text segmentation"""
        # Load trained model if available
        import glob

        # First try to load final model
        final_checkpoint = f"{self.setting.ckpt_dir}/model2_final.pth"
        if os.path.exists(final_checkpoint):
            checkpoint = torch.load(final_checkpoint, map_location=self.setting.device)
            self.model2.load_state_dict(checkpoint["model_state_dict"])
            print(f"[Model2] Loaded final model from {final_checkpoint}")
        else:
            # Try to load latest epoch checkpoint
            checkpoint_pattern = f"{self.setting.ckpt_dir}/model2_epoch_*.pth"
            checkpoint_files = glob.glob(checkpoint_pattern)

            if checkpoint_files:
                # Get the latest checkpoint by epoch number
                latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split("_")[-1].split(".")[0]))
                checkpoint = torch.load(latest_checkpoint, map_location=self.setting.device)
                self.model2.load_state_dict(checkpoint["model_state_dict"])
                print(f"[Model2] Loaded checkpoint from {latest_checkpoint}")
            else:
                print("[Model2] No checkpoint found, using pretrained weights")

        # Images are already resized and validated, no need for additional processing
        print(f"[Model2 Inference] Processing {len(texted_images_for_model2)} pre-processed images")

        results = []

        # Run inference
        self.model2.eval()
        with torch.no_grad():
            for i, texted_image in enumerate(tqdm(texted_images_for_model2, desc="Model2 Inference")):
                # Prepare input
                timg = texted_image.timg.unsqueeze(0).to(self.setting.device)

                # Get prediction
                predicted_mask = self.model2.predict_mask(timg, threshold=0.5)

                # Update mask - ensure it maintains correct format (1, H, W)
                texted_image.mask = predicted_mask.unsqueeze(0).cpu()

                results.append(texted_image)

        i = 0
        for idx, texted_image in enumerate(self.texted_images):
            texted_image.merge_cropped(results[i : i + len(texted_image.bboxes)])
            texted_image.visualize(self.setting.output_img_dir, f"model2_inference_{idx}.png")
            i += len(texted_image.bboxes)
