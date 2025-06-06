from peft import LoraConfig
import torch
from tqdm import tqdm
from copy import copy
from torch.utils.data import DataLoader, random_split
import torchvision.transforms.functional as VTF
import os

from .datas.ImageLoader import ImageLoader
from .datas.TextedImage import TextedImage
from .datas.Dataset import MangaDataset1
from .models.Utils import ModelMode
from .models.Model1 import Model1
from .models.Model3 import Model3
from .Utils import PipelineSetting, ImagePolicy

class PipelineMgr:
    def __init__(self, setting: PipelineSetting, policy: ImagePolicy):
        self.setting = setting
        self.imageloader = ImageLoader(setting, policy)

    def run(self):
        ################################################### Step 1: Load Images ##############################################
        print("[Pipeline] Loading Images")
        self.texted_images: list[TextedImage] = self.imageloader.load_images(
            self.setting.num_images,
            self.setting.clear_img_dir,
            max_text_size=self.setting.model3_input_size,
        )
        ################################################### Step 2: BBox Merge ###############################################
        print(f"[Pipeline] Merging bboxes with margin {self.setting.margin}")
        for texted_image in self.texted_images:
            texted_image.merge_bboxes_with_margin(self.setting.margin)

        # Might need to change device to GPU
        self.setting.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(f"[Pipeline] Using Device: {self.setting.device}")
        # Might need to change device to GPU
        self.setting.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
            
        

        ################################################### Step 3: Model 1 ##################################################
        if self.setting.model1_mode != ModelMode.SKIP:
            texted_images_for_model1 = [
                copy(texted_image) for texted_image in self.texted_images
            ]
            if self.setting.model1_mode == ModelMode.TRAIN:
                print("[Pipeline] Training Model 1")
                # TODO: model 1 train, viz

            elif self.setting.model1_mode == ModelMode.INFERENCE:
                print("[Pipeline] Running Model 1 Inference")
                # TODO: model 1 inference, viz, apply
        else:
            print("[Pipeline] Skipping Model 1")

        ################################################### Step 4: Model 1 output apply #####################################
        if self.setting.model1_mode == ModelMode.INFERENCE:
            pass

        ################################################### Step 5: Model 2 ##################################################
        if self.setting.model2_mode != ModelMode.SKIP:
            texted_images_for_model2 = [
                _splitted
                for texted_image in self.texted_images
                for _splitted in texted_image.split_margin_crop(self.setting.margin)
            ]
            if self.setting.model2_mode == ModelMode.TRAIN:
                print("[Pipeline] Training Model 2")
                # TODO: model 2 train, viz
            elif self.setting.model2_mode == ModelMode.INFERENCE:
                print("[Pipeline] Running Model 2 Inference")
                # TODO: model 2 inference, viz, apply
        else:
            print("[Pipeline] Skipping Model 2")

        ################################################### Step 6: Model 2 output apply #####################################
        if self.setting.model2_mode == ModelMode.INFERENCE:
            pass


        # for i, texted_image in enumerate(self.texted_images):
        #     texted_image.visualize(dir="trit/datas/images/output", filename=f"before_model3_images{i}.png")

        ################################################### Step 7: Model 3 ##################################################
        if self.setting.model3_mode != ModelMode.SKIP:
            
            
            model_config = {
                    "model_id" : "stabilityai/stable-diffusion-3.5-medium",
                    "prompts" : "pure black and white manga style image with no color tint, absolute grayscale, contextual manga style",
                    "negative_prompt" : "photo, realistic, color, colorful, purple, violet, sepia, any color tint, blurry",
                    "lora_path" : self.setting.lora_weight_path,
                    "epochs": self.setting.epochs,
                    "batch_size": self.setting.batch_size,
                    "inference_steps" : 3, # 기본값 : 10
                    "input_size": self.setting.model3_input_size,
                    "gradient_accumulation_steps": 8, # 조절 가능 기본값 : 4
                    "guidance_scale": 7.5, #  기본값 : 7.5
                    "lora_rank": self.setting.lora_rank, # LoRA rank 값 - 작은 값으로 조정
                    "lora_alpha": self.setting.lora_alpha, # LoRA alpha 값 - 보통 rank * 2가 적당
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
                
                for i, texted_image in enumerate(texted_images_for_model3):
                    texted_image.visualize(dir="trit/datas/images/output", filename=f"before_inpainting{i}.png")
                
                
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
        if self.setting.model3_mode == ModelMode.INFERENCE:
            pass
