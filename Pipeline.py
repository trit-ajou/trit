from peft import LoraConfig
import torch
from tqdm import tqdm
from copy import copy
from torch.utils.data import DataLoader, random_split


from .datas.ImageLoader import ImageLoader
from .datas.TextedImage import TextedImage
from .datas.Dataset import MangaDataset1
from .models.Utils import ModelMode
from .models.Model3 import Model3
from .Utils import PipelineSetting, ImagePolicy
from accelerate import Accelerator

class PipelineMgr:
    def __init__(self, setting: PipelineSetting, policy: ImagePolicy):
        self.setting = setting
        self.imageloader = ImageLoader(setting, policy)

    def run(self):
        ################################################### Step 1: Load Images ##############################################
        print("[Pipeline] Loading Images")
        self.texted_images: list[TextedImage] = self.imageloader.load_images(
            self.setting.num_images, self.setting.clear_img_dir
        )

        ################################################### Step 2: BBox Merge ###############################################
        print(f"[Pipeline] Merging bboxes with margin {self.setting.margin}")
        for texted_image in self.texted_images:
            texted_image.merge_bboxes_with_margin(self.setting.margin)

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

        ################################################### Step 7: Model 3 ##################################################
        if self.setting.model3_mode != ModelMode.SKIP:
            # texted_images_for_model3 = [
            #     _splitted
            #     for texted_image in self.texted_images
            #     for _splitted in texted_image.split_center_crop(
            #         self.setting.model3_input_size
            #     )
            # ]
            batch_size = self.setting.batch_size
            texted_images_for_model3 = []
            
            for i in range(0, len(self.texted_images), batch_size):
                # 현재 배치만 메모리에 로드
                batch_images = self.texted_images[i:i+batch_size]
                
                # 배치 내 각 이미지 처리
                for texted_image in batch_images:
                    splits = texted_image.split_center_crop(self.setting.model3_input_size)
                    texted_images_for_model3.extend(splits)
                
                # 배치 처리 후 불필요한 메모리 정리
                torch.cuda.empty_cache()
            
            model_config = {
                    "model_id" : "stable-diffusion-v1-5/stable-diffusion-v1-5",
                    "prompts" : "pure black and white manga style image with no color tint, absolute grayscale, contextual manga style",
                    "negative_prompt" : "color, colorful, blurry, low quality, jpeg artifacts, photo, realistic, color, colorful, purple, violet, sepia, any color tint, blurry, low quality",
                    "lora_path" : "trit/models/lora",
                    "lora_weight_name" : "best_model.safetensors", # 변경가능
                    "epochs": self.setting.epochs,
                    "batch_size": self.setting.batch_size,
                    "lr": self.setting.lr,
                    "input_size": self.setting.model3_input_size,
                    "vis_interval": self.setting.vis_interval,
                    "ckpt_interval": self.setting.ckpt_interval,
                    "gradient_accumulation_steps": 4, # 조절 가능 기본값 : 4
                    "validation_epochs": 10, # 검증 주기
                    "lambda_ssim": 0.5, # ssim 손실 가중치
                    "lora_rank": 4, # LoRA rank 값 - 작은 값으로 조정
                    "lora_alpha": 8, # LoRA alpha 값 - 보통 rank * 2가 적당
                    "output_dir": "trit/datas/images/output" # 학습 중 시각화 결과 저장 경로
            }
            if self.setting.model3_mode == ModelMode.TRAIN:
                print("[Pipeline] Training Model 3")
                #accelerator  생성
                accelerator = Accelerator(
                    mixed_precision="fp16", # 혼합 정밀도 사용
                    gradient_accumulation_steps=model_config["gradient_accumulation_steps"],
                )
                # Model3 생성 시 lora_config 제거
                model3 = Model3(model_config) 
                
                # lora_train 호출 시 lora_config 제거
                model3.lora_train(texted_images_for_model3, accelerator)
                
            elif self.setting.model3_mode == ModelMode.INFERENCE:
                print("[Pipeline] Running Model 3 Inference")
                
                # TODO: model 3 inference, viz, apply
                model3 = Model3(model_config)
                model3.inference(texted_images_for_model3)
                
                
        else:
            print("[Pipeline] Skipping Model 3")

        ################################################### Step 8: Model 3 output apply #####################################
        if self.setting.model3_mode == ModelMode.INFERENCE:
            pass
