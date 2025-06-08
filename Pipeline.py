from peft import LoraConfig
import torch
from tqdm import tqdm
from copy import deepcopy  # Changed from copy to deepcopy for TextedImage objects
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
from .datas.textedImage_save_utils import load_timgs, LoadingMode

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

    def run(self):
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
            texted_images_for_model3 = load_timgs(
                self.setting.texted_img_dir,
                self.setting.device,
                mode=LoadingMode.MODEL3_TRAIN,
                max_num=self.setting.num_images,
            )
            
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
                    
                for i, texted_image in enumerate(self.texted_images):
                    texted_image.visualize(self.setting.output_img_dir, f"final_{texted_image}.{i}.png")

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
                    
                for i, texted_image in enumerate(self.texted_images):
                    texted_image.visualize(self.setting.output_img_dir, f"final_{texted_image}.{i}.png")

    def _run_imageloader(self):
        print("[Pipeline] Loading Images")
        if self.setting.timg_generation == TimgGeneration.generate_only:
            print("[Pipeline] Generating TextedImages")
            self.imageloader.start_loading_async(
                num_images=self.setting.num_images,
                dir=self.setting.clear_img_dir,
                max_text_size=self.setting.model3_input_size,
            )
            self.texted_images = self.imageloader.get_loaded_images()
        elif self.setting.timg_generation == TimgGeneration.generate_save:
            print(
                "[Pipeline] Generating TextedImages and Save",
                self.setting.model3_input_size,
            )
            self.imageloader.start_loading_async(
                num_images=self.setting.num_images,
                dir=self.setting.clear_img_dir,
                max_text_size=self.setting.model3_input_size,
            )
            self.texted_images = self.imageloader.get_loaded_images()
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

        ################################################### Step 2: BBox Merge ###############################################
        print(f"[Pipeline] Merging bboxes with margin {self.setting.margin}")
        for texted_image in self.texted_images:
            texted_image.merge_bboxes_with_margin(self.setting.margin)

        # Might need to change device to GPU
        self.setting.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
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
                    collate_fn=None,  # 또는 명시적으로 default_collate 사용 (보통 None이면 알아서 default_collate)
                    persistent_workers=(self.setting.num_workers > 0),
                    pin_memory=True,
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
                        return BBox(
                            int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
                        )

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
                # Create flat list of crops for training
                texted_images_for_model2_flat = [
                    _splitted
                    for texted_image in self.texted_images  # Use potentially updated self.texted_images
                    for _splitted in texted_image.split_margin_crop(self.setting.margin)
                ]
                if not texted_images_for_model2_flat:
                    print(
                        "[Pipeline] No crops generated for Model 2 training. Skipping Model 2 training."
                    )
                else:
                    print("[Pipeline] Training Model 2")
                    model2 = Model2(
                        n_channels=3, n_classes=1, device=self.setting.device
                    )
                    train_dataset = MangaDataset2(
                        texted_images_for_model2_flat,
                        self.setting.model2_input_size,
                    )  # Use the flat list
                    train_loader = DataLoader(
                        train_dataset,
                        batch_size=self.setting.batch_size,
                        shuffle=True,
                        num_workers=self.setting.num_workers,
                    )
                    optimizer = torch.optim.Adam(
                        model2.parameters(), lr=self.setting.lr
                    )
                    criterion = torch.nn.BCEWithLogitsLoss()
                    model2_ckpt_path = os.path.join(self.setting.ckpt_dir, "model2.pth")
                    start_epoch = model2.load_checkpoint(model2_ckpt_path, optimizer)

                    print(
                        f"[Pipeline] Starting Model 2 training from epoch {start_epoch}"
                    )
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

                        if (epoch + 1) % self.setting.vis_interval == 0 and len(
                            texted_images_for_model2_flat
                        ) > 0:
                            print(
                                f"[Pipeline] Visualizing Model 2 output for epoch {epoch+1}"
                            )
                            model2.eval()
                            vis_sample = texted_images_for_model2_flat[0]
                            input_img_tensor_for_vis = vis_sample.timg.unsqueeze(0).to(
                                self.setting.device
                            )
                            gt_mask_tensor_for_vis = vis_sample.mask.cpu()
                            with torch.no_grad():
                                pred_logits = model2(input_img_tensor_for_vis)
                                pred_prob = torch.sigmoid(pred_logits).squeeze(0).cpu()
                                pred_mask_for_vis = (pred_prob > 0.5).float()
                            plt.figure(figsize=(15, 5))
                            plt.subplot(1, 3, 1)
                            plt.imshow(VTF.to_pil_image(vis_sample.timg.cpu()))
                            plt.title(f"Input Crop (Epoch {epoch+1})")
                            plt.axis("off")
                            plt.subplot(1, 3, 2)
                            plt.imshow(
                                VTF.to_pil_image(gt_mask_tensor_for_vis, mode="L"),
                                cmap="gray",
                            )
                            plt.title("Ground Truth Mask")
                            plt.axis("off")
                            plt.subplot(1, 3, 3)
                            plt.imshow(
                                VTF.to_pil_image(pred_mask_for_vis, mode="L"),
                                cmap="gray",
                            )
                            plt.title("Predicted Mask")
                            plt.axis("off")
                            vis_output_filename = f"model2_train_viz.png"
                            plt.savefig(
                                os.path.join(
                                    self.setting.output_img_dir, vis_output_filename
                                )
                            )
                            plt.close()
                            print(
                                f"[Pipeline] Saved Model 2 training visualization to {os.path.join(self.setting.output_img_dir, vis_output_filename)}"
                            )
                            model2.train()

            elif self.setting.model2_mode == ModelMode.INFERENCE:
                print("[Pipeline] Running Model 2 Inference")
                model2 = Model2(n_channels=3, n_classes=1, device=self.setting.device)
                model2_ckpt_path = os.path.join(self.setting.ckpt_dir, "model2.pth")
                model2.load_checkpoint(
                    model2_ckpt_path
                )  # Optimizer not needed for inference
                model2.eval()

                print(
                    f"[Pipeline] Applying Model 2 (Mask Generation) to {len(self.texted_images)} images..."
                )
                for original_idx, original_texted_image in enumerate(
                    tqdm(self.texted_images, desc="Model2 Inference on original images")
                ):
                    # 1. Split into crops
                    list_of_crops = original_texted_image.split_margin_crop(
                        self.setting.margin
                    )

                    if not list_of_crops:
                        # print(f"[Pipeline] No crops for image {original_idx} for Model 2.") # Optional: reduce verbosity
                        continue

                    processed_crops_for_this_original = []
                    for (
                        crop_texted_image
                    ) in list_of_crops:  # Removed crop_idx as it's not used
                        # Model2.forward handles moving to device if input is on CPU
                        # But good practice to ensure it if BaseModel doesn't enforce it on input to forward
                        input_tensor = crop_texted_image.timg.unsqueeze(0).to(
                            self.setting.device
                        )

                        with torch.no_grad():
                            pred_logits = model2(input_tensor)

                        pred_probabilities = torch.sigmoid(pred_logits)
                        pred_binary_mask = (pred_probabilities > 0.5).float()

                        # Update the mask of the crop_texted_image object (can be in-place if it's a copy, or use deepcopy)
                        # The split_margin_crop already returns deepcopies of data if internal tensors are sliced.
                        # Let's assume crop_texted_image from list_of_crops can be modified directly.
                        # If not, a deepcopy(crop_texted_image) would be needed here before modification.
                        # The prompt's code for Model1 inference used deepcopies of texted_images_for_model1,
                        # but here list_of_crops are new objects from split_margin_crop.
                        crop_texted_image.mask = pred_binary_mask.squeeze(
                            0
                        ).cpu()  # Ensure mask is on CPU
                        processed_crops_for_this_original.append(crop_texted_image)

                    original_texted_image.merge_cropped(
                        processed_crops_for_this_original
                    )

                print(
                    "[Pipeline] Model 2 Inference complete. Masks in self.texted_images updated."
                )

                num_viz_samples = min(3, len(self.texted_images))
                if num_viz_samples > 0:
                    print(
                        f"[Pipeline] Visualizing first {num_viz_samples} Model 2 inference results (merged masks)..."
                    )
                for i in range(num_viz_samples):
                    img_to_viz = self.texted_images[i]

                    plt.figure(figsize=(10, 5))  # Adjusted figsize for 2 panels
                    plt.subplot(1, 2, 1)
                    plt.imshow(VTF.to_pil_image(img_to_viz.timg.cpu()))
                    plt.title(f"Input Image (Sample {i})")  # timg shows image with text
                    plt.axis("off")

                    plt.subplot(1, 2, 2)
                    plt.imshow(
                        VTF.to_pil_image(img_to_viz.mask.cpu().squeeze(0)),
                        cmap="gray",
                    )
                    plt.title("Predicted & Merged Mask")
                    plt.axis("off")

                    viz_filename = (
                        f"model2_inference_viz_sample{i}.png"  # Corrected filename
                    )
                    plt.savefig(os.path.join(self.setting.output_img_dir, viz_filename))
                    plt.close()
                    print(
                        f"Saved Model 2 inference visualization to {os.path.join(self.setting.output_img_dir, viz_filename)}"
                    )
        else:
            print("[Pipeline] Skipping Model 2")

        ################################################### Step 6: Model 2 output apply #####################################
        # This step is now effectively handled within the INFERENCE block for Model2,
        # where original_texted_image.merge_cropped() is called.
        if self.setting.model2_mode == ModelMode.INFERENCE:
            print(
                "[Pipeline] Model 2 output (masks) has been applied to self.texted_images by merging crops."
            )
            pass  # Placeholder, can be removed if no other common logic is needed.

        # for i, texted_image in enumerate(self.texted_images):
        #     texted_image.visualize(dir="trit/datas/images/output", filename=f"before_model3_images{i}.png")

        ################################################### Step 7: Model 3 ##################################################
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

            if self.setting.model3_mode == ModelMode.TRAIN:

                # 학습시에만 사용용
                texted_images_for_model3 = [
                    _splitted
                    for texted_image in self.texted_images
                    for _splitted in texted_image.split_center_crop(
                        self.setting.model3_input_size
                    )
                ]

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
                    texted_image.visualize(
                        dir=output_dir, filename=f"before_inpainting{i}.png"
                    )

                # 패치들을 인페인팅
                inpainted_patches = model3.inference(texted_images_for_model3)

                # # 인페인팅된 패치들을 원본 이미지에 다시 합성
                # # 패치들을 원본 이미지별로 그룹화하여 합성
                patch_idx = 0
                for texted_image in self.texted_images:
                    # 현재 이미지의 bbox 개수만큼 패치 가져오기
                    num_bboxes = len(texted_image.bboxes)
                    current_patches = inpainted_patches[
                        patch_idx : patch_idx + num_bboxes
                    ]

                    # 패치들을 원본 이미지에 합성
                    texted_image.merge_cropped(current_patches)

                    patch_idx += num_bboxes

                # 결과 이미지 저장
                output_dir = "trit/datas/images/output"
                os.makedirs(output_dir, exist_ok=True)
                for i, texted_image in enumerate(self.texted_images):
                    texted_image.visualize(
                        dir=output_dir, filename=f"final_inpainted_{i}.png"
                    )

                print(f"[Pipeline] Inpainting completed. Results saved to {output_dir}")

            # 사전 훈련 모델 훈련
            elif self.setting.model3_mode == ModelMode.PRETRAINED_TRAIN:

                # 학습시에만 사용
                texted_images_for_model3 = [
                    _splitted
                    for texted_image in self.texted_images
                    for _splitted in texted_image.split_center_crop(
                        self.setting.model3_input_size
                    )
                ]

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
                    texted_image.visualize(
                        dir=output_dir, filename=f"before_inpainting{i}.png"
                    )

                # 패치들을 인페인팅
                inpainted_patches = model3.inference(texted_images_for_model3)

                # # 인페인팅된 패치들을 원본 이미지에 다시 합성
                # # 패치들을 원본 이미지별로 그룹화하여 합성
                patch_idx = 0
                for texted_image in self.texted_images:
                    # 현재 이미지의 bbox 개수만큼 패치 가져오기
                    num_bboxes = len(texted_image.bboxes)
                    current_patches = inpainted_patches[
                        patch_idx : patch_idx + num_bboxes
                    ]

                    # 패치들을 원본 이미지에 합성
                    texted_image.merge_cropped(current_patches)

                    patch_idx += num_bboxes

                # 결과 이미지 저장
                output_dir = "trit/datas/images/output"
                os.makedirs(output_dir, exist_ok=True)
                for i, texted_image in enumerate(self.texted_images):
                    texted_image.visualize(
                        dir=output_dir, filename=f"final_inpainted_{i}.png"
                    )

                print(f"[Pipeline] Inpainting completed. Results saved to {output_dir}")

        else:
            print("[Pipeline] Skipping Model 3")

        ################################################### Step 8: Model 3 output apply #####################################

        if (
            self.setting.model3_mode == ModelMode.INFERENCE
            or self.setting.model3_mode == ModelMode.PRETRAINED
        ):
            pass
