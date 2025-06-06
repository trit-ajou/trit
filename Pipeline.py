import glob
import torch
from tqdm import tqdm
from copy import copy
from torch.utils.data import DataLoader, random_split
import time
import os

from .datas.ImageLoader import ImageLoader
from .datas.TextedImage import TextedImage

from .datas.Dataset import MangaDataset2
from .models.Utils import ModelMode
from .models.Model2 import Model2
from .Utils import PipelineSetting, ImagePolicy


class PipelineMgr:
    def __init__(self, setting: PipelineSetting, policy: ImagePolicy):
        self.setting = setting
        self.imageloader = ImageLoader(setting, policy)

        # Initialize models
        self.model2 = None

    def run(self):
        ################################################### Step 1: Load Images ##############################################
        print("[Pipeline] Loading Images")
        # 이미지로더 사용 방법 예시(NEW)
        self.imageloader.start_loading_async(
            num_images=self.setting.num_images,
            dir=self.setting.clear_img_dir,
            max_text_size=self.setting.model3_input_size,
        )
        # 할일 하기
        time.sleep(5)
        # 로딩된 이미지 불러오기(덜끝났으면 끝날 때까지 대기)
        self.texted_images = self.imageloader.get_loaded_images()
        # 프로그램 종료 시
        self.imageloader.shutdown()

        ################################################### Step 2: BBox Merge ###############################################
        print(f"[Pipeline] Merging bboxes with margin {self.setting.margin}")
        for texted_image in self.texted_images:
            texted_image.merge_bboxes_with_margin(self.setting.margin)

        # Might need to change device to GPU
        self.setting.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Pipeline] Using Device: {self.setting.device}")
        ################################################### Step 3: Model 1 ##################################################
        if self.setting.model1_mode != ModelMode.SKIP:
            texted_images_for_model1 = [copy(texted_image) for texted_image in self.texted_images]
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
        texted_images_for_model2 = []
        if self.setting.model2_mode != ModelMode.SKIP:
            # Create split images and filter out invalid ones
            for idx, texted_image in enumerate(self.texted_images):
                try:
                    splitted_images = texted_image.split_margin_crop(self.setting.margin)
                    texted_images_for_model2.extend(splitted_images)

                except Exception as e:
                    print(f"[Warning] Error processing image: {e}")
                    continue

                # Visualize original image
                # texted_image.visualize(dir=self.setting.output_img_dir, filename=f"model2_original_{idx}")

            # Resize images to model input size and visualize
            print(f"[Pipeline] Resizing {len(texted_images_for_model2)} images to {self.setting.model2_input_size}")
            resized_images = []
            for i, texted_image in enumerate(texted_images_for_model2):
                try:
                    # Print original dimensions
                    _, orig_H, orig_W = texted_image.timg.shape
                    # print(f"[Debug] Image {i}: Original size ({orig_H}, {orig_W}) -> Target size {self.setting.model2_input_size}")

                    texted_image._resize(self.setting.model2_input_size)
                    # texted_image.visualize(dir=self.setting.output_img_dir, filename=f"model2_resized_{i}")

                    # Verify resize result
                    _, new_H, new_W = texted_image.timg.shape
                    # print(f"[Debug] Image {i}: Resized to ({new_H}, {new_W})")

                    resized_images.append(texted_image)

                    # Visualize resized images (every 10th image to avoid too many files)
                    # if i % 10 == 0:
                    #     texted_image.visualize(dir=self.setting.output_img_dir, filename=f"model2_resized_{i}")

                except ValueError as e:
                    print(f"[Warning] Skipping image {i} due to resize error: {e}")
                    continue

            texted_images_for_model2 = resized_images
            print(f"[Pipeline] Successfully resized {len(texted_images_for_model2)} images")

            if len(texted_images_for_model2) == 0:
                print("[Error] No valid images for Model2 processing after resize")
                return

            if self.setting.model2_mode == ModelMode.TRAIN:
                print("[Pipeline] Training Model 2")
                self.train_model2(texted_images_for_model2)
        else:
            print("[Pipeline] Skipping Model 2")

        ################################################### Step 6: Model 2 output apply #####################################
        if self.setting.model2_mode == ModelMode.INFERENCE:
            print("[Pipeline] Applying Model 2 results")
            # self.setting.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model2 = Model2(num_classes=2, pretrained=True).to(self.setting.device)
            self.inference_model2(texted_images_for_model2)

        ################################################### Step 7: Model 3 ##################################################
        if self.setting.model3_mode != ModelMode.SKIP:
            texted_images_for_model3 = [
                _splitted
                for texted_image in self.texted_images
                for _splitted in texted_image.split_center_crop(self.setting.model3_input_size)
            ]
            if self.setting.model3_mode == ModelMode.TRAIN:
                print("[Pipeline] Training Model 3")
                # TODO: model 3 train, viz
            elif self.setting.model3_mode == ModelMode.INFERENCE:
                print("[Pipeline] Running Model 3 Inference")
                # TODO: model 3 inference, viz, apply
        else:
            print("[Pipeline] Skipping Model 3")

        ################################################### Step 8: Model 3 output apply #####################################
        if self.setting.model3_mode == ModelMode.INFERENCE:
            pass

    def train_model2(self, texted_images_for_model2: list[TextedImage]):
        """Train Model2 for pixel-wise text segmentation"""
        # Images are already resized and validated, no need for additional checks
        print(f"[Model2 Train] Received {len(texted_images_for_model2)} pre-processed images")

        # Create dataset
        dataset = MangaDataset2(texted_images_for_model2, transform=True)

        # Split dataset
        train_size = int((1 - self.setting.train_valid_split) * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.setting.batch_size,
            shuffle=True,
            num_workers=self.setting.num_workers,
            drop_last=True,  # Drop last incomplete batch to avoid BatchNorm issues
            pin_memory=True if self.setting.device.type == "cuda" else False,  # GPU 사용시 pin_memory 활성화
            persistent_workers=True if self.setting.num_workers > 0 else False,  # 워커 재사용
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.setting.batch_size,
            shuffle=False,
            num_workers=self.setting.num_workers,
            drop_last=True,  # Drop last incomplete batch to avoid BatchNorm issues
            pin_memory=True if self.setting.device.type == "cuda" else False,  # GPU 사용시 pin_memory 활성화
            persistent_workers=True if self.setting.num_workers > 0 else False,  # 워커 재사용
        )

        # self.setting.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model2 = Model2(num_classes=2, pretrained=True).to(self.setting.device)

        # Setup training
        optimizer = torch.optim.Adam(
            self.model2.parameters(), lr=self.setting.lr, weight_decay=self.setting.weight_decay
        )
        criterion = torch.nn.BCELoss()
        scaler = torch.amp.GradScaler("cuda") if self.setting.use_amp else None

        # First try to load final model
        final_checkpoint = f"{self.setting.ckpt_dir}/model2_final.pth"
        if os.path.exists(final_checkpoint):
            checkpoint = torch.load(final_checkpoint, map_location=self.setting.device)
            self.model2.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint["epoch"] + 1  # Continue from next epoch
            print(f"[Model2] Resuming training from {final_checkpoint}, starting at epoch {start_epoch}")
        else:
            # Try to load latest epoch checkpoint
            checkpoint_pattern = f"{self.setting.ckpt_dir}/model2_epoch_*.pth"
            checkpoint_files = glob.glob(checkpoint_pattern)

            if checkpoint_files:
                # Get the latest checkpoint by epoch number
                latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split("_")[-1].split(".")[0]))
                checkpoint = torch.load(latest_checkpoint, map_location=self.setting.device)
                self.model2.load_state_dict(checkpoint["model_state_dict"])
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                start_epoch = checkpoint["epoch"] + 1  # Continue from next epoch
                print(f"[Model2] Resuming training from {latest_checkpoint}, starting at epoch {start_epoch}")
            else:
                print("[Model2] No checkpoint found, starting fresh training")

        # Training loop
        for epoch in range(self.setting.epochs):
            # Training phase
            self.model2.train()
            train_loss = 0.0

            for batch_idx, (timg, mask) in enumerate(
                tqdm(train_loader, desc=f"Epoch {epoch + 1}/{self.setting.epochs}")
            ):
                timg = timg.to(self.setting.device)
                mask = mask.to(self.setting.device)

                # Debug: print shapes
                # if batch_idx == 0:
                #     print(f"[Debug] timg shape: {timg.shape}, mask shape: {mask.shape}")

                # Skip batch if size is 1 to avoid BatchNorm issues
                if timg.size(0) == 1:
                    print("[Warning] Skipping batch of size 1 to avoid BatchNorm issues")
                    continue

                optimizer.zero_grad()

                if self.setting.use_amp and scaler:
                    with torch.amp.autocast("cuda"):
                        output = self.model2(timg)
                        if isinstance(output, dict):
                            # Ensure mask has the same dimensions as output
                            if mask.dim() == 2:
                                mask = mask.unsqueeze(1)  # (B, H, W) -> (B, 1, H, W)
                            elif mask.dim() == 3 and mask.shape[1] != 1:
                                mask = mask.unsqueeze(1)  # (B, H, W) -> (B, 1, H, W)
                            print(
                                f"[Debug] After adjustment - output shape: {output['out'].shape}, mask shape: {mask.shape}"
                            )
                            main_loss = criterion(output["out"][:, 1:2], mask)
                            aux_loss = criterion(output["aux"][:, 1:2], mask) if "aux" in output else 0
                            loss = main_loss + 0.4 * aux_loss
                        else:
                            if mask.dim() == 2:
                                mask = mask.unsqueeze(1)
                            elif mask.dim() == 3 and mask.shape[1] != 1:
                                mask = mask.unsqueeze(1)
                            loss = criterion(output, mask)

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    output = self.model2(timg)
                    if isinstance(output, dict):
                        # Ensure mask has the same dimensions as output
                        if mask.dim() == 2:
                            mask = mask.unsqueeze(1)  # (B, H, W) -> (B, 1, H, W)
                        elif mask.dim() == 3 and mask.shape[1] != 1:
                            mask = mask.unsqueeze(1)  # (B, H, W) -> (B, 1, H, W)
                        main_loss = criterion(output["out"][:, 1:2], mask)
                        aux_loss = criterion(output["aux"][:, 1:2], mask) if "aux" in output else 0
                        loss = main_loss + 0.4 * aux_loss
                    else:
                        if mask.dim() == 2:
                            mask = mask.unsqueeze(1)
                        elif mask.dim() == 3 and mask.shape[1] != 1:
                            mask = mask.unsqueeze(1)
                        loss = criterion(output, mask)

                    loss.backward()
                    optimizer.step()

                train_loss += loss.item()

            # Validation phase
            if epoch % self.setting.vis_interval == 0 and len(val_loader) > 0:
                self.model2.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for timg, mask in val_loader:
                        timg = timg.to(self.setting.device)
                        mask = mask.to(self.setting.device)

                        output = self.model2(timg)
                        if isinstance(output, dict):
                            # Ensure mask has the same dimensions as output
                            if mask.dim() == 2:
                                mask = mask.unsqueeze(1)  # (B, H, W) -> (B, 1, H, W)
                            elif mask.dim() == 3 and mask.shape[1] != 1:
                                mask = mask.unsqueeze(1)  # (B, H, W) -> (B, 1, H, W)
                            main_loss = criterion(output["out"][:, 1:2], mask)
                            aux_loss = criterion(output["aux"][:, 1:2], mask) if "aux" in output else 0
                            loss = main_loss + 0.4 * aux_loss
                        else:
                            if mask.dim() == 2:
                                mask = mask.unsqueeze(1)
                            elif mask.dim() == 3 and mask.shape[1] != 1:
                                mask = mask.unsqueeze(1)
                            loss = criterion(output, mask)

                        val_loss += loss.item()

                print(
                    f"Epoch {epoch + 1}: Train Loss: {train_loss / len(train_loader):.4f}, Val Loss: {val_loss / len(val_loader):.4f}"
                )

            # Save checkpoint
            if epoch % self.setting.ckpt_interval == 0:
                os.makedirs(self.setting.ckpt_dir, exist_ok=True)
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model2.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": train_loss / len(train_loader),
                    },
                    f"{self.setting.ckpt_dir}/model2_epoch_{epoch}.pth",
                )

        # Save final model
        os.makedirs(self.setting.ckpt_dir, exist_ok=True)
        torch.save(
            {
                "epoch": self.setting.epochs - 1,
                "model_state_dict": self.model2.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": train_loss / len(train_loader),
            },
            f"{self.setting.ckpt_dir}/model2_final.pth",
        )
        print(f"[Model2] Training completed. Final model saved to {self.setting.ckpt_dir}/model2_final.pth")

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
