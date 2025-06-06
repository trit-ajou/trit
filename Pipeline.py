import gc
import glob
import torch
from tqdm import tqdm
from copy import copy, deepcopy
from torch.utils.data import DataLoader, random_split, Dataset
from typing import Optional
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
        self.model2: Optional[Model2] = None
        self.texted_images: list[TextedImage] = []

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

        if self.setting.model1_mode != ModelMode.SKIP:
            print("[Pipeline] Processing Model 1...")
            if self.setting.model1_mode == ModelMode.TRAIN:
                print("[Pipeline] Training Model 1")
            elif self.setting.model1_mode == ModelMode.INFERENCE:
                print("[Pipeline] Running Model 1 Inference")
        else:
            print("[Pipeline] Skipping Model 1")

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

        if self.setting.model3_mode != ModelMode.SKIP:
            print("[Pipeline] Processing Model 3...")
            texted_images_for_model3 = []
            for texted_image in self.texted_images:
                texted_images_for_model3.extend(texted_image.split_center_crop(self.setting.model3_input_size))
            if self.setting.model3_mode == ModelMode.TRAIN:
                print("[Pipeline] Training Model 3")
            elif self.setting.model3_mode == ModelMode.INFERENCE:
                print("[Pipeline] Running Model 3 Inference")
        else:
            print("[Pipeline] Skipping Model 3")

        self.imageloader.shutdown()
        print("[Pipeline] Run completed.")

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
