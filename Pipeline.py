import torch
from tqdm import tqdm
from copy import deepcopy  # Changed from copy to deepcopy for TextedImage objects
from torch.utils.data import DataLoader, random_split
import os  # Added for path joining and directory creation
import torchvision.transforms.functional as VTF  # Added for visualization
from PIL import ImageDraw  # Added for visualization
from .models.model1_util.test import test_net
from .datas.ImageLoader import ImageLoader
from .datas.TextedImage import TextedImage, save_timgs, load_timgs
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

# Collate function for Model1 DataLoader
def detection_collate_fn(batch):
    images = []
    targets_list = []
    for item in batch:
        images.append(item[0])  # item[0] is the image tensor

        # item[1] is the list of BBox tuples from MangaDataset1
        bboxes_tensor = torch.tensor(
            [list(bbox) for bbox in item[1]], dtype=torch.float32
        )
        num_boxes = bboxes_tensor.shape[0]

        # Ensure that even if there are no boxes, the tensors are correctly shaped.
        if num_boxes == 0:
            # Faster R-CNN expects boxes to be [N, 4] and labels [N]
            # If no boxes, provide empty tensors of the correct shape.
            bboxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)
        else:
            labels_tensor = torch.ones(
                (num_boxes,), dtype=torch.int64
            )  # All are 'text' class (id 1)

        targets_list.append({"boxes": bboxes_tensor, "labels": labels_tensor})

    return images, targets_list


class PipelineMgr:
    def __init__(self, setting: PipelineSetting, policy: ImagePolicy):
        self.setting = setting
        self.imageloader = ImageLoader(setting, policy)
        # Ensure output_img_dir and ckpt_dir exist
        os.makedirs(self.setting.output_img_dir, exist_ok=True)
        os.makedirs(self.setting.ckpt_dir, exist_ok=True)

        self.texted_images = None

    def run(self):
        ################################################### Step 1: Load Images ##############################################
        if self.setting.model1_mode == ModelMode.TRAIN: # 모델1 학습 시에만 GT 생성
            should_generate_craft_gt_for_step1 = True
        else: # 추론 또는 다른 모델 사용 시에는 GT 생성 안 함 (기존 동작)
            should_generate_craft_gt_for_step1 = False

        if self.setting.timg_generation == TimgGeneration.generate_only:
            print("[Pipeline] Generating TextedImages")
            self.texted_images = self.imageloader.load_images(
                self.setting.num_images, self.setting.clear_img_dir, should_generate_craft_gt_for_step1)

        elif self.setting.timg_generation == TimgGeneration.generate_save:
            print("[Pipeline] Generating TextedImages and Save", should_generate_craft_gt_for_step1)
            self.texted_images = self.imageloader.load_images(
                self.setting.num_images, self.setting.clear_img_dir)
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
        # print(f"[Pipeline] Merging bboxes with margin {self.setting.margin}")
        # for texted_image in self.texted_images:
        #     texted_image.merge_bboxes_with_margin(self.setting.margin)
        if hasattr(self.texted_images[0], 'merge_bboxes_with_margin') and callable(getattr(self.texted_images[0], 'merge_bboxes_with_margin')):
            print(f"[Pipeline] Merging bboxes with margin {self.setting.margin}")
            for texted_image in self.texted_images:
                texted_image.merge_bboxes_with_margin(self.setting.margin)
        else:
            print("[Pipeline] Skipping BBox Merge (method not found in TextedImage)")
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
                train_dataset = MangaDataset1(
                    texted_images_for_model1,
                    # ImageLoader에서 이미지를 생성하므로, Dataset에서 transforms는 주로 정규화 등
                    # torchvision.transforms.Compose([...]) 등으로 전달 가능
                    # CRAFT 모델은 입력 이미지를 특정 방식으로 정규화해야 할 수 있음
                    # (예: ImageNet 평균/표준편차) - 이 부분은 모델 정의 또는 학습 스크립트에서 명시되어야 함.
                    # 여기서는 ImageLoader가 이미 올바른 Tensor를 생성했다고 가정.
                    generate_craft_gt=True  # Dataset이 GT를 반환하도록 명시 (선택적 파라미터로 추가 가능)
                )
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
                    model1.train()
                    epoch_loss_sum = 0.0

                    batch_iterator = tqdm(
                        train_loader,
                        desc=f"Epoch {epoch+1}/{self.setting.epochs} - Model1 Train",
                        # desc=f"Epoch {epoch + 1}/{3} - Model1 Train",
                        leave=False,
                    )
                    for images, region_targets, affinity_targets in batch_iterator:  # 언패킹 수정
                        images = images.to(model1.device)  # 필요시 device로 이동
                        region_targets = region_targets.to(model1.device)
                        affinity_targets = affinity_targets.to(model1.device)

                        optimizer.zero_grad()
                        tqdm.write(f"images shape:{tuple(images.shape) }")
                        # CRAFT 모델의 forward는 (B, 2, H/2, W/2) 형태의 맵과 feature를 반환 가정



                        output_maps, _ = model1(images)  # model1.forward(self, x) -> y, feature
                        tqdm.write(f"output_maps shape: {tuple(output_maps.shape)}")
                        # 모델 출력에서 region/affinity map 분리
                        # y의 형태가 (B, H_out, W_out, 2) 였다면:
                        # pred_region_map_batch = output_maps[..., 0].permute(0,3,1,2) # (B,1,H,W) 필요시
                        # pred_affinity_map_batch = output_maps[..., 1].permute(0,3,1,2)
                        # 현재 모델 출력이 (B, 2, H/2, W/2)로 가정:
                        # output_maps: (B, H, W, 2)
                        output_maps = output_maps.permute(0, 3, 1, 2)  # → (B, 2, H, W)

                        pred_region_map_batch = output_maps[:, 0:1]  # (B, 1, H, W)
                        pred_affinity_map_batch = output_maps[:, 1:2]  # (B, 1, H, W)

                        # 1) 학습 루프에서 손실 정의
                        pos = region_targets.sum()
                        neg = region_targets.numel() - pos
                        w = neg / (pos + 1e-6)  # pos_weight

                        loss_r = F.binary_cross_entropy_with_logits(
                            pred_region_map_batch, region_targets,
                            pos_weight=torch.tensor([w], device=model1.device))

                        loss_a = F.binary_cross_entropy_with_logits(
                            pred_affinity_map_batch, affinity_targets,
                            pos_weight=torch.tensor([w], device=model1.device))
                        total_batch_loss = loss_r + loss_a
                        tqdm.write(
                            f"pred region min/max: {pred_region_map_batch.min():.3f} / {pred_region_map_batch.max():.3f}")
                        tqdm.write(f"GT   region min/max: {region_targets.min():.3f} / {region_targets.max():.3f}")

                        total_batch_loss.backward()
                        optimizer.step()

                        epoch_loss_sum += total_batch_loss.item()
                        batch_iterator.set_postfix(loss=f"{total_batch_loss.item():.4f}")

                    avg_epoch_loss = epoch_loss_sum / len(train_loader)
                    print(
                        f"[Pipeline] Epoch {epoch+1}/{self.setting.epochs} - Model 1 Average Training Loss: {avg_epoch_loss:.4f}"
                    )

                    if (epoch + 1) % self.setting.ckpt_interval == 0:
                        model1.save_checkpoint(
                            model1_ckpt_path, epoch, optimizer.state_dict()
                        )

                    if (epoch + 1) % self.setting.vis_interval == 0 and len(texted_images_for_model1) > 0:
                        print(f"[Pipeline] Visualizing Model 1 (CRAFT) output for epoch {epoch + 1}")
                        model1.eval()

                        # 시각화할 샘플 가져오기 (MangaDataset1에서 직접)
                        # Dataset의 __getitem__은 정규화된 이미지와 GT맵을 반환.
                        # 시각화를 위해서는 정규화 전 이미지와 GT맵이 필요.
                        # TextedImage 객체를 직접 사용하는 것이 좋음.
                        vis_idx = 0  # 첫 번째 샘플 시각화
                        vis_texted_image_obj = texted_images_for_model1[vis_idx]

                        # 모델 입력 준비 (정규화 필요시 적용)
                        img_for_pred_tensor = vis_texted_image_obj.timg.unsqueeze(0).to(model1.device)
                        # if train_dataset.transforms: # Dataset에 transform이 있다면 동일하게 적용
                        #    img_for_pred_tensor = train_dataset.transforms(img_for_pred_tensor)

                        with torch.no_grad():
                            # 모델 예측 (튜플 반환: (score_maps_permuted, features) 또는 (score_maps_B2HW, features) )
                            pred_maps_model_output, _ = model1(img_for_pred_tensor)

                            # 모델 출력 형식에 따라 pred_region_vis, pred_affinity_vis 추출
                            # 예: pred_maps_model_output가 (1, 2, H/2, W/2) 라면
                            maps_CHW = pred_maps_model_output.permute(0, 3, 1, 2)  # -> (B, 2, H, W)
                            pred_region_vis = torch.sigmoid(maps_CHW[0, 0]).cpu().numpy()
                            pred_affinity_vis = torch.sigmoid(maps_CHW[0, 1]).cpu().numpy()
                            tqdm.write(f"pred_region_vis shape:{tuple(pred_region_vis.shape)}")
                            tqdm.write(f"pred_affinity_vis shape:{tuple(pred_affinity_vis.shape)}")
                            # 예: pred_maps_model_output가 (1, H/2, W/2, 2) 라면
                            # pred_region_vis = pred_maps_model_output[0, :, :, 0].cpu().numpy()
                            # pred_affinity_vis = pred_maps_model_output[0, :, :, 1].cpu().numpy()

                        # 시각화 함수 호출 (이전 답변에서 제안한 visualize_texted_image_data 사용)
                        # 이 함수는 TextedImage 객체를 받으므로, 예측값을 여기에 임시로 넣어주거나,
                        # 별도의 시각화 함수를 만들어 GT와 Prediction을 함께 그림.

                        # visualize_texted_image_data 함수를 확장하여 예측값도 함께 표시하도록 수정 필요.
                        # 또는, 간단히 GT와 Prediction을 나란히 표시.

                        # 임시 TextedImage 객체 생성 (시각화용)
                        # GT 맵은 texted_images_for_model1[vis_idx]에 이미 있어야 함.
                        gt_region_map_vis = vis_texted_image_obj.region_score_map.squeeze().cpu().numpy() \
                            if vis_texted_image_obj.region_score_map is not None else None
                        gt_affinity_map_vis = vis_texted_image_obj.affinity_score_map.squeeze().cpu().numpy() \
                            if vis_texted_image_obj.affinity_score_map is not None else None

                        # Matplotlib으로 GT와 Prediction 나란히 그리기
                        num_cols_vis = 2  # GT Region, Pred Region (Affinity도 추가 가능)
                        if gt_affinity_map_vis is not None and pred_affinity_vis is not None:
                            num_cols_vis += 2

                        fig_vis, axes_vis = plt.subplots(1, num_cols_vis, figsize=(5 * num_cols_vis, 5))
                        fig_vis.suptitle(f"Epoch {epoch + 1} - Sample {vis_idx}", fontsize=16)

                        current_ax_idx = 0
                        if gt_region_map_vis is not None:
                            axes_vis[current_ax_idx].imshow(gt_region_map_vis, cmap='jet', vmin=0, vmax=1)
                            axes_vis[current_ax_idx].set_title("GT Region")
                            axes_vis[current_ax_idx].axis('off');
                            current_ax_idx += 1

                        axes_vis[current_ax_idx].imshow(pred_region_vis, cmap='jet', vmin=0, vmax=1)
                        axes_vis[current_ax_idx].set_title("Pred Region")
                        axes_vis[current_ax_idx].axis('off');
                        current_ax_idx += 1

                        if gt_affinity_map_vis is not None:
                            axes_vis[current_ax_idx].imshow(gt_affinity_map_vis, cmap='jet', vmin=0, vmax=1)
                            axes_vis[current_ax_idx].set_title("GT Affinity")
                            axes_vis[current_ax_idx].axis('off');
                            current_ax_idx += 1

                        if pred_affinity_vis is not None:
                            axes_vis[current_ax_idx].imshow(pred_affinity_vis, cmap='jet', vmin=0, vmax=1)
                            axes_vis[current_ax_idx].set_title("Pred Affinity")
                            axes_vis[current_ax_idx].axis('off');
                            current_ax_idx += 1

                        # 원본 이미지 및 문자 폴리곤도 함께 표시하면 좋음 (이전 visualize_texted_image_data 참고)
                        # (예: 첫 번째 행에 이미지, 두 번째 행에 스코어 맵)

                        vis_output_filename_train = f"model1_train_epoch{epoch + 1}_sample{vis_idx}_pred_gt.png"
                        save_path_train_vis = os.path.join(self.setting.output_img_dir, vis_output_filename_train)
                        plt.tight_layout(rect=[0, 0, 1, 0.96])  # suptitle 공간 확보
                        plt.savefig(save_path_train_vis)
                        plt.close(fig_vis)
                        print(f"[Pipeline] Saved Model 1 training visualization to {save_path_train_vis}")

                        model1.train()  # Set back to train mode
                    # -------- 수정 끝 (학습 중 시각화 부분) --------

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

        ################################################### Step 7: Model 3 ##################################################
        if self.setting.model3_mode != ModelMode.SKIP:
            if self.setting.model3_mode == ModelMode.TRAIN:
                # Create flat list of crops for Model3 training
                texted_images_for_model3_train = [
                    _splitted
                    for texted_image in self.texted_images  # Use potentially updated self.texted_images
                    for _splitted in texted_image.split_center_crop(
                        self.setting.model3_input_size
                    )
                ]
                print("[Pipeline] Training Model 3")
                if not hasattr(self.setting, "model3_input_size"):
                    print(
                        "[Pipeline] CRITICAL WARNING: self.setting.model3_input_size is not defined. This is essential for Model3 data preparation. Training may fail or be incorrect."
                    )

                if not texted_images_for_model3_train:  # Check the correct list
                    print(
                        "[Pipeline] No crops generated for Model 3 training (texted_images_for_model3_train is empty). Skipping Model 3 training."
                    )
                else:
                    model3 = Model3(
                        img_channels=3,
                        mask_channels=1,
                        output_channels=3,
                        device=self.setting.device,
                    )
                    # Use the training-specific list
                    train_dataset = MangaDataset3(
                        texted_images_for_model3_train,
                        self.setting.model2_input_size,
                    )
                    train_loader = DataLoader(
                        train_dataset,
                        batch_size=self.setting.batch_size,
                        shuffle=True,
                        num_workers=self.setting.num_workers,
                    )
                    optimizer = torch.optim.Adam(
                        model3.parameters(), lr=self.setting.lr
                    )
                    criterion = torch.nn.L1Loss()
                    model3_ckpt_path = os.path.join(self.setting.ckpt_dir, "model3.pth")
                    start_epoch = model3.load_checkpoint(model3_ckpt_path, optimizer)

                    print(
                        f"[Pipeline] Starting Model 3 training from epoch {start_epoch}"
                    )
                    for epoch in range(start_epoch, self.setting.epochs):
                        model3.train()
                        epoch_loss = 0.0
                        batch_iterator = tqdm(
                            train_loader,
                            desc=f"Epoch {epoch+1}/{self.setting.epochs} - Model3 Train",
                            leave=False,
                        )
                        for input_dict, target_original_images in batch_iterator:
                            images_with_text = input_dict["image_with_text"].to(
                                self.setting.device
                            )
                            masks = input_dict["mask"].to(self.setting.device)
                            target_original_images = target_original_images.to(
                                self.setting.device
                            )
                            pred_inpainted_images = model3(images_with_text, masks)
                            loss = criterion(
                                pred_inpainted_images, target_original_images
                            )
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()
                            epoch_loss += loss.item()
                            batch_iterator.set_postfix(loss=f"{loss.item():.4f}")
                        avg_epoch_loss = epoch_loss / len(train_loader)
                        print(
                            f"[Pipeline] Epoch {epoch+1}/{self.setting.epochs} - Model 3 Average Training Loss: {avg_epoch_loss:.4f}"
                        )

                        if (epoch + 1) % self.setting.ckpt_interval == 0:
                            model3.save_checkpoint(
                                model3_ckpt_path, epoch, optimizer.state_dict()
                            )

                        # Use the training-specific list for visualization sample
                        if (epoch + 1) % self.setting.vis_interval == 0 and len(
                            texted_images_for_model3_train
                        ) > 0:
                            print(
                                f"[Pipeline] Visualizing Model 3 output for epoch {epoch+1}"
                            )
                            model3.eval()
                            vis_sample_crop = texted_images_for_model3_train[0]
                            vis_img_texted = vis_sample_crop.timg.unsqueeze(0).to(
                                self.setting.device
                            )
                            vis_mask = vis_sample_crop.mask.unsqueeze(0).to(
                                self.setting.device
                            )
                            vis_gt_orig_crop = vis_sample_crop.orig.cpu()
                            with torch.no_grad():
                                pred_inpainted_vis = (
                                    model3(vis_img_texted, vis_mask).squeeze(0).cpu()
                                )
                            plt.figure(figsize=(20, 5))
                            plt.subplot(1, 4, 1)
                            plt.imshow(VTF.to_pil_image(vis_sample_crop.timg.cpu()))
                            plt.title(f"Input Texted (Epoch {epoch+1})")
                            plt.axis("off")
                            plt.subplot(1, 4, 2)
                            plt.imshow(
                                VTF.to_pil_image(
                                    vis_sample_crop.mask.cpu().squeeze(0), mode="L"
                                ),
                                cmap="gray",
                            )
                            plt.title("Input Mask")
                            plt.axis("off")
                            plt.subplot(1, 4, 3)
                            plt.imshow(VTF.to_pil_image(pred_inpainted_vis))
                            plt.title("Predicted Inpainted")
                            plt.axis("off")
                            plt.subplot(1, 4, 4)
                            plt.imshow(VTF.to_pil_image(vis_gt_orig_crop))
                            plt.title("Ground Truth Original")
                            plt.axis("off")
                            vis_output_filename = f"model3_train_viz.png"
                            plt.savefig(
                                os.path.join(
                                    self.setting.output_img_dir, vis_output_filename
                                )
                            )
                            plt.close()
                            print(
                                f"[Pipeline] Saved Model 3 training visualization to {os.path.join(self.setting.output_img_dir, vis_output_filename)}"
                            )
                            model3.train()

            elif self.setting.model3_mode == ModelMode.INFERENCE:
                print("[Pipeline] Running Model 3 Inference")
                model3 = Model3(
                    img_channels=3,
                    mask_channels=1,
                    output_channels=3,
                    device=self.setting.device,
                )
                model3_ckpt_path = os.path.join(self.setting.ckpt_dir, "model3.pth")
                model3.load_checkpoint(
                    model3_ckpt_path
                )  # Optimizer not needed for inference
                model3.eval()

                print(
                    f"[Pipeline] Applying Model 3 (Inpainting) to {len(self.texted_images)} images..."
                )
                for original_idx, original_texted_image in enumerate(
                    tqdm(self.texted_images, desc="Model3 Inference on original images")
                ):
                    # 1. Split into crops using center_crop
                    list_of_crops = original_texted_image.split_center_crop(
                        self.setting.model3_input_size
                    )

                    if not list_of_crops:
                        # print(f"[Pipeline] No crops for image {original_idx} for Model 3.") # Reduce verbosity
                        continue

                    processed_crops_for_this_original = []
                    for (
                        crop_texted_image
                    ) in list_of_crops:  # crop_idx removed as not used
                        crop_img_w_text = crop_texted_image.timg
                        crop_mask = crop_texted_image.mask

                        with torch.no_grad():
                            # Model3.forward handles device movement for its inputs.
                            # Ensure inputs have batch dimension.
                            pred_inpainted_tensor = model3(
                                crop_img_w_text.unsqueeze(0), crop_mask.unsqueeze(0)
                            )

                        # Update the 'orig' attribute, ensuring it's on CPU.
                        crop_texted_image.orig = pred_inpainted_tensor.squeeze(0).cpu()
                        processed_crops_for_this_original.append(crop_texted_image)

                    original_texted_image.merge_cropped(
                        processed_crops_for_this_original
                    )

                print(
                    "[Pipeline] Model 3 Inference complete. 'orig' attributes in self.texted_images updated."
                )

                num_viz_samples = min(3, len(self.texted_images))
                if num_viz_samples > 0:
                    print(
                        f"[Pipeline] Visualizing first {num_viz_samples} Model 3 inference results (inpainted)..."
                    )
                for i in range(num_viz_samples):
                    img_to_viz = self.texted_images[i]

                    plt.figure(figsize=(15, 5))
                    plt.subplot(1, 3, 1)
                    plt.imshow(VTF.to_pil_image(img_to_viz.timg.cpu()))
                    plt.title(f"Input Texted (Sample {i})")
                    plt.axis("off")

                    plt.subplot(1, 3, 2)
                    plt.imshow(
                        VTF.to_pil_image(img_to_viz.mask.cpu().squeeze(0), mode="L"),
                        cmap="gray",
                    )
                    plt.title("Input Mask (from M2)")
                    plt.axis("off")

                    plt.subplot(1, 3, 3)
                    plt.imshow(
                        VTF.to_pil_image(img_to_viz.orig.cpu())
                    )  # Inpainted result is in .orig
                    plt.title("Output Inpainted (M3)")
                    plt.axis("off")

                    viz_filename = f"model3_inference_viz_sample{i}.png"
                    plt.savefig(os.path.join(self.setting.output_img_dir, viz_filename))
                    plt.close()
                    print(
                        f"Saved Model 3 inference visualization to {os.path.join(self.setting.output_img_dir, viz_filename)}"
                    )
        else:
            print("[Pipeline] Skipping Model 3")

        ################################################### Step 8: Model 3 output apply #####################################
        # This step is now effectively handled within the INFERENCE block for Model3,
        # where original_texted_image.merge_cropped() is called and updates original_texted_image.orig.
        if self.setting.model3_mode == ModelMode.INFERENCE:
            print(
                "[Pipeline] Model 3 output (inpainted images) has been applied to self.texted_images' 'orig' attribute."
            )
            pass
