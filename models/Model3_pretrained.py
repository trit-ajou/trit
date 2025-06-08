import torch
import torch.nn as nn
import os
import numpy as np
import gc
from torch import autocast
from ..datas.Dataset import MangaDataset3
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
import torch.nn.functional as F
from peft import get_peft_model, LoraConfig, PeftModel
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
from transformers.optimization import Adafactor
from torchvision import transforms
from tqdm import tqdm

from ..datas.TextedImage import TextedImage

def identity_collate(batch_images):
    return batch_images

# 부동소수점 행렬 곱셈 정밀도 설정
torch.set_float32_matmul_precision('high')
print("부동소수점 행렬 곱셈 정밀도를 'high'로 설정")
class Model3_pretrained(nn.Module):
    def __init__(self, model_config: dict, device: str = "cuda"):
        super().__init__()
        self.model_config = model_config
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

    def lora_train(self, texted_images_for_model3: list[TextedImage]):
        print("Loading SD2 pipeline for LoRA training...")

        # 학습은 fp16으로 설정
        weight_dtype = torch.float16
        model_id = self.model_config["model_id"]
        lora_weights_path = self.model_config["lora_path"]
        num_epochs = self.model_config.get("epochs", 10)
        batch_size = self.model_config.get("batch_size", 4)
        # 텍스트 인코딩
        prompt = self.model_config.get("prompts", "pure black and white manga style image with no color tint, absolute grayscale, contextual manga style, remove lettering, remove text, remove logo, remove watermark, consistent with surrounding")
        negative_prompt = self.model_config.get("negative_prompt", "photo, realistic, color, colorful, purple, violet, sepia, any color tint, blurry")        

        # 파이프라인 로딩
        try:
            pipe = StableDiffusionInpaintPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float32,
            )
            pipe.to(self.device)
            print(f"SD2 inpainting pipeline loaded successfully to {self.device}.")
            
            # 필요한 부분 로드 (SD2는 단일 텍스트 인코더와 UNet 사용)
            vae = pipe.vae
            text_encoder = pipe.text_encoder
            unet = pipe.unet
            tokenizer = pipe.tokenizer
            print("SD2 components loaded successfully.")
            
        except Exception as e:
            print(f"Error loading SD2 pipeline: {e}")
            return
        
        # pipe 객체 지우기
        del pipe
        gc.collect(); torch.cuda.empty_cache()
        
        try:
            #기존 lora 가중치 로드
            if os.path.exists(lora_weights_path) and os.path.exists(os.path.join(lora_weights_path, "best_model.safetensors")):
                unet_lora = PeftModel.from_pretrained(unet, lora_weights_path)
                print(f"LoRA weights loaded from {lora_weights_path}")
            else:
                # SD2 UNet용 LoRA 설정
                lora_config = LoraConfig(
                    r = self.model_config.get("lora_rank", 8),
                    lora_alpha = self.model_config.get("lora_alpha", 16),
                    target_modules = ["to_k", "to_q", "to_v", "to_out.0", "proj_in", "proj_out"],
                    lora_dropout = 0.05,
                    bias = "none",
                    init_lora_weights = "gaussian",
                )
                unet_lora = get_peft_model(unet, lora_config)
            # vae 및 텍스트 인코더 가중치 동결
            vae.requires_grad_(False)
            text_encoder.requires_grad_(False)
        except Exception as e:
            print(f"Error setting up LoRA: {e}")
            return
        
        # 데이터셋을 훈련/검증 세트로 분할 (70:30 비율)
        train_size = int(0.7 * len(texted_images_for_model3))
        val_size = len(texted_images_for_model3) - train_size
        
        full_train_set = MangaDataset3(texted_images_for_model3)
        
        train_sets, valid_sets = torch.utils.data.random_split(full_train_set, [train_size, val_size])

        # 디버깅: 분할된 데이터셋의 인덱스 확인
        print(f"\n=== DATASET SPLIT DEBUG ===")
        print(f"Total images: {len(texted_images_for_model3)}")
        print(f"Train size: {train_size}, Val size: {val_size}")
        print(f"Train indices (first 10): {list(train_sets.indices[:10])}")
        print(f"Val indices (first 10): {list(valid_sets.indices[:10])}")

        # 실제 이미지 식별자 확인
        print("Train images (first 5):")
        for i in range(min(5, len(train_sets.indices))):
            idx = train_sets.indices[i]
            img = texted_images_for_model3[idx]
            identifier = getattr(img, 'filename', None) or getattr(img, 'path', None) or f"img_id_{id(img)}"
            print(f"  Train idx[{idx}]: {identifier}")

        print("Val images (first 5):")
        for i in range(min(5, len(valid_sets.indices))):
            idx = valid_sets.indices[i]
            img = texted_images_for_model3[idx]
            identifier = getattr(img, 'filename', None) or getattr(img, 'path', None) or f"img_id_{id(img)}"
            print(f"  Val idx[{idx}]: {identifier}")
        print("===========================\n")

        train_loader = torch.utils.data.DataLoader(
            train_sets,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=identity_collate,
        )
        val_loader = torch.utils.data.DataLoader(
            valid_sets,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=identity_collate,
        )
        print(f"데이터셋 분할: 훈련 {len(train_loader)}개, 검증 {len(val_loader)}개")
        
        # optimizer 설정 - Adafactor 자체 스케일링 사용
        optimizer = Adafactor(
            unet_lora.parameters(),
            lr= None,
            scale_parameter=True,
            relative_step=True,
            warmup_init=True,
        )
        
        print("[Model3-pretrained] Optimizer set up.")
        
        # LoRA 학습 루프
        output_dir = self.model_config.get("output_dir", "datas/images/output/model3_pretrained_lora")
        os.makedirs(output_dir, exist_ok=True)
        
        print("[Model3-pretrained] Lora 학습 시작")
        # 최고 검증 손실 추적
        best_val_loss = float('inf')

        # 손실 기록을 위한 리스트 추가
        train_losses = []
        val_losses = []
        epochs_recorded = []

        # 모든 모델을 fp16으로 설정 (훈련 시작 전)
        if weight_dtype == torch.float16:
            unet_lora = unet_lora.half()
            vae = vae.half()
            text_encoder = text_encoder.half()

        # CUDNN 벤치마크 활성화 (반복적인 크기의 입력에 대해 최적화)
        torch.backends.cudnn.benchmark = True

        # 디버깅: 훈련 시작 시 모델 데이터 타입 확인
        print(f"[Training Debug] Model dtypes at start:")
        print(f"  UNet dtype: {next(unet_lora.parameters()).dtype}")
        print(f"  weight_dtype setting: {weight_dtype}")
        print(f"  Mixed precision enabled: {weight_dtype == torch.float16}")
        
        # 노이즈 스케줄러 설정
        noise_scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False
            )
        
        for epoch in tqdm(range(num_epochs), desc="Epochs"):
            unet_lora.train()
            epoch_loss = 0.0
            num_train_batches = 0

            step = 0
            for batch_images in tqdm(train_loader, desc="Training batches"):
                # 디버깅: 첫 번째 에포크의 첫 번째 배치에서 파일명 출력
                if step == 0 and epoch == 0:
                    print("\n=== TRAINING BATCH DEBUG ===")
                    print("First training batch images:")
                    for i, img in enumerate(batch_images):
                        # TextedImage 객체의 고유 식별자 출력 (가능한 속성들 시도)
                        identifier = getattr(img, 'filename', None) or getattr(img, 'path', None) or f"img_id_{id(img)}"
                        print(f"  Train[{i}]: {identifier}")
                    print("=============================\n")

                step += 1
                # 메모리 정리
                torch.cuda.empty_cache()
                
                # 모든 입력 데이터를 fp16으로 통일
                original_pixel_values_batch = torch.stack(
                    [img.orig for img in batch_images]
                ).to(self.device, dtype=weight_dtype)  # fp16으로 변경

                mask_pixel_values_batch = torch.stack(
                    [img.mask for img in batch_images]
                ).to(self.device, dtype=weight_dtype)

                with torch.no_grad():
                    # VAE 인코딩 (이미 fp16으로 설정됨)
                    vae.to(self.device)
                    
                    
                    target_latents_batch = vae.encode(original_pixel_values_batch).latent_dist.sample() * vae.config.scaling_factor
                    target_latents_batch = target_latents_batch.to(dtype=weight_dtype)
                    
                    latent_mask_batch = F.interpolate(
                        mask_pixel_values_batch,
                        size=target_latents_batch.shape[-2:],
                        mode="nearest"
                    ) # [B, 1, H_lat, W_lat]
                    
                      
                    # 텍스트 임베딩 생성 - SD2는 단일 텍스트 인코더 사용
                    text_encoder.to(self.device)

                    prompt_embeds, negative_prompt_embeds = self._encode_prompt_sd2(
                        prompt, negative_prompt, tokenizer, text_encoder, self.device, len(batch_images)
                    )

                    # 임베딩 준비 - CFG를 위해 negative와 positive 결합 (이미 fp16이므로 변환 불필요)
                    prompt_embeds_full = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)


                noise_batch = torch.randn_like(target_latents_batch)
                timesteps_batch = torch.randint(0, noise_scheduler.num_train_timesteps, (len(batch_images), ), device=self.device).long()
                
                # 타겟 잠재 벡터에 노이즈 추가가
                noisy_target_latents = noise_scheduler.add_noise(target_latents_batch, noise_batch, timesteps_batch) # type: ignore
                # [B, C_lat, H_lat, W_lat] 
                #입력 모델 구성
                initial_lantents = target_latents_batch * (1 - latent_mask_batch) + noisy_target_latents * latent_mask_batch
                    
                    
                with autocast("cuda",dtype=weight_dtype):
                    # SD2 Inpainting용 9채널 입력 구성
                    # [latent(4) + masked_latent(4) + mask(1)] = 9채널

                    # 마스크된 latent 생성 (텍스트 영역을 노이즈로 채움)
                    masked_latents = initial_lantents * (1 - latent_mask_batch)

                    # 마스크를 latent 크기에 맞게 리사이즈
                    mask_for_unet = F.interpolate(
                        latent_mask_batch,
                        size=initial_lantents.shape[-2:],
                        mode="nearest"
                    )

                    # 9채널 입력 구성: [latent, masked_latent, mask]
                    latent_model_input = torch.cat([
                        initial_lantents,      # 4채널: 현재 latent
                        masked_latents,        # 4채널: 마스크된 latent
                        mask_for_unet          # 1채널: 마스크
                    ], dim=1)  # 총 9채널

                    # CFG를 위해 복제
                    latent_model_input = torch.cat([latent_model_input] * 2, dim=0)
                    timesteps_input = torch.cat([timesteps_batch] * 2, dim=0)

                    # 데이터 타입 일관성 확보
                    latent_model_input = latent_model_input.to(dtype=weight_dtype)
                    timesteps_input = timesteps_input.to(dtype=torch.long)  # timesteps는 long 타입이어야 함

                    noise_pred = unet_lora(
                        sample=latent_model_input,
                        timestep=timesteps_input,
                        encoder_hidden_states=prompt_embeds_full,
                        return_dict=False
                    )[0]  # return_dict=False일 때는 tuple의 첫 번째 요소가 sample

                    #noise_pred 분리리
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    # 손실 계산
                    loss = F.mse_loss(noise_pred_text.to(torch.float32), noise_batch.to(torch.float32), reduction="none")
                    # 손실 가중치 계산
                    mask_weight = self.model_config.get("mask_weight", 2.0)
                    unmask_weight = 1.0
                    weight_map_per_element = (
                        latent_mask_batch.to(loss.device, dtype=torch.float32) * (mask_weight - unmask_weight) 
                        + unmask_weight
                    )
                    weighted_loss_map = loss * weight_map_per_element
                    loss = weighted_loss_map.mean()
                
                
                
                optimizer.zero_grad(set_to_none=True)

                # Adafactor는 자체 스케일링을 하므로 일반 역전파 사용
                loss.backward()

                # 그래디언트 클리핑 (Adafactor와 함께 사용)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in unet_lora.parameters() if p.requires_grad],
                    max_norm=self.model_config.get("max_grad_norm", 1.0)
                )

                # 옵티마이저 스텝 (Adafactor가 자체적으로 스케일링 처리)
                optimizer.step()
            
                
                # 손실 추적
                epoch_loss += loss.detach().item()
                num_train_batches += 1

                # 메모리 정리
                del loss, noise_pred, latent_model_input, timesteps_input
                torch.cuda.empty_cache()

            # 에폭 종료 후 검증 손실 계산
            if epoch > 0:
                # 올바른 평균 계산: 총 손실을 배치 수로 나눔
                avg_train_loss = epoch_loss / num_train_batches if num_train_batches > 0 else 0.0

                print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}")

                # 손실 기록 저장
                train_losses.append(avg_train_loss) 
                epochs_recorded.append(epoch + 1)

                # 검증 손실 계산 (validation set이 있을 때만)
                if val_loader is not None:
                    try:
                        val_loss = self._calculate_validation_loss_sd2(
                            unet_lora, vae, text_encoder, tokenizer, noise_scheduler,
                            val_loader, weight_dtype
                        )
                    except Exception as e:
                        print(f"Validation error: {e}")
                        print("Using train loss as validation loss")
                        val_loss = avg_train_loss
                else:
                    val_loss = float('inf')  # validation set이 없으면 무한대로 설정
                print(f"Epoch {epoch+1}/{num_epochs} - Validation Loss: {val_loss:.4f}")

                # validation 손실도 기록
                val_losses.append(val_loss)

                # 에폭마다 모델 저장
                # 검증 손실이 개선되면 best_model 저장
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    print(f"새로운 최고 검증 손실: {best_val_loss:.4f}, 모델 저장 중...")

                    # 모델 저장 경로 설정
                    os.makedirs(lora_weights_path, exist_ok=True)
                    unet_lora.save_pretrained(lora_weights_path)
                    print("모델 저장 완료.")

        # 훈련 완료 후 최종 손실 기록 저장 및 시각화
        print("\n📊 Saving training results and generating final plots...")
        self._save_and_visualize_losses(train_losses, val_losses, epochs_recorded, lora_weights_path)

    def _encode_prompt_sd2(self, prompt, negative_prompt, tokenizer, text_encoder, device, batch_size=1):
        """
        SD2용 텍스트 인코딩 헬퍼 메서드 - 단일 텍스트 인코더만 사용
        """
        # 배치 크기에 맞게 프롬프트 복제
        prompts = [prompt] * batch_size
        negative_prompts = [negative_prompt] * batch_size

        # 공통 max_length 설정
        max_length = 77  # SD2의 표준 시퀀스 길이

        # 텍스트 인코더 처리 (CLIP ViT-L/14)
        text_inputs = tokenizer(
            prompts,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(device)

        uncond_input = tokenizer(
            negative_prompts,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
        uncond_input_ids = uncond_input.input_ids.to(device)

        # 텍스트 인코더 출력 (fp16으로 통일)
        with torch.no_grad():
            prompt_embeds = text_encoder(text_input_ids)[0]  # [B, S, D]
            negative_prompt_embeds = text_encoder(uncond_input_ids)[0]  # [B, S, D]

            # 텍스트 인코더가 fp16이므로 출력도 자동으로 fp16이 됨
            # 명시적으로 확인하고 변환 (안전장치)
            if prompt_embeds.dtype != torch.float16:
                prompt_embeds = prompt_embeds.half()
            if negative_prompt_embeds.dtype != torch.float16:
                negative_prompt_embeds = negative_prompt_embeds.half()

        return prompt_embeds, negative_prompt_embeds

    def _save_and_visualize_losses(self, train_losses, val_losses, epochs, save_dir):
        """
        훈련 완료 후 최종 손실 데이터 저장 및 시각화
        """
        import matplotlib.pyplot as plt
        import json

        try:
            # 1. 손실 데이터를 JSON으로 저장
            loss_data = {
                "epochs": epochs,
                "train_losses": train_losses,
                "val_losses": val_losses,
                "total_epochs": len(epochs),
                "final_train_loss": train_losses[-1] if train_losses else None,
                "final_val_loss": val_losses[-1] if val_losses else None,
                "best_val_loss": min(val_losses) if val_losses else None,
                "best_val_epoch": epochs[val_losses.index(min(val_losses))] if val_losses else None
            }

            loss_file_path = os.path.join(save_dir, "training_losses.json")
            with open(loss_file_path, 'w') as f:
                json.dump(loss_data, f, indent=2)
            print(f"📊 Loss data saved to {loss_file_path}")

            # 2. 최종 손실 그래프 생성
            plt.figure(figsize=(15, 10))

            # 서브플롯 1: 전체 손실 비교
            plt.subplot(2, 2, 1)
            if train_losses:
                plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2, marker='o', markersize=3)
            if val_losses:
                plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2, marker='s', markersize=3)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training and Validation Loss')
            plt.legend()
            plt.grid(True, alpha=0.3)

            # 서브플롯 2: 로그 스케일
            plt.subplot(2, 2, 2)
            if train_losses:
                plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2, marker='o', markersize=3)
            if val_losses:
                plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2, marker='s', markersize=3)
            plt.xlabel('Epoch')
            plt.ylabel('Loss (Log Scale)')
            plt.title('Loss (Log Scale)')
            plt.yscale('log')
            plt.legend()
            plt.grid(True, alpha=0.3)

            # 서브플롯 3: 손실 차이
            plt.subplot(2, 2, 3)
            if train_losses and val_losses and len(train_losses) == len(val_losses):
                loss_diff = [v - t for t, v in zip(train_losses, val_losses)]
                plt.plot(epochs, loss_diff, 'g-', label='Val - Train Loss', linewidth=2, marker='^', markersize=3)
                plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
                plt.xlabel('Epoch')
                plt.ylabel('Loss Difference')
                plt.title('Validation - Training Loss')
                plt.legend()
                plt.grid(True, alpha=0.3)

            # 서브플롯 4: 통계 정보
            plt.subplot(2, 2, 4)
            plt.axis('off')

            # 통계 값들을 미리 계산
            final_train = f"{train_losses[-1]:.4f}" if train_losses else "N/A"
            final_val = f"{val_losses[-1]:.4f}" if val_losses else "N/A"
            best_val = f"{min(val_losses):.4f}" if val_losses else "N/A"
            best_epoch = epochs[val_losses.index(min(val_losses))] if val_losses else "N/A"

            if train_losses and len(train_losses) > 1:
                loss_reduction = f"{((train_losses[0] - train_losses[-1]) / train_losses[0] * 100):.1f}%"
            else:
                loss_reduction = "N/A"

            stats_text = f"""
Training Summary:
• Total Epochs: {len(epochs)}
• Final Train Loss: {final_train}
• Final Val Loss: {final_val}
• Best Val Loss: {best_val}
• Best Val Epoch: {best_epoch}
• Loss Reduction: {loss_reduction}
            """
            plt.text(0.1, 0.9, stats_text, transform=plt.gca().transAxes,
                    verticalalignment='top', fontsize=10, fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

            plt.tight_layout()

            # 최종 그래프 저장
            final_plot_path = os.path.join(save_dir, "final_training_analysis.png")
            plt.savefig(final_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"📈 Final training analysis saved to {final_plot_path}")

            # 3. 요약 통계 저장
            summary_file_path = os.path.join(save_dir, "training_summary.json")
            with open(summary_file_path, 'w') as f:
                json.dump(loss_data, f, indent=2)
            print(f"📋 Training summary saved to {summary_file_path}")

        except Exception as e:
            print(f"Error saving final visualization: {e}")

    def _calculate_validation_loss_sd2(self, unet_lora, vae, text_encoder, tokenizer, noise_scheduler,
                             val_loader, weight_dtype):
        """
        SD2용 인페인팅 학습 방식에 맞게 수정된 검증 손실 계산 함수
        """
        # 모든 모델을 fp16으로 설정하고 평가 모드로 전환
        unet_lora.eval()
        unet_lora = unet_lora.half()  # UNet도 fp16으로 명시적 변환
        vae = vae.half()  # VAE를 fp16으로
        text_encoder = text_encoder.half()  # Text Encoder를 fp16으로

        device = self.device
        total_val_loss = 0.0
        num_val_batches = 0
        
        prompt = self.model_config.get("prompts", "pure black and white manga style image with no color tint, absolute grayscale, contextual manga style, remove lettering, remove text, remove logo, remove watermark, consistent with surrounding")
        negative_prompt = self.model_config.get("negative_prompt", "photo, realistic, color, colorful, purple, violet, sepia, any color tint, blurry")
        
        with torch.no_grad():
            for batch_images in val_loader:
                # 디버깅: 첫 번째 검증 배치에서 파일명 출력
                if num_val_batches == 0:
                    print("\n=== VALIDATION BATCH DEBUG ===")
                    print("First validation batch images:")
                    for i, img in enumerate(batch_images):
                        # TextedImage 객체의 고유 식별자 출력 (가능한 속성들 시도)
                        identifier = getattr(img, 'filename', None) or getattr(img, 'path', None) or f"img_id_{id(img)}"
                        print(f"  Val[{i}]: {identifier}")
                    print("===============================\n")

                torch.cuda.empty_cache()

                # 모든 입력 데이터를 fp16으로 통일
                original_pixel_values = torch.stack([img.orig for img in batch_images]).to(device, dtype=weight_dtype)  # fp16
                mask_pixel_values = torch.stack([img.mask for img in batch_images]).to(device, dtype=weight_dtype)  # fp16
                
                vae.to(device)
                target_latents = vae.encode(original_pixel_values).latent_dist.sample() * vae.config.scaling_factor
                target_latents = target_latents.to(dtype=weight_dtype)
                
                latent_mask = F.interpolate(
                    mask_pixel_values, size=target_latents.shape[-2:], mode="nearest"
                )
                
                # KEY CHANGE: 검증에서도 학습과 동일한 방식으로 입력을 구성합니다.
                # 1. 타겟 노이즈와 타임스텝 생성
                noise = torch.randn_like(target_latents)
                timesteps = torch.randint(
                    0, noise_scheduler.num_train_timesteps, (len(batch_images),), device=device
                ).long()
                
                # 2. 노이즈가 추가된 타겟 잠재 벡터 생성
                noisy_target_latents = noise_scheduler.add_noise(target_latents, noise, timesteps)
                
                # 3. 모델의 실제 입력(initial_latents) 구성
                initial_latents = target_latents * (1 - latent_mask) + noisy_target_latents * latent_mask
                
                # 텍스트 인코딩 (SD2용)
                text_encoder.to(device)

                prompt_embeds, negative_prompt_embeds = self._encode_prompt_sd2(
                    prompt, negative_prompt, tokenizer, text_encoder, device, len(batch_images)
                )

                # 텍스트 임베딩 결합
                prompt_embeds_full = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0).to(dtype=weight_dtype)

                # SD2 Inpainting용 9채널 입력 구성 (validation에서도 동일)
                # 마스크된 latent 생성
                masked_latents = initial_latents * (1 - latent_mask)

                # 마스크를 latent 크기에 맞게 리사이즈
                mask_for_unet = F.interpolate(
                    latent_mask,
                    size=initial_latents.shape[-2:],
                    mode="nearest"
                )

                # 9채널 입력 구성: [latent, masked_latent, mask]
                latent_model_input = torch.cat([
                    initial_latents,      # 4채널: 현재 latent
                    masked_latents,       # 4채널: 마스크된 latent
                    mask_for_unet         # 1채널: 마스크
                ], dim=1)  # 총 9채널

                # CFG를 위해 복제
                latent_model_input = torch.cat([latent_model_input] * 2, dim=0)
                timesteps_input = torch.cat([timesteps] * 2, dim=0)

                # validation에서도 훈련과 동일한 fp16 사용
                latent_model_input = latent_model_input.to(dtype=weight_dtype)
                timesteps_input = timesteps_input.to(dtype=torch.long)
                prompt_embeds_full = prompt_embeds_full.to(dtype=weight_dtype)

                noise_pred = unet_lora(
                    sample=latent_model_input,
                    timestep=timesteps_input,
                    encoder_hidden_states=prompt_embeds_full,
                    return_dict=False
                )[0]
                
                _, noise_pred_text = noise_pred.chunk(2)
                
                # 손실 계산 (타겟은 실제 추가된 노이즈 `noise`)
                loss = F.mse_loss(noise_pred_text.to(torch.float32), noise.to(torch.float32), reduction="none")
                
                mask_weight = self.model_config.get("mask_weight", 2.0)
                unmask_weight = 1.0
                weight_map_per_element = (
                    latent_mask.to(loss.device, dtype=torch.float32) * (mask_weight - unmask_weight) 
                    + unmask_weight
                )
                weighted_loss_map = loss * weight_map_per_element
                val_loss = weighted_loss_map.mean()
                
                total_val_loss += val_loss.item()
                num_val_batches += 1
                
                del val_loss, noise_pred, latent_model_input, timesteps_input
                torch.cuda.empty_cache()
        
        avg_val_loss = total_val_loss / num_val_batches if num_val_batches > 0 else float('inf')
        return avg_val_loss


    def inference(self, texted_images_to_inpaint: list[TextedImage]) -> list[TextedImage]:
        """
        Stable Diffusion 2.0 Inpainting 모델을 사용하여 각 TextedImage의 마스크된 영역을 인페인팅

        Args:
            texted_images_to_inpaint: 인페인팅할 TextedImage 객체 리스트

        Returns:
            list[TextedImage]: 인페인팅된 결과가 orig에 저장된 TextedImage 객체 리스트
        """
        if len(texted_images_to_inpaint) == 0:
            print("No images to process for Model3 inference.")
            return []

        print("Loading Stable Diffusion Inpainting pipeline...")

        try:
            # 1. 설정 가져오기
            model_id = self.model_config["model_id"]
            prompt = self.model_config.get("prompts", "pure black and white manga style image with no color tint, absolute grayscale, contextual manga style, remove lettering, remove text, remove logo, remove watermark, consistent with surrounding")
            negative_prompt = self.model_config.get("negative_prompt", "photo, realistic, color, colorful, purple, violet, sepia, any color tint, blurry")
            guidance_scale = self.model_config.get("guidance_scale", 7.5)
            num_inference_steps = self.model_config.get("inference_steps", 28)

            # 2. Stable Diffusion 2.0 Inpainting 파이프라인 로드 (고품질 만화 스타일에 적합)
            pipe = StableDiffusionInpaintPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
            )

            # 3. LoRA 가중치 로드 및 적용 (있는 경우에만)
            lora_weights_path = self.model_config.get("lora_path", "")
            if lora_weights_path and os.path.exists(lora_weights_path):
                try:
                    # LoRA 가중치가 실제로 존재하는지 확인
                    if os.path.exists(os.path.join(lora_weights_path, "adapter_model.safetensors")) or \
                       os.path.exists(os.path.join(lora_weights_path, "best_model.safetensors")):

                        unet = pipe.unet
                        unet_lora = PeftModel.from_pretrained(unet, lora_weights_path)
                        pipe.unet = unet_lora  # UNet을 LoRA 버전으로 교체
                        print(f"✅ LoRA weights loaded from {lora_weights_path}")
                    else:
                        print(f"⚠️  LoRA weight files not found in {lora_weights_path}, using base model")
                except Exception as e:
                    print(f"⚠️  Error loading LoRA weights: {e}")
                    print("Using base model instead")
            else:
                print("🔧 No LoRA path specified or path doesn't exist, using base model")

            # 4. 모델을 GPU로 이동
            pipe.to(self.device)

            # 5. 출력 디렉토리 설정
            output_dir = self.model_config.get("output_dir", "trit/datas/images/output")
            os.makedirs(output_dir, exist_ok=True)

            # 6. 각 패치 처리
            for i, current_patch in enumerate(tqdm(texted_images_to_inpaint, desc="Inpainting patches")):
                try:
                    # VRAM 관리
                    torch.cuda.empty_cache()

                    # 텐서 크기 디버깅
                    print(f"[Model3 Inference] Patch {i+1} tensor shapes:")
                    print(f"  current_patch.orig: {current_patch.orig.shape} (C x H x W)")
                    print(f"  current_patch.mask: {current_patch.mask.shape} (C x H x W)")

                    # 원본 이미지와 마스크를 PIL로 변환
                    to_pil = transforms.ToPILImage()
                    orig_pil = to_pil(current_patch.orig.cpu())
                    mask_pil = to_pil(current_patch.mask.cpu().squeeze(0))

                    # PIL 이미지 크기 디버깅
                    print(f"[Model3 Inference] PIL image sizes:")
                    print(f"  orig_pil: {orig_pil.size} (W x H)")
                    print(f"  mask_pil: {mask_pil.size} (W x H)")

                    # 마스크 이진화 (0 또는 255)
                    mask_np = np.array(mask_pil)
                    mask_binary = (mask_np > 127).astype(np.uint8) * 255
                    mask_binary_pil = Image.fromarray(mask_binary, "L")

                    # 이미지 크기 가져오기
                    width, height = orig_pil.size

                    # Stable Diffusion 입력 크기 디버깅
                    print(f"[Model3 Inference] SD input parameters:")
                    print(f"  width: {width}, height: {height}")
                    print(f"  width % 8 = {width % 8} ({'OK' if width % 8 == 0 else 'ERROR!'})")
                    print(f"  height % 8 = {height % 8} ({'OK' if height % 8 == 0 else 'ERROR!'})")

                    # SD 인페인팅 실행
                    result = pipe(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        height=height,
                        width=width,
                        image=orig_pil,
                        mask_image=mask_binary_pil,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        strength=0.8
                    ).images[0]

                    # 인페인팅 결과 저장
                    result.save(f"{output_dir}/result_patch_{i:03d}.png")

                    # 결과를 텐서로 변환하여 TextedImage에 저장
                    to_tensor = transforms.ToTensor()
                    result_tensor = to_tensor(result).to(
                        device=current_patch.orig.device,
                        dtype=current_patch.orig.dtype
                    )

                    # 결과 텐서 크기 검증
                    if result_tensor.shape[1] == 0 or result_tensor.shape[2] == 0:
                        print(f"Warning: Inpainting result has zero dimensions {result_tensor.shape}. Skipping patch {i+1}")
                        continue

                    # 원본 텐서와 크기 맞추기
                    if result_tensor.shape != current_patch.orig.shape:
                        result_tensor = result_tensor.unsqueeze(0)

                    # 마스크 영역만 인페인팅 결과로 대체
                    mask_for_compositing = current_patch.mask.to(
                        dtype=result_tensor.dtype,
                        device=result_tensor.device
                    )

                    # 원본 이미지 복사
                    original_tensor = current_patch.orig.clone()

                    # 마스크 영역만 인페인팅 결과로 대체
                    composited_result = original_tensor * (1 - mask_for_compositing) + result_tensor * mask_for_compositing

                    # TextedImage 객체 업데이트
                    current_patch.orig = composited_result

                    print(f"Successfully inpainted patch {i+1}/{len(texted_images_to_inpaint)}")

                except Exception as e:
                    print(f"Error during inference for patch {i+1}: {e}")
                    # 오류 발생 시 원본 패치 유지
                    continue

            # 7. 메모리 정리
            del pipe
            gc.collect()
            torch.cuda.empty_cache()

            print(f"Inference completed. Results saved to {output_dir}")
            return texted_images_to_inpaint

        except Exception as e:
            print(f"Error initializing Stable Diffusion Inpainting pipeline: {e}")
            return texted_images_to_inpaint