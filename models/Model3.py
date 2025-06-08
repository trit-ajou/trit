import math
import torch
import os
import numpy as np
import gc
import torch.nn.functional as F
from torch.amp.grad_scaler import GradScaler
from torch import autocast
from ..datas.Dataset import MangaDataset3
from ..datas.TextedImage import TextedImage
from torch import FloatTensor, nn
from torchvision import transforms # 이미지 전처리를 위해 추가 임포트
from transformers.optimization import Adafactor
from tqdm import tqdm

from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.models.transformers.transformer_sd3 import SD3Transformer2DModel
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import StableDiffusion3Pipeline
from peft import get_peft_model, LoraConfig, PeftModel
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

def identity_collate(batch_images):
    return batch_images

# 부동소수점 행렬 곱셈 정밀도 설정
torch.set_float32_matmul_precision('high')
print("부동소수점 행렬 곱셈 정밀도를 'high'로 설정")

class Model3(nn.Module):
    def __init__(self, model_config: dict, device: str = "cuda"):
        super().__init__()
        self.model_config = model_config
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
    
    def lora_train(self, texted_images_for_model3: list[TextedImage]):
        print("Loading SD3 pipeline for LoRA training...")

        weight_dtype = torch.float32
        model_id = self.model_config["model_id"]
        lora_weights_path = self.model_config["lora_path"]
        num_epochs = self.model_config.get("epochs", 10)
        batch_size = self.model_config.get("batch_size", 4)
        # 텍스트 인코딩
        prompt = self.model_config.get("prompts", "pure black and white manga style image with no color tint, absolute grayscale, contextual manga style")
        negative_prompt = self.model_config.get("negative_prompt", "photo, realistic, color, colorful, purple, violet, sepia, any color tint, blurry")
        
        # 파이프라인 로딩
        try:
            pipe = StableDiffusion3Pipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float32,
            )
            pipe.to(self.device)
            print(f"SD3 pipeline loaded successfully to {self.device}.")
            
            # 필요한 부분 로드
            vae = pipe.vae
            text_encoder = pipe.text_encoder
            text_encoder_2 = pipe.text_encoder_2
            text_encoder_3 = pipe.text_encoder_3
            transformer = pipe.transformer
            tokenizer = pipe.tokenizer
            tokenizer_2 = pipe.tokenizer_2
            tokenizer_3 = pipe.tokenizer_3
            print("SD3 components loaded successfully.")
            
        except Exception as e:
            print(f"Error loading SD3 pipeline: {e}")
            return
        
        # pipe 객체 지우기
        del pipe
        gc.collect(); torch.cuda.empty_cache()
        
        try:
            #기존 lora 가중치 로드
            if os.path.exists(lora_weights_path) and os.path.exists(os.path.join(lora_weights_path, "adapter_model.safetensors")):
                transformer_lora = PeftModel.from_pretrained(transformer, lora_weights_path, is_trainable = True)
                print(f"LoRA weights loaded from {lora_weights_path}")
            else:

                lora_config = LoraConfig(
                    r = self.model_config.get("lora_rank", 8),
                    lora_alpha = self.model_config.get("lora_alpha", 16),
                    target_modules = ["to_k", "to_q", "to_v", "to_out.0"],
                    lora_dropout = 0.1,
                    bias = "none",
                    init_lora_weights = "gaussian",
                )
                transformer_lora = get_peft_model(transformer, lora_config)
            transformer_lora.to(self.device)
            # vae 및 텍스트 인코더 가중치 동결
            vae.requires_grad_(False)
            text_encoder.requires_grad_(False)
            if text_encoder_2: text_encoder_2.requires_grad_(False)
            if text_encoder_3: text_encoder_3.requires_grad_(False)
        except Exception as e:
            print(f"Error setting up LoRA: {e}")
            return
        
        
        # 데이터셋을 훈련/검증 세트로 분할
        train_size = int(0.8 * len(texted_images_for_model3))
        val_size = len(texted_images_for_model3) - train_size
        
        full_train_set = MangaDataset3(texted_images_for_model3)
        
        train_sets, valid_sets = torch.utils.data.random_split(full_train_set, [train_size, val_size])
        
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
        
        optimizer = torch.optim.AdamW(
            transformer_lora.parameters(),
            lr=1e-5,  # Adafactor보다 훨씬 작은 학습률로 시작 (예: 1e-5 또는 5e-6)
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=1e-2 # L2 정규화(과적합 방지)
        )
        
        print("[Model3] Optimizer set up.")
        # LoRA 학습 루프
        output_dir = self.model_config.get("output_dir", "datas/images/output/model3_sd3_lora")
        os.makedirs(output_dir, exist_ok=True)
        
        print("[Model3] Lora 학습 시작")
        # 최고 검증 손실 추적
        best_val_loss = float('inf')
        # CUDNN 벤치마크 활성화 (반복적인 크기의 입력에 대해 최적화)
        torch.backends.cudnn.benchmark = True
        
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
            transformer_lora.train()
            epoch_loss = 0.0
            
            for i, batch_images in enumerate(train_loader):
                torch.cuda.empty_cache()
                
                original_pixel_values_batch = torch.stack(
                    [img.orig for img in batch_images]
                ).to(self.device, dtype=torch.float32) 

                mask_pixel_values_batch = torch.stack(
                    [img.mask for img in batch_images]
                ).to(self.device, dtype=weight_dtype)   
                
                with torch.no_grad():
                    vae.to(self.device)
                    target_latents_batch = vae.encode(original_pixel_values_batch).latent_dist.sample() * vae.config.scaling_factor
                    
                    latent_mask_batch = F.interpolate(
                        mask_pixel_values_batch,
                        size=target_latents_batch.shape[-2:],
                        mode="nearest"
                    )
                    
                    text_encoder.to(self.device)
                    if text_encoder_2: text_encoder_2.to(self.device)
                    if text_encoder_3: text_encoder_3.to(self.device)

                    prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = self._encode_prompt(
                        prompt, negative_prompt, tokenizer, tokenizer_2, tokenizer_3,
                        text_encoder, text_encoder_2, text_encoder_3, self.device, len(batch_images)
                    )

                    prompt_embeds_full = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
                    pooled_embeddings_full = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
                # =================== with torch.no_grad() 블록 종료 ===================

                noise_batch = torch.randn_like(target_latents_batch)
                timesteps_batch = torch.randint(0, noise_scheduler.config.num_train_timesteps, (len(batch_images), ), device=self.device).long()
                
                noisy_target_latents = noise_scheduler.add_noise(target_latents_batch, noise_batch, timesteps_batch) #type: ignore
                
                initial_latents = target_latents_batch * (1 - latent_mask_batch) + noisy_target_latents * latent_mask_batch
                
                latent_model_input = torch.cat([initial_latents]*2, dim=0)
                timesteps_input = torch.cat([timesteps_batch] * 2, dim = 0)
                
                # 학습 대상 모델의 순전파
                noise_pred = transformer_lora(
                    hidden_states=latent_model_input,
                    timestep=timesteps_input,
                    encoder_hidden_states=prompt_embeds_full,
                    pooled_projections=pooled_embeddings_full,
                    return_dict=False
                )[0]
                
                _, noise_pred_text = noise_pred.chunk(2)
                
                loss = F.mse_loss(noise_pred_text, noise_batch, reduction="none")
                
                mask_weight = self.model_config.get("mask_weight", 2.0)
                unmask_weight = 1.0
                weight_map_per_element = (
                    latent_mask_batch * (mask_weight - unmask_weight) 
                    + unmask_weight
                )
                weighted_loss_map = loss * weight_map_per_element
                loss = weighted_loss_map.mean()
                optimizer.zero_grad(set_to_none=True)
            
                # 역전파
                loss.backward()

                # (선택) 그래디언트 클리핑
                torch.nn.utils.clip_grad_norm_(
                    [p for p in transformer_lora.parameters() if p.requires_grad],
                    max_norm=self.model_config.get("max_grad_norm", 1.0) 
                )
                
                optimizer.step()
                
                epoch_loss += loss.detach().item()
                del loss, noise_pred, latent_model_input, timesteps_input
                torch.cuda.empty_cache()
                
             # 에폭 종료 후 검증 손실 계산
            if epoch > 0:
                avg_train_loss = epoch_loss / epoch
                print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}")
                
                # 검증 손실 계산
                val_loss = self._calculate_validation_loss(
                    transformer_lora, vae, text_encoder, text_encoder_2, text_encoder_3,
                    tokenizer, tokenizer_2, tokenizer_3, noise_scheduler,
                    val_loader, weight_dtype
                )
                print(f"Epoch {epoch+1}/{num_epochs} - Validation Loss: {val_loss:.4f}")
                
                # 에폭마다 모델 저장
                # 검증 손실이 개선되면 best_model 저장
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    print(f"새로운 최고 검증 손실: {best_val_loss:.4f}, 모델 저장 중...")

                    # 모델 저장 경로 설정
                    os.makedirs(lora_weights_path, exist_ok=True)
                    transformer_lora.save_pretrained(lora_weights_path)
                    print("모델 저장 완료.")



    def _calculate_validation_loss(self, transformer_lora, vae, text_encoder, text_encoder_2, text_encoder_3,
                             tokenizer, tokenizer_2, tokenizer_3, noise_scheduler,
                             val_loader, weight_dtype):
        """
        인페인팅 학습 방식에 맞게 수정된 검증 손실 계산 함수
        """
        transformer_lora.eval()
        device = self.device
        total_val_loss = 0.0
        num_val_batches = 0
        
        prompt = self.model_config.get("prompts", "pure black and white manga style image with no color tint, absolute grayscale, contextual manga style")
        negative_prompt = self.model_config.get("negative_prompt", "photo, realistic, color, colorful, purple, violet, sepia, any color tint, blurry")
        
        with torch.no_grad():
            for batch_images in val_loader:
                torch.cuda.empty_cache()
                
                original_pixel_values = torch.stack([img.orig for img in batch_images]).to(device, dtype=torch.float32)
                mask_pixel_values = torch.stack([img.mask for img in batch_images]).to(device, dtype=weight_dtype)
                
                vae.to(device)
                target_latents = vae.encode(original_pixel_values).latent_dist.sample() * vae.config.scaling_factor
                target_latents = target_latents.to(dtype=weight_dtype)
                
                latent_mask = F.interpolate(
                    mask_pixel_values, size=target_latents.shape[-2:], mode="nearest"
                )
                
                # KEY CHANGE: 검증에서도 학습과 동일한 방식으로 입력을 구성합니다.
                # 1. 타겟 노이즈와 타임스텝 생성
                noise = torch.randn_like(target_latents)
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (len(batch_images), ), device=self.device).long()
                
                # 2. 노이즈가 추가된 타겟 잠재 벡터 생성
                noisy_target_latents = noise_scheduler.add_noise(target_latents, noise, timesteps)
                
                # 3. 모델의 실제 입력(initial_latents) 구성
                initial_latents = target_latents * (1 - latent_mask) + noisy_target_latents * latent_mask
                
                # 텍스트 인코딩
                text_encoder.to(device)
                if text_encoder_2: text_encoder_2.to(device)
                if text_encoder_3: text_encoder_3.to(device)
                
                prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = self._encode_prompt(
                    prompt, negative_prompt, tokenizer, tokenizer_2, tokenizer_3,
                    text_encoder, text_encoder_2, text_encoder_3, device, len(batch_images)
                )
                
                prompt_embeds_full = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0).to(dtype=weight_dtype)
                pooled_embeddings_full = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0).to(dtype=weight_dtype)
                
                # 모델 입력 준비 및 노이즈 예측
                latent_model_input = torch.cat([initial_latents] * 2, dim=0)
                timesteps_input = torch.cat([timesteps] * 2, dim=0)
                
                noise_pred = transformer_lora(
                    hidden_states=latent_model_input,
                    timestep=timesteps_input,
                    encoder_hidden_states=prompt_embeds_full,
                    pooled_projections=pooled_embeddings_full,
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


    def _encode_prompt(self, prompt, negative_prompt, tokenizer, tokenizer_2, tokenizer_3,
                     text_encoder, text_encoder_2, text_encoder_3, device, batch_size=1):
        """
        텍스트 인코딩 헬퍼 메서드 - SD3 파이프라인 방식으로 수정
        """
        # 배치 크기에 맞게 프롬프트 복제
        prompts = [prompt] * batch_size
        negative_prompts = [negative_prompt] * batch_size
        
        # 공통 max_length 설정
        max_length = 77  # SD3의 표준 시퀀스 길이
        
        # 텍스트 인코더 1 처리 (CLIP ViT-L/14)
        text_inputs = tokenizer(
            prompts,  # 배치 크기 적용
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(device)
        
        uncond_input = tokenizer(
            negative_prompts,  # 배치 크기 적용
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
        uncond_input_ids = uncond_input.input_ids.to(device)
        
        # 텍스트 인코더 1 출력 (CLIP: 768 차원)
        encoder_output_1 = text_encoder(text_input_ids)
        
        # CLIPTextModelOutput 객체 처리
        if hasattr(encoder_output_1, 'last_hidden_state'):
            prompt_embeds = encoder_output_1.last_hidden_state  # (B, S, 768)
        elif isinstance(encoder_output_1, tuple) and len(encoder_output_1) > 0:
            prompt_embeds = encoder_output_1[0]
        else:
            prompt_embeds = encoder_output_1  # 직접 텐서인 경우
        
        # 차원 확인 및 수정
        if isinstance(prompt_embeds, torch.Tensor) and len(prompt_embeds.shape) == 2:  # (B, D) 형태인 경우
            # 시퀀스 차원 추가 (B, D) -> (B, 1, D)
            prompt_embeds = prompt_embeds.unsqueeze(1)
            # 시퀀스 길이를 max_length로 확장
            prompt_embeds = prompt_embeds.repeat(1, max_length, 1)
        
        # 동일한 처리를 negative_prompt에도 적용
        uncond_output_1 = text_encoder(uncond_input_ids)
        
        # CLIPTextModelOutput 객체 처리
        if hasattr(uncond_output_1, 'last_hidden_state'):
            negative_prompt_embeds = uncond_output_1.last_hidden_state
        elif isinstance(uncond_output_1, tuple) and len(uncond_output_1) > 0:
            negative_prompt_embeds = uncond_output_1[0]
        else:
            negative_prompt_embeds = uncond_output_1
        
        if isinstance(negative_prompt_embeds, torch.Tensor) and len(negative_prompt_embeds.shape) == 2:
            negative_prompt_embeds = negative_prompt_embeds.unsqueeze(1)
            negative_prompt_embeds = negative_prompt_embeds.repeat(1, max_length, 1)
        
        # 텍스트 인코더 2 처리 (OpenCLIP ViT-bigG/14)
        pooled_prompt_embeds = None
        negative_pooled_prompt_embeds = None
        
        if text_encoder_2 is not None:
            try:
                text_inputs_2 = tokenizer_2(
                    prompts,
                    padding="max_length",
                    max_length=max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                text_input_ids_2 = text_inputs_2.input_ids.to(device)
                
                uncond_input_2 = tokenizer_2(
                    negative_prompts,
                    padding="max_length",
                    max_length=max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                uncond_input_ids_2 = uncond_input_2.input_ids.to(device)
                
                # 텍스트 인코더 2 출력 (OpenCLIP)
                encoder_output_2 = text_encoder_2(text_input_ids_2)
                uncond_output_2 = text_encoder_2(uncond_input_ids_2)
                
                # 출력 형태 확인 및 처리 - CLIPTextModelOutput 객체 처리
                if hasattr(encoder_output_2, 'last_hidden_state') and hasattr(encoder_output_2, 'pooler_output'):
                    text_embeds = encoder_output_2.last_hidden_state
                    pooled_prompt_embeds = encoder_output_2.pooler_output
                    
                    negative_text_embeds = uncond_output_2.last_hidden_state
                    negative_pooled_prompt_embeds = uncond_output_2.pooler_output
                elif isinstance(encoder_output_2, tuple) and len(encoder_output_2) > 1:
                    text_embeds = encoder_output_2[0]
                    pooled_prompt_embeds = encoder_output_2[1]
                    
                    negative_text_embeds = uncond_output_2[0]
                    negative_pooled_prompt_embeds = uncond_output_2[1]
                else:
                    text_embeds = encoder_output_2 if not hasattr(encoder_output_2, 'last_hidden_state') else encoder_output_2.last_hidden_state
                    negative_text_embeds = uncond_output_2 if not hasattr(uncond_output_2, 'last_hidden_state') else uncond_output_2.last_hidden_state
                    
                    # pooled 출력이 없는 경우 시퀀스 출력의 첫 번째 토큰 사용
                    if isinstance(text_embeds, torch.Tensor):
                        pooled_prompt_embeds = text_embeds[:, 0]
                        negative_pooled_prompt_embeds = negative_text_embeds[:, 0]
                
                # 차원 확인 - pooled 출력은 (B, D) 형태여야 함
                if isinstance(pooled_prompt_embeds, torch.Tensor) and len(pooled_prompt_embeds.shape) == 3:
                    # 첫 번째 토큰만 사용 (B, S, D) -> (B, D)
                    pooled_prompt_embeds = pooled_prompt_embeds[:, 0]
                    negative_pooled_prompt_embeds = negative_pooled_prompt_embeds[:, 0]
                    
            except Exception as e:
                print(f"텍스트 인코더 2 처리 중 오류: {e}")
        
        # 텍스트 인코더 3 처리 (T5-XL)
        if text_encoder_3 is not None:
            try:
                text_inputs_3 = tokenizer_3(
                    prompts,
                    padding="max_length",
                    max_length=max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                text_input_ids_3 = text_inputs_3.input_ids.to(device)
                
                uncond_input_3 = tokenizer_3(
                    negative_prompts,
                    padding="max_length",
                    max_length=max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                uncond_input_ids_3 = uncond_input_3.input_ids.to(device)
                
                # 텍스트 인코더 3 출력 (T5)
                encoder_output_3 = text_encoder_3(text_input_ids_3)
                uncond_output_3 = text_encoder_3(uncond_input_ids_3)
                
                # T5 출력 처리
                if hasattr(encoder_output_3, 'last_hidden_state'):
                    encoder_output_3 = encoder_output_3.last_hidden_state
                    uncond_output_3 = uncond_output_3.last_hidden_state
                elif isinstance(encoder_output_3, tuple) and len(encoder_output_3) > 0:
                    encoder_output_3 = encoder_output_3[0]
                    uncond_output_3 = uncond_output_3[0]
                
                # 시퀀스 길이 확인 및 조정
                if isinstance(prompt_embeds, torch.Tensor) and isinstance(encoder_output_3, torch.Tensor):
                    if encoder_output_3.shape[1] == prompt_embeds.shape[1]:
                        # 임베딩 차원 결합
                        prompt_embeds = torch.cat([prompt_embeds, encoder_output_3], dim=-1)
                        negative_prompt_embeds = torch.cat([negative_prompt_embeds, uncond_output_3], dim=-1)
                    else:
                        # 시퀀스 길이 맞추기
                        min_seq_len = min(prompt_embeds.shape[1], encoder_output_3.shape[1])
                        prompt_embeds = prompt_embeds[:, :min_seq_len, :]
                        negative_prompt_embeds = negative_prompt_embeds[:, :min_seq_len, :]
                        encoder_output_3 = encoder_output_3[:, :min_seq_len, :]
                        uncond_output_3 = uncond_output_3[:, :min_seq_len, :]
                        
                        # 임베딩 차원 결합
                        prompt_embeds = torch.cat([prompt_embeds, encoder_output_3], dim=-1)
                        negative_prompt_embeds = torch.cat([negative_prompt_embeds, uncond_output_3], dim=-1)
            except Exception as e:
                print(f"텍스트 인코더 3 처리 중 오류: {e}")
        
        # pooled embeddings 처리 (중요: 이 부분이 timesteps_emb와 더해지는 부분)
        if pooled_prompt_embeds is None and isinstance(prompt_embeds, torch.Tensor):
            # pooled_embeds가 None인 경우 첫 번째 토큰 사용
            pooled_prompt_embeds = prompt_embeds[:, 0]  # (B, D)
            negative_pooled_prompt_embeds = negative_prompt_embeds[:, 0]  # (B, D)
        
        # pooled_embeds 차원 확인 - 반드시 (B, D) 형태여야 함
        if isinstance(pooled_prompt_embeds, torch.Tensor):
            if len(pooled_prompt_embeds.shape) != 2:
                if len(pooled_prompt_embeds.shape) == 3:  # (B, S, D) 형태인 경우
                    # 첫 번째 토큰만 사용 (B, S, D) -> (B, D)
                    pooled_prompt_embeds = pooled_prompt_embeds[:, 0]
                    negative_pooled_prompt_embeds = negative_pooled_prompt_embeds[:, 0]
                elif len(pooled_prompt_embeds.shape) == 1:  # (D) 형태인 경우
                    # 배치 차원 추가 (D) -> (B, D)
                    pooled_prompt_embeds = pooled_prompt_embeds.unsqueeze(0).repeat(batch_size, 1)
                    negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.unsqueeze(0).repeat(batch_size, 1)
        
            # pooled_embeds 차원 검증 및 수정 (SD3 표준: 2048)
            expected_pooled_dim = 2048
            if pooled_prompt_embeds.shape[-1] != expected_pooled_dim:
                if pooled_prompt_embeds.shape[-1] > expected_pooled_dim:
                    # 차원이 큰 경우 자르기
                    pooled_prompt_embeds = pooled_prompt_embeds[:, :expected_pooled_dim]
                    negative_pooled_prompt_embeds = negative_pooled_prompt_embeds[:, :expected_pooled_dim]
                else:
                    # 차원이 작은 경우 패딩
                    current_dim = pooled_prompt_embeds.shape[-1]
                    padding_size = expected_pooled_dim - current_dim
                    pooled_prompt_embeds = F.pad(pooled_prompt_embeds, (0, padding_size))
                    negative_pooled_prompt_embeds = F.pad(negative_pooled_prompt_embeds, (0, padding_size))
        
        # encoder_hidden_states 차원 검증 및 수정 (SD3 표준: 4096)
        if isinstance(prompt_embeds, torch.Tensor):
            expected_encoder_dim = 4096
            if prompt_embeds.shape[-1] != expected_encoder_dim:
                if prompt_embeds.shape[-1] > expected_encoder_dim:
                    # 차원이 큰 경우 자르기
                    prompt_embeds = prompt_embeds[:, :, :expected_encoder_dim]
                    negative_prompt_embeds = negative_prompt_embeds[:, :, :expected_encoder_dim]
                else:
                    # 차원이 작은 경우 패딩
                    current_dim = prompt_embeds.shape[-1]
                    padding_size = expected_encoder_dim - current_dim
                    prompt_embeds = F.pad(prompt_embeds, (0, padding_size))
                    negative_prompt_embeds = F.pad(negative_prompt_embeds, (0, padding_size))
        
        return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds

    def inference(self, texted_images_to_inpaint: list[TextedImage]) -> list[TextedImage]:
        """
        SD3 LoRA 모델을 사용하여 각 TextedImage의 마스크된 영역을 인페인팅

        Args:
            texted_images_to_inpaint: 인페인팅할 TextedImage 객체 리스트
                                     각 객체는 원본 이미지의 특정 bbox 주변을 center-cropped한 패치

        Returns:
            list[TextedImage]: 인페인팅된 결과가 orig에 저장된 TextedImage 객체 리스트
        """
        if len(texted_images_to_inpaint) == 0:
            print("No images to process for Model3 inference.")
            return []

        print("Loading SD3 pipeline for inference...")

        # 1. 파이프라인 및 모델 로딩
        model_id = self.model_config["model_id"]
        lora_weights_path = self.model_config["lora_path"]

        try:
            # SD3 파이프라인 로드 (기본 fp16)
            pipe = StableDiffusion3Pipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16,  # Transformer, Text Encoders는 fp16
            )

            # LoRA 가중치 로드 및 적용
            if os.path.exists(lora_weights_path):
                transformer = pipe.transformer
                transformer = PeftModel.from_pretrained(transformer, lora_weights_path)
                pipe.transformer = transformer
                print(f"LoRA weights loaded from {lora_weights_path}")
            else:
                print(f"Warning: LoRA weights not found at {lora_weights_path}, using base model")

            # 파이프라인을 GPU로 이동
            pipe.to(self.device)

            # VAE만 fp32로 변경 (재구성 품질 향상)
            pipe.vae = pipe.vae.to(dtype=torch.float32)

            # 추론 설정
            prompt = self.model_config.get("prompts", "pure black and white manga style image with no color tint, absolute grayscale, contextual manga style")
            negative_prompt = self.model_config.get("negative_prompt", "photo, realistic, color, colorful, purple, violet, sepia, any color tint, blurry")
            guidance_scale = self.model_config.get("guidance_scale", 7.5)
            num_inference_steps = self.model_config.get("inference_steps", 5)

            # 출력 디렉토리 설정
            output_dir = self.model_config.get("output_dir", "trit/datas/images/output")
            os.makedirs(output_dir, exist_ok=True)

            # 2. 각 패치 TextedImage에 대한 반복 처리
            for i, current_patch_texted_image in enumerate(tqdm(texted_images_to_inpaint, desc="Inpainting patches")):
                try:
                    # VRAM 관리를 위해 각 패치 처리 전 메모리 정리
                    torch.cuda.empty_cache()

                    # A. 입력 준비 (패치 단위)
                    # 패치 이미지와 마스크를 PIL로 변환
                    patch_orig_pil = transforms.ToPILImage()(current_patch_texted_image.orig.cpu())
                    patch_mask_pil = transforms.ToPILImage()(current_patch_texted_image.mask.cpu().squeeze(0))  # (1,H,W) -> (H,W)

                    # 마스크 이진화 (0 또는 255)
                    mask_np = np.array(patch_mask_pil)
                    mask_binary = (mask_np > 127).astype(np.uint8) * 255
                    mask_binary_pil = Image.fromarray(mask_binary, "L")

                    # 입력 이미지를 [-1,1] 범위로 정규화하고 fp32로 변환 (VAE용)
                    init_image_for_vae = transforms.ToTensor()(patch_orig_pil).unsqueeze(0)  # [1,C,H,W]
                    init_image_for_vae = (init_image_for_vae * 2.0 - 1.0).to(self.device, dtype=torch.float32)

                    # 마스크를 fp16으로 변환 (Transformer용)
                    mask_tensor_for_transformer = transforms.ToTensor()(mask_binary_pil).unsqueeze(0)  # [1,1,H,W]
                    mask_tensor_for_transformer = mask_tensor_for_transformer.to(self.device, dtype=torch.float16)

                    with torch.no_grad():
                        # B. VAE 인코딩 (fp32 연산)
                        latent_image = pipe.vae.encode(init_image_for_vae).latent_dist.sample()
                        latent_image = latent_image * pipe.vae.config.scaling_factor  # fp32

                        # C. Latent 마스크 준비
                        latent_mask_for_transformer = F.interpolate(
                            mask_tensor_for_transformer,
                            size=latent_image.shape[-2:],  # latent 해상도에 맞춤
                            mode="nearest"
                        )  # fp16

                        # D. 텍스트 임베딩 생성 (fp16 연산)
                        try:
                            # SD3 파이프라인의 내장 인코딩 메서드 사용
                            (
                                prompt_embeds,
                                negative_prompt_embeds,
                                pooled_prompt_embeds,
                                negative_pooled_prompt_embeds,
                            ) = pipe.encode_prompt(
                                prompt=prompt,
                                prompt_2=prompt,  # SD3는 여러 프롬프트 입력 받음
                                prompt_3=prompt,
                                device=self.device,
                                num_images_per_prompt=1,
                                do_classifier_free_guidance=True,
                                negative_prompt=negative_prompt,
                                negative_prompt_2=negative_prompt,
                                negative_prompt_3=negative_prompt,
                            )

                            # CFG를 위해 임베딩 결합 (None 체크 추가)
                            if negative_prompt_embeds is not None and prompt_embeds is not None:
                                prompt_embeds_cfg = torch.cat([negative_prompt_embeds, prompt_embeds])  # fp16
                            else:
                                raise ValueError("prompt_embeds 또는 negative_prompt_embeds가 None입니다.")

                            if negative_pooled_prompt_embeds is not None and pooled_prompt_embeds is not None:
                                pooled_embeds_cfg = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds])  # fp16
                            else:
                                # pooled embeds가 None인 경우 첫 번째 토큰 사용
                                pooled_embeds_cfg = prompt_embeds_cfg[:, 0]

                        except Exception as e:
                            print(f"Text encoding error for patch {i+1}: {e}")
                            # 대체 방법: 간단한 토크나이징
                            text_inputs = pipe.tokenizer(
                                [negative_prompt, prompt],
                                padding="max_length",
                                max_length=77,
                                truncation=True,
                                return_tensors="pt",
                            ).to(self.device)

                            prompt_embeds_cfg = pipe.text_encoder(text_inputs.input_ids)[0]  # fp16
                            pooled_embeds_cfg = prompt_embeds_cfg[:, 0]  # 첫 번째 토큰을 pooled로 사용

                        # E. 인페인팅을 위한 초기 Latent 준비
                        noise = torch.randn_like(latent_image, dtype=latent_image.dtype)  # fp32
                        latent_mask_for_compositing = latent_mask_for_transformer.to(
                            dtype=latent_image.dtype, device=latent_image.device
                        )  # fp32로 변환
                        
                        

                        # 마스크된 영역에만 노이즈 적용
                        latents = latent_image * (1 - latent_mask_for_compositing) + noise * latent_mask_for_compositing # resizeing 

                        # F. 디퓨전 루프 (Denoising Loop)
                        pipe.scheduler.set_timesteps(num_inference_steps, device=self.device)
                        timesteps = pipe.scheduler.timesteps

                        for t in timesteps:
                            # CFG를 위해 latents 복제하고 Transformer 입력 정밀도(fp16)로 변환
                            latent_model_input_cfg = torch.cat([latents] * 2).to(dtype=torch.float16)

                            # 타임스텝 준비
                            t_input = t.expand(latents.shape[0])

                            # Transformer로 노이즈 예측 (fp16 연산)
                            noise_pred_transformer = pipe.transformer(
                                hidden_states=latent_model_input_cfg,
                                timestep=t_input,
                                encoder_hidden_states=prompt_embeds_cfg,
                                pooled_projections=pooled_embeds_cfg,
                                return_dict=False
                            )[0]  # fp16

                            # CFG 적용
                            noise_pred_uncond, noise_pred_cond = noise_pred_transformer.chunk(2)
                            noise_pred_cfg = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

                            # 스케줄러 스텝 (fp32로 변환하여 전달)
                            latents = pipe.scheduler.step(
                                noise_pred_cfg.to(dtype=latents.dtype), t, latents, return_dict=False
                            )[0]  # fp32 유지

                            # 맥락 재주입 (중요: 마스크되지 않은 영역은 원본 유지)
                            latents = latent_image * (1 - latent_mask_for_compositing) + latents * latent_mask_for_compositing

                        # G. VAE 디코딩 (fp32 연산)
                        image_tensor_pred = pipe.vae.decode(
                            latents / pipe.vae.config.scaling_factor, return_dict=False
                        )[0]  # fp32, [-1,1] 범위
                        
                        print(image_tensor_pred.shape)

                        # H. 후처리 및 TextedImage 객체 업데이트
                        # [0,1] 범위로 정규화
                        image_tensor_pred = (image_tensor_pred / 2 + 0.5).clamp(0, 1)

                        # [1,C,H,W] -> [C,H,W]로 변환하고 원본 dtype/device로 맞춤
                        final_tensor = image_tensor_pred.squeeze(0).to(
                            dtype=current_patch_texted_image.orig.dtype,
                            device=current_patch_texted_image.orig.device
                        )

                        # 마스크 영역만 적용하기 : 원본 이미지 복사 후 마스크 영역만 인페인팅 결과로 대체
                        mask_for_compositing = current_patch_texted_image.mask.to(
                            dtype=final_tensor.dtype,
                            device=final_tensor.device
                        ) 
                        original_tensor = current_patch_texted_image.orig.clone()
                        composited_result = original_tensor * (1 - mask_for_compositing) + final_tensor * mask_for_compositing

                        # I. 패치 크기 조정 (중요: 원본 bbox 크기에 맞게 리사이즈) --재검토토
                        # 현재 패치의 _bbox (center crop된 좌표)
                        # patch_bbox = current_patch_texted_image.bboxes[0]

                        # 패치에서 실제 텍스트 영역만 추출 (center crop에서 잘린 부분 제거)
                        # cropped_final_tensor = final_tensor[patch_bbox.slice]

                        # TextedImage 객체의 orig 속성 업데이트 (cropped된 버전으로)
                        current_patch_texted_image.orig = composited_result
                        
                        # orig = final_tensor
                        
                        # a = TextedImage(orig, orig, torch.zeros(1, orig.shape[1], orig.shape[2],), current_patch_texted_image.bboxes)
                        # a.visualize(dir="trit/datas/images/output", filename=f"inpainting.png")

                        print(f"Successfully inpainted patch {i+1}/{len(texted_images_to_inpaint)}")

                except Exception as e:
                    print(f"Error during inference for patch {i+1}: {e}")
                    # 오류 발생 시 원본 패치 유지
                    continue

            # 3. 메모리 관리
            del pipe
            gc.collect()
            torch.cuda.empty_cache()

            print(f"Inference completed. Results saved to {output_dir}")
            return texted_images_to_inpaint

        except Exception as e:
            print(f"Error initializing SD3 pipeline: {e}")
            return texted_images_to_inpaint

