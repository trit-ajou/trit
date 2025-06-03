import math
import torch
import os
import numpy as np
import gc
import torch.nn.functional as F
from ..datas.TextedImage import TextedImage
from torch import FloatTensor, nn
from torchvision import transforms # 이미지 전처리를 위해 추가 임포트
from tqdm import tqdm
from pytorch_msssim import ms_ssim
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.models.transformers.transformer_sd3 import SD3Transformer2DModel
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import StableDiffusion3Pipeline
from diffusers.optimization import get_scheduler
from accelerate import Accelerator
from peft import get_peft_model, LoraConfig, PeftModel
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
def identity_collate(batch):
    return batch

class Model3(nn.Module):
    def __init__(self, model_config: dict, accelerator: Accelerator):
        super().__init__()
        self.model_config = model_config
        self.accelerator = accelerator

    def ssim_loss(self, pred_img, target_img):
        pred_img_0_1 = (pred_img.clamp(-1, 1) + 1) / 2
        target_img_0_1 = (target_img.clamp(-1, 1) + 1) / 2
        pred_img_f = pred_img_0_1.to(torch.float32)
        target_img_f = target_img_0_1.to(torch.float32)
        ssim_val_per_sample = ms_ssim(pred_img_f, target_img_f, data_range=1.0, size_average=False)
        return 1.0 - ssim_val_per_sample.mean()
    
    def lora_train(self, texted_images_for_model3: list[TextedImage]):
        if not self.accelerator.is_main_process: return
        print("Loading SD3 pipeline for LoRA training...")

        weight_dtype = torch.float16
        model_id = self.model_config["model_id"]
        lora_weights_path = self.model_config["lora_path"]
        num_epochs = self.model_config.get("epochs", 10)
        batch_size = self.model_config.get("batch_size", 4)
        
        # 파이프라인 로딩
        try:
            pipe = StableDiffusion3Pipeline.from_pretrained(
                model_id,
                torch_dtype=weight_dtype,
            )
            pipe.to(self.accelerator.device)
            print(f"SD3 pipeline loaded successfully to {self.accelerator.device}.")
            
            # 필요한 부분 로드
            vae = pipe.vae
            text_encoder = pipe.text_encoder
            text_encoder_2 = pipe.text_encoder_2
            text_encoder_3 = pipe.text_encoder_3
            transformer = pipe.transformer
            scheduler = pipe.scheduler
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
            # LoRA 설정
            lora_config = LoraConfig(
                r = self.model_config.get("lora_rank", 16),
                lora_alpha = self.model_config.get("lora_alpha", 32),
                target_modules = ["to_k", "to_q", "to_v", "to_out.0"],
                lora_dropout = 0.05,
                bias = "none",
            )
            
            transformer_lora = get_peft_model(transformer, lora_config)
            transformer_lora.print_trainable_parameters()
            
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
        
        # 랜덤 분할
        indices = torch.randperm(len(texted_images_for_model3)).tolist()
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        train_images = [texted_images_for_model3[i] for i in train_indices]
        val_images = [texted_images_for_model3[i] for i in val_indices]
        
        print(f"데이터셋 분할: 훈련 {len(train_images)}개, 검증 {len(val_images)}개")
        
        # optimizer scheduler 설정
        optimizer = torch.optim.Adafactor(transformer_lora.parameters())
        max_train_steps = num_epochs * len(train_images) // batch_size
        lr_scheduler = get_scheduler(
            "cosine",
            optimizer=optimizer,
            num_warmup_steps=4,
            num_training_steps = max_train_steps * self.accelerator.gradient_accumulation_steps,
        )
        print(f"Optimizer and scheduler set up with {max_train_steps} total training steps.")
        
        # LoRA 학습 루프
        output_dir = self.model_config.get("output_dir", "datas/images/output/model3_sd3_lora")
        os.makedirs(output_dir, exist_ok=True)
        print("Starting LoRA training...")
        
        total_batch_size = batch_size * self.accelerator.num_processes * self.accelerator.gradient_accumulation_steps
        num_update_steps_per_epoch = math.ceil(len(train_images) / self.accelerator.gradient_accumulation_steps)
        max_train_steps = num_epochs * num_update_steps_per_epoch
        print(f"Total batch size: {total_batch_size}, Max training steps: {max_train_steps}")
        
        progress_bar = tqdm(range(max_train_steps), desc="Training LoRA", disable=not self.accelerator.is_main_process)
        global_step = 0
        
        # 최고 검증 손실 추적
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            transformer_lora.train()
            epoch_loss = 0.0
            num_batches_in_epoch = 0
            
            # 훈련 루프
            for step in range(0, len(train_images), batch_size):
                batch_end = min(step + batch_size, len(train_images))
                batch = train_images[step:batch_end]
                
                with self.accelerator.accumulate(transformer_lora):
                    with torch.no_grad():
                        vae.to(self.accelerator.device)
                        
                        # 배치 처리를 위한 준비
                        batch_size_actual = len(batch)
                        original_pixel_values = torch.stack([img.orig for img in batch]).to(self.accelerator.device, dtype=weight_dtype)
                        mask_pixel_values = torch.stack([img.mask for img in batch]).to(self.accelerator.device, dtype=weight_dtype)
                        
                        # VAE 인코딩
                        target_latents = vae.encode(original_pixel_values).latent_dist.sample() * vae.config.scaling_factor
                        
                        # 마스크 준비
                        latent_mask = F.interpolate(
                            mask_pixel_values,
                            size=target_latents.shape[-2:],
                            mode="nearest"
                        )
                    
                    # 노이즈 및 타임스텝 샘플링
                    noise = torch.rand_like(target_latents, device=self.accelerator.device, dtype=weight_dtype)
                    bsz = target_latents.shape[0]
                    noise_scheduler = DDIMScheduler(
                        num_train_timesteps=self.model_config.get("max_train_timesteps", 1000),
                        beta_start=0.00085,
                        beta_end=0.012,
                        beta_schedule="scaled_linear",
                        clip_sample=False,
                        set_alpha_to_one=False,
                    )
                    timesteps = torch.randint(
                        0,
                        noise_scheduler.num_train_timesteps,
                        (bsz, ),
                        device=target_latents.device,
                    ).long()
                    
                    noisy_latents = noise_scheduler.add_noise(
                        target_latents, noise, timesteps.cpu().to(torch.int32) # type: ignore
                    )
                    
                    # 텍스트 인코딩
                    prompt = self.model_config.get("prompts", "pure black and white manga style image with no color tint, absolute grayscale, contextual manga style")
                    negative_prompt = self.model_config.get("negative_prompt", "photo, realistic, color, colorful, purple, violet, sepia, any color tint, blurry")
                    
                    with torch.no_grad():
                        text_encoder.to(self.accelerator.device)
                        text_encoder_2.to(self.accelerator.device)
                        text_encoder_3.to(self.accelerator.device)
                        
                        # 텍스트 임베딩 생성
                        prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = self._encode_prompt(
                            prompt, negative_prompt, tokenizer, tokenizer_2, tokenizer_3,
                            text_encoder, text_encoder_2, text_encoder_3, self.accelerator.device
                        )
                        
                        # 임베딩 준비
                        prompt_embeds_full = torch.cat([negative_prompt_embeds.to(weight_dtype), prompt_embeds.to(weight_dtype)], dim=0)
                        pooled_embeddings_full = torch.cat([negative_pooled_prompt_embeds.to(weight_dtype), pooled_prompt_embeds.to(weight_dtype)], dim=0)
                        
                        text_encoder.to("cpu")
                        text_encoder_2.to("cpu")
                        text_encoder_3.to("cpu")
                    
                    # 노이즈 예측 및 손실 계산
                    latent_model_input = torch.cat([noisy_latents] * 2)
                    timestep_tensor = timesteps.to(self.accelerator.device)
                    
                    # 트랜스포머 모델로 노이즈 예측
                    noise_pred = transformer_lora(
                        hidden_states = latent_model_input,
                        timestep = timestep_tensor,
                        encoder_hidden_states = prompt_embeds_full,
                        pooled_projections = pooled_embeddings_full,
                        return_dict=False
                    )[0]
                    
                    guidance_scale = self.model_config.get("guidance_scale", 7.5)
                    # CFG
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                    
                    # 원본 노이즈와 예측 노이즈 간의 mse 손실 계산
                    loss = F.mse_loss(noise_pred, noise, reduction="mean")
                    
                    # 손실 역전파 및 optimizer step
                    self.accelerator.backward(loss)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    
                    # 손실 기록
                    epoch_loss += loss.detach().item()
                    num_batches_in_epoch += 1
                    
                    # 진행상황 업데이트
                    progress_bar.update(1)
                    global_step += 1
            
            # 에폭 종료 후 검증 손실 계산
            if num_batches_in_epoch > 0:
                avg_train_loss = epoch_loss / num_batches_in_epoch
                print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}")
                
                # 검증 손실 계산
                val_loss = self._calculate_validation_loss(
                    transformer_lora, vae, text_encoder, text_encoder_2, text_encoder_3,
                    tokenizer, tokenizer_2, tokenizer_3, noise_scheduler,
                    val_images, weight_dtype
                )
                print(f"Epoch {epoch+1}/{num_epochs} - Validation Loss: {val_loss:.4f}")
                
                # 에폭마다 모델 저장
                if self.accelerator.is_main_process:
                    unwrapped_model = self.accelerator.unwrap_model(transformer_lora)
                    
                    # 검증 손실이 개선되면 best_model 저장
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_save_path = os.path.join(output_dir, "best_model")
                        os.makedirs(best_save_path, exist_ok=True)
                        unwrapped_model.save_pretrained(best_save_path)
                        print(f"New best model saved with validation loss: {best_val_loss:.4f}")

    def _calculate_validation_loss(self, transformer_lora, vae, text_encoder, text_encoder_2, text_encoder_3,
                                 tokenizer, tokenizer_2, tokenizer_3, noise_scheduler,
                                 val_images, weight_dtype):
        """
        검증 세트에 대한 손실 계산
        
        Args:
            transformer_lora: LoRA가 적용된 트랜스포머 모델
            vae: VAE 모델
            text_encoder, text_encoder_2, text_encoder_3: 텍스트 인코더 모델들
            tokenizer, tokenizer_2, tokenizer_3: 토크나이저들
            noise_scheduler: 노이즈 스케줄러
            val_images: 검증 이미지 리스트
            weight_dtype: 가중치 데이터 타입
        
        Returns:
            float: 평균 검증 손실
        """
        # 평가 모드로 전환
        transformer_lora.eval()
        device = self.accelerator.device
        
        # 검증 손실 계산을 위한 변수 초기화
        total_val_loss = 0.0
        num_val_batches = 0
        batch_size = min(4, len(val_images))  # 검증에는 더 작은 배치 크기 사용
        
        # 프롬프트 설정
        prompt = self.model_config.get("prompts", "pure black and white manga style image with no color tint, absolute grayscale, contextual manga style")
        negative_prompt = self.model_config.get("negative_prompt", "photo, realistic, color, colorful, purple, violet, sepia, any color tint, blurry")
        guidance_scale = self.model_config.get("guidance_scale", 7.5)
        
        with torch.no_grad():
            for step in range(0, len(val_images), batch_size):
                # 배치 준비
                batch_end = min(step + batch_size, len(val_images))
                batch = val_images[step:batch_end]
                
                # 이미지와 마스크 텐서 준비
                original_pixel_values = torch.stack([img.orig for img in batch]).to(device, dtype=weight_dtype)
                mask_pixel_values = torch.stack([img.mask for img in batch]).to(device, dtype=weight_dtype)
                
                # 1. VAE 인코딩: 이미지를 latent 공간으로 변환
                vae.to(device)
                target_latents = vae.encode(original_pixel_values).latent_dist.sample() * vae.config.scaling_factor
                
                # 2. 마스크 다운샘플링: latent 공간 크기에 맞게 조정
                latent_mask = F.interpolate(
                    mask_pixel_values,
                    size=target_latents.shape[-2:],
                    mode="nearest"
                )
                
                # 3. 노이즈 및 타임스텝 샘플링
                noise = torch.rand_like(target_latents, device=device, dtype=weight_dtype)
                bsz = target_latents.shape[0]
                timesteps = torch.randint(
                    0,
                    noise_scheduler.num_train_timesteps,
                    (bsz, ),
                    device=target_latents.device,
                ).long()
                
                # 4. 노이즈 추가: 디퓨전 모델의 forward process
                noisy_latents = noise_scheduler.add_noise(
                    target_latents, noise, timesteps.cpu().to(torch.int32)
                )
                
                # 5. 텍스트 인코딩: 프롬프트를 임베딩으로 변환
                text_encoders = [text_encoder, text_encoder_2, text_encoder_3]
                tokenizers = [tokenizer, tokenizer_2, tokenizer_3]
                
                # 모든 텍스트 인코더를 GPU로 이동
                for encoder in text_encoders:
                    if encoder is not None:
                        encoder.to(device)
                
                # 텍스트 임베딩 생성
                prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = self._encode_prompt(
                    prompt, negative_prompt, tokenizer, tokenizer_2, tokenizer_3,
                    text_encoder, text_encoder_2, text_encoder_3, device
                )
                
                # 임베딩 준비 (CFG를 위해 negative와 positive 임베딩 결합)
                prompt_embeds_full = torch.cat([negative_prompt_embeds.to(weight_dtype), prompt_embeds.to(weight_dtype)], dim=0)
                pooled_embeddings_full = torch.cat([negative_pooled_prompt_embeds.to(weight_dtype), pooled_prompt_embeds.to(weight_dtype)], dim=0)
                
                # 텍스트 인코더를 CPU로 이동하여 GPU 메모리 확보
                for encoder in text_encoders:
                    if encoder is not None:
                        encoder.to("cpu")
                
                # 6. 노이즈 예측: 트랜스포머 모델로 노이즈 예측
                latent_model_input = torch.cat([noisy_latents] * 2)  # CFG를 위해 입력 복제
                timestep_tensor = timesteps.to(device)
                
                # 트랜스포머 모델 실행
                noise_pred = transformer_lora(
                    hidden_states=latent_model_input,
                    timestep=timestep_tensor,
                    encoder_hidden_states=prompt_embeds_full,
                    pooled_projections=pooled_embeddings_full,
                    return_dict=False
                )[0]
                
                # 7. CFG 적용: 조건부 및 무조건부 예측 결합
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                
                # 8. 손실 계산: 예측된 노이즈와 실제 노이즈 간의 MSE
                val_loss = F.mse_loss(noise_pred, noise, reduction="mean")
                total_val_loss += val_loss.item()
                num_val_batches += 1
        
        # 평균 검증 손실 계산
        avg_val_loss = total_val_loss / num_val_batches if num_val_batches > 0 else float('inf')
        
        # 훈련 모드로 복귀
        transformer_lora.train()
        return avg_val_loss

    def _encode_prompt(self, prompt, negative_prompt, tokenizer, tokenizer_2, tokenizer_3,
                     text_encoder, text_encoder_2, text_encoder_3, device):
        """
        텍스트 인코딩 헬퍼 메서드

        Args:
            prompt: 긍정적 프롬프트
            negative_prompt: 부정적 프롬프트
            tokenizer, tokenizer_2, tokenizer_3: 토크나이저들
            text_encoder, text_encoder_2, text_encoder_3: 텍스트 인코더 모델들
            device: 계산 장치

        Returns:
            tuple: (prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds)
        """
        # 공통 max_length 설정 (모든 토크나이저에서 가장 작은 값 사용)
        max_length = 77  # SD3의 표준 시퀀스 길이

        # 텍스트 인코더 1
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(device)

        # 네거티브 프롬프트
        uncond_input = tokenizer(
            negative_prompt,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
        uncond_input_ids = uncond_input.input_ids.to(device)

        # 텍스트 인코더 출력 처리
        try:
            # SD3 모델은 text_encoder가 다양한 출력 형식을 가질 수 있음
            text_encoder_output = text_encoder(text_input_ids, output_hidden_states=True)
            if isinstance(text_encoder_output, tuple):
                prompt_embeds = text_encoder_output[0]
            else:
                prompt_embeds = text_encoder_output.hidden_states[-1]

            uncond_output = text_encoder(uncond_input_ids, output_hidden_states=True)
            if isinstance(uncond_output, tuple):
                negative_prompt_embeds = uncond_output[0]
            else:
                negative_prompt_embeds = uncond_output.hidden_states[-1]
        except Exception as e:
            print(f"텍스트 인코더 1 처리 중 오류: {e}")
            # 기본 출력 시도
            prompt_embeds = text_encoder(text_input_ids)[0]
            negative_prompt_embeds = text_encoder(uncond_input_ids)[0]
        
        # 텍스트 인코더 2 (있는 경우)
        pooled_prompt_embeds = None
        negative_pooled_prompt_embeds = None
        
        if text_encoder_2 is not None:
            try:
                text_inputs_2 = tokenizer_2(
                    prompt,
                    padding="max_length",
                    max_length=max_length,  # 공통 max_length 사용
                    truncation=True,
                    return_tensors="pt",
                )
                text_input_ids_2 = text_inputs_2.input_ids.to(device)
                
                # 텍스트 인코더 2 출력 처리
                text_encoder_2_output = text_encoder_2(text_input_ids_2, output_hidden_states=True)
                if isinstance(text_encoder_2_output, tuple) and len(text_encoder_2_output) >= 2:
                    prompt_embeds_2, pooled_prompt_embeds = text_encoder_2_output[:2]
                elif hasattr(text_encoder_2_output, 'hidden_states'):
                    prompt_embeds_2 = text_encoder_2_output.hidden_states[-1]
                    pooled_prompt_embeds = text_encoder_2_output.pooler_output if hasattr(text_encoder_2_output, 'pooler_output') else None
                else:
                    prompt_embeds_2 = text_encoder_2_output[0]
                    pooled_prompt_embeds = None
                
                # 네거티브 프롬프트
                uncond_input_2 = tokenizer_2(
                    negative_prompt,
                    padding="max_length",
                    max_length=max_length,  # 공통 max_length 사용
                    truncation=True,
                    return_tensors="pt",
                )
                uncond_input_ids_2 = uncond_input_2.input_ids.to(device)
                
                uncond_output_2 = text_encoder_2(uncond_input_ids_2, output_hidden_states=True)
                if isinstance(uncond_output_2, tuple) and len(uncond_output_2) >= 2:
                    negative_prompt_embeds_2, negative_pooled_prompt_embeds = uncond_output_2[:2]
                elif hasattr(uncond_output_2, 'hidden_states'):
                    negative_prompt_embeds_2 = uncond_output_2.hidden_states[-1]
                    negative_pooled_prompt_embeds = uncond_output_2.pooler_output if hasattr(uncond_output_2, 'pooler_output') else None
                else:
                    negative_prompt_embeds_2 = uncond_output_2[0]
                    negative_pooled_prompt_embeds = None
                
                # 임베딩 결합
                prompt_embeds = torch.cat([prompt_embeds, prompt_embeds_2], dim=-1)
                negative_prompt_embeds = torch.cat([negative_prompt_embeds, negative_prompt_embeds_2], dim=-1)
                
            except Exception as e:
                print(f"텍스트 인코더 2 처리 중 오류: {e}")
                # 오류 발생 시 pooled_embeds는 None으로 유지
        
        # 텍스트 인코더 3 (있는 경우)
        if text_encoder_3 is not None:
            try:
                text_inputs_3 = tokenizer_3(
                    prompt,
                    padding="max_length",
                    max_length=max_length,  # 공통 max_length 사용
                    truncation=True,
                    return_tensors="pt",
                )
                text_input_ids_3 = text_inputs_3.input_ids.to(device)
                
                # 텍스트 인코더 3 출력 처리
                text_encoder_3_output = text_encoder_3(text_input_ids_3, output_hidden_states=True)
                if isinstance(text_encoder_3_output, tuple):
                    prompt_embeds_3 = text_encoder_3_output[0]
                else:
                    prompt_embeds_3 = text_encoder_3_output.hidden_states[-1]
                
                # 네거티브 프롬프트
                uncond_input_3 = tokenizer_3(
                    negative_prompt,
                    padding="max_length",
                    max_length=max_length,  # 공통 max_length 사용
                    truncation=True,
                    return_tensors="pt",
                )
                uncond_input_ids_3 = uncond_input_3.input_ids.to(device)
                
                uncond_output_3 = text_encoder_3(uncond_input_ids_3, output_hidden_states=True)
                if isinstance(uncond_output_3, tuple):
                    negative_prompt_embeds_3 = uncond_output_3[0]
                else:
                    negative_prompt_embeds_3 = uncond_output_3.hidden_states[-1]
                
                # 임베딩 결합 - 차원 확인 후 결합
                if prompt_embeds_3.shape[1] == prompt_embeds.shape[1]:  # 시퀀스 길이가 같은 경우
                    prompt_embeds = torch.cat([prompt_embeds, prompt_embeds_3], dim=-1)
                    negative_prompt_embeds = torch.cat([negative_prompt_embeds, negative_prompt_embeds_3], dim=-1)
                else:
                    print(f"텍스트 인코더 3 시퀀스 길이 불일치: {prompt_embeds_3.shape[1]} vs {prompt_embeds.shape[1]}")
                    # 시퀀스 길이를 맞춰서 결합
                    min_seq_len = min(prompt_embeds.shape[1], prompt_embeds_3.shape[1])
                    prompt_embeds = torch.cat([prompt_embeds[:, :min_seq_len], prompt_embeds_3[:, :min_seq_len]], dim=-1)
                    negative_prompt_embeds = torch.cat([negative_prompt_embeds[:, :min_seq_len], negative_prompt_embeds_3[:, :min_seq_len]], dim=-1)
                
            except Exception as e:
                print(f"텍스트 인코더 3 처리 중 오류: {e}")
        
        # pooled_embeds가 None인 경우 기본값 생성
        if pooled_prompt_embeds is None:
            # 기본 pooled 임베딩 생성 (첫 번째 토큰의 임베딩 사용)
            pooled_prompt_embeds = prompt_embeds[:, 0]
            negative_pooled_prompt_embeds = negative_prompt_embeds[:, 0]
        
        return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds

    def _fix_bbox_issues(self, texted_image: TextedImage, image_index: int) -> TextedImage:
        """
        BBox 관련 문제를 사전에 검증하고 수정하는 메서드

        Args:
            texted_image: 검증할 TextedImage 객체
            image_index: 이미지 인덱스 (디버깅용)

        Returns:
            TextedImage: 수정된 TextedImage 객체
        """
        try:
            # 이미지 크기 가져오기
            _, H, W = texted_image.orig.shape

            # BBox 리스트 검증 및 수정
            valid_bboxes = []
            for j, bbox in enumerate(texted_image.bboxes):
                try:
                    # BBox가 올바른 형태인지 확인
                    if hasattr(bbox, '__len__') and len(bbox) == 4:
                        # 4개 요소를 가진 경우 언패킹 테스트
                        x1, y1, x2, y2 = bbox

                        # 숫자인지 확인
                        if all(isinstance(val, (int, float)) for val in [x1, y1, x2, y2]):
                            # 좌표 범위 검증 및 수정
                            x1 = max(0, min(int(x1), W-1))
                            y1 = max(0, min(int(y1), H-1))
                            x2 = max(x1+1, min(int(x2), W))
                            y2 = max(y1+1, min(int(y2), H))

                            # 유효한 크기인지 확인 (최소 1x1 픽셀)
                            if x2 > x1 and y2 > y1:
                                # 수정된 BBox 생성
                                from ..datas.Utils import BBox
                                corrected_bbox = BBox(x1, y1, x2, y2)
                                valid_bboxes.append(corrected_bbox)

                                # 원본과 다르면 경고 출력
                                if (x1, y1, x2, y2) != (bbox.x1, bbox.y1, bbox.x2, bbox.y2):
                                    print(f"Warning: Corrected BBox {j} in image {image_index+1}: {bbox} -> {corrected_bbox}")
                            else:
                                print(f"Warning: Invalid BBox size in image {image_index+1}, BBox {j}: {bbox}")
                        else:
                            print(f"Warning: Invalid BBox coordinates in image {image_index+1}, BBox {j}: {bbox}")
                    else:
                        print(f"Warning: Invalid BBox format in image {image_index+1}, BBox {j}: {bbox}")
                except Exception as bbox_error:
                    print(f"Error validating BBox {j} in image {image_index+1}: {bbox_error}")
                    continue

            # 원본 TextedImage의 BBox 리스트를 직접 수정
            if len(valid_bboxes) != len(texted_image.bboxes):
                print(f"Fixed {len(texted_image.bboxes) - len(valid_bboxes)} invalid BBoxes in image {image_index+1}")
                texted_image.bboxes = valid_bboxes

            return texted_image

        except Exception as e:
            print(f"Error fixing BBoxes in image {image_index+1}: {e}")
            # 모든 BBox를 제거하고 빈 리스트로 설정
            texted_image.bboxes = []
            return texted_image



    def inference(self, texted_images_for_model3: list[TextedImage]) -> list[TextedImage]:
        """
        SD3 LoRA 모델을 사용하여 마스크된 영역을 인페인팅하고 결과를 TextedImage 리스트로 반환

        Args:
            texted_images_for_model3: 인페인팅할 TextedImage 객체 리스트

        Returns:
            list[TextedImage]: 인페인팅된 결과가 orig에 저장된 TextedImage 객체 리스트
        """
        if len(texted_images_for_model3) == 0:
            print("No images to process for Model3 inference.")
            return []

        if not self.accelerator.is_main_process:
            return texted_images_for_model3  # 메인 프로세스가 아닌 경우 원본 반환

        print("Loading SD3 pipeline for inference...")

        # 설정 및 모델 경로
        weight_dtype = torch.float16
        model_id = self.model_config["model_id"]
        lora_weights_path = self.model_config["lora_path"]
        output_dir = self.model_config.get("output_dir", "datas/images/output/model3_sd3_lora_inference")
        os.makedirs(output_dir, exist_ok=True)
        
        # 결과를 저장할 리스트
        inpainted_images = []
        
        try:
            # SD3 파이프라인 로드
            pipe = StableDiffusion3Pipeline.from_pretrained(
                model_id,
                torch_dtype=weight_dtype,
            )
            
            # LoRA 가중치 로드
            transformer = pipe.transformer
            transformer = PeftModel.from_pretrained(transformer, lora_weights_path)
            pipe.transformer = transformer
            
            # 파이프라인을 GPU로 이동
            pipe.to(self.accelerator.device)
            print(f"SD3 pipeline with LoRA weights loaded successfully to {self.accelerator.device}.")
            
            # 추론 설정
            prompt = self.model_config.get("prompts", "pure black and white manga style image with no color tint, absolute grayscale, contextual manga style")
            negative_prompt = self.model_config.get("negative_prompt", "photo, realistic, color, colorful, purple, violet, sepia, any color tint, blurry")
            guidance_scale = self.model_config.get("guidance_scale", 7.5)
            num_inference_steps = self.model_config.get("num_inference_steps", 3)



            # 결과 저장을 위한 디렉토리 생성
            output_dir = "trit/datas/images/output"
            os.makedirs(output_dir, exist_ok=True)
            inpainted_images = []

            # 각 이미지에 대해 인페인팅 수행
            for i, original_texted_image in enumerate(tqdm(texted_images_for_model3, desc="Inpainting images")):
                try:
                    # 마스크 차원 문제 수정 - ImageLoader에서 (1, w, h) 대신 (1, h, w)로 생성되어야 함
                    if original_texted_image.mask.shape[1:] != original_texted_image.orig.shape[1:]:
                        # 마스크를 올바른 차원으로 수정
                        original_texted_image.mask = original_texted_image.mask.permute(0, 2, 1)

                    # BBox 검증 및 수정 - inference 시작 전에 미리 처리
                    original_texted_image = self._fix_bbox_issues(original_texted_image, i)

                    # 원본 이미지, 마스크 준비
                    try:
                        original_pil, text_pil, mask_pil = original_texted_image._to_pil()
                    except Exception as to_pil_error:
                        print(f"Warning: _to_pil() failed for image {i+1}, retrying with empty BBoxes: {to_pil_error}")
                        # 강제로 빈 BBox로 재시도
                        original_bboxes = original_texted_image.bboxes  # 백업
                        original_texted_image.bboxes = []  # 임시로 빈 리스트
                        original_pil, text_pil, mask_pil = original_texted_image._to_pil()
                        original_texted_image.bboxes = original_bboxes  # 복원
                    
                    # 마스크 이진화 (0 또는 1로 변환)
                    mask_np = np.array(mask_pil)
                    mask_binary = (mask_np > 127).astype(np.uint8) * 255
                    mask_binary_pil = Image.fromarray(mask_binary, "L")
                    
                    # 인페인팅 수행
                    with torch.no_grad():
                        # 이미지와 마스크를 모델 입력 형식으로 변환
                        init_image = transforms.ToTensor()(original_pil).unsqueeze(0).to(self.accelerator.device, dtype=weight_dtype)
                        mask_image = transforms.ToTensor()(mask_binary_pil).unsqueeze(0).to(self.accelerator.device, dtype=weight_dtype)
                        
                        # VAE 인코딩
                        latent_image = pipe.vae.encode(init_image).latent_dist.sample() * pipe.vae.config.scaling_factor
                        
                        # 마스크 다운샘플링
                        latent_mask = F.interpolate(
                            mask_image,
                            size=latent_image.shape[-2:],
                            mode="nearest"
                        )
                        
                        # 텍스트 임베딩 생성 - 파이프라인의 내장 메서드 사용
                        try:
                            # SD3 파이프라인의 내장 인코딩 메서드 사용
                            (
                                prompt_embeds,
                                negative_prompt_embeds,
                                pooled_prompt_embeds,
                                negative_pooled_prompt_embeds,
                            ) = pipe.encode_prompt(
                                prompt=prompt,
                                prompt_2=prompt,  # 필수 매개변수
                                prompt_3=prompt,  # 필수 매개변수
                                device=self.accelerator.device,
                                num_images_per_prompt=1,
                                do_classifier_free_guidance=True,
                                negative_prompt=negative_prompt,
                                negative_prompt_2=negative_prompt,
                                negative_prompt_3=negative_prompt,
                            )

                            # 임베딩 준비 (None 체크 추가)
                            if negative_prompt_embeds is not None and prompt_embeds is not None:
                                prompt_embeds_full = torch.cat([negative_prompt_embeds, prompt_embeds])
                            else:
                                raise ValueError("prompt_embeds 또는 negative_prompt_embeds가 None입니다.")

                            if negative_pooled_prompt_embeds is not None and pooled_prompt_embeds is not None:
                                pooled_prompt_embeds_full = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds])
                            else:
                                # pooled embeds가 None인 경우 첫 번째 토큰 사용
                                pooled_prompt_embeds_full = prompt_embeds_full[:, 0]

                        except Exception as e:
                            print(f"이미지 {i+1} 텍스트 인코딩 중 오류: {e}")
                            print("대체 인코딩 방법을 시도합니다.")

                            # 대체 방법: 간단한 토크나이징과 인코딩
                            with torch.no_grad():
                                # 간단한 토크나이징
                                text_inputs = pipe.tokenizer(
                                    [negative_prompt, prompt],
                                    padding="max_length",
                                    max_length=77,
                                    truncation=True,
                                    return_tensors="pt",
                                ).to(self.accelerator.device)

                                # 텍스트 인코더 1만 사용
                                prompt_embeds_full = pipe.text_encoder(text_inputs.input_ids)[0]
                                pooled_prompt_embeds_full = prompt_embeds_full[:, 0]  # 첫 번째 토큰을 pooled로 사용
                        
                        # 타임스텝 설정 - 파이프라인의 기존 스케줄러 사용
                        pipe.scheduler.set_timesteps(num_inference_steps, device=self.accelerator.device)
                        timesteps = pipe.scheduler.timesteps
                        
                        # 초기 노이즈 생성 (마스크 영역만)
                        current_device = latent_image.device
                        noise = torch.randn_like(latent_image, device=current_device, dtype=weight_dtype)
                        latents = latent_image * (1 - latent_mask) + noise * latent_mask
                        
                        # 디퓨전 과정 (노이즈 → 이미지)
                        for t in tqdm(timesteps, desc=f"Inpainting image {i+1}", leave=False):
                            # 입력 준비 (CFG를 위해 복제)
                            latent_model_input = torch.cat([latents] * 2)

                            # 현재 타임스텝 - 스케줄러의 타임스텝을 그대로 사용
                            timestep = t.expand(latents.shape[0])

                            # 노이즈 예측
                            noise_pred = pipe.transformer(
                                hidden_states=latent_model_input,
                                timestep=timestep,
                                encoder_hidden_states=prompt_embeds_full,
                                pooled_projections=pooled_prompt_embeds_full,
                                return_dict=False
                            )[0]

                            # CFG 적용
                            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                            # 스케줄러 스텝 - 타임스텝을 그대로 전달
                            latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0].to(weight_dtype)

                            # 마스크되지 않은 영역은 원본 유지 (인페인팅의 핵심)
                            latents = latent_image * (1 - latent_mask) + latents * latent_mask
                        
                        # VAE 디코딩하여 이미지 생성
                        image_tensor = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False)[0]
                        
                        # 이미지 후처리
                        image_tensor = (image_tensor / 2 + 0.5).clamp(0, 1)
                        final_image_tensor_0_1 = image_tensor[0]
                        
                        # PIL 이미지로 변환
                        final_inpainted_pil = Image.fromarray(
                            (final_image_tensor_0_1.cpu().permute(1,2,0).numpy() * 255).astype(np.uint8), 
                            "RGB"
                        )
                        
                        # 시각화 및 저장
                        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
                        axes[0].imshow(original_pil); axes[0].set_title("Original"); axes[0].axis("off")
                        axes[1].imshow(text_pil); axes[1].set_title("Text Input"); axes[1].axis("off")
                        axes[2].imshow(mask_pil, cmap='gray'); axes[2].set_title("Mask"); axes[2].axis("off")
                        axes[3].imshow(final_inpainted_pil); axes[3].set_title("SD3 LoRA Inpainted Result"); axes[3].axis("off")

                        plt.tight_layout()
                        save_filename = os.path.join(output_dir, f"sd3_lora_inpainted_result_{i}.png")
                        plt.savefig(save_filename)
                        plt.close(fig)
                        
                        # 원본 TextedImage 객체의 orig 필드를 직접 업데이트
                        # 인페인팅 결과를 텐서로 변환하여 orig에 저장
                        inpainted_tensor = transforms.ToTensor()(final_inpainted_pil).to(original_texted_image.orig.device)
                        original_texted_image.orig = inpainted_tensor

                        inpainted_images.append(original_texted_image)
                        print(f"Successfully inpainted image {i+1}")

                except Exception as e:
                    print(f"Error during SD3 inference for image {i+1}: {e}")

                    # 오류 발생 시 원본 이미지의 마스크 영역에 노이즈를 채워 시각적으로 표시
                    try:
                        original_pil, text_pil, mask_pil = original_texted_image._to_pil()
                    except Exception as error_to_pil:
                        print(f"Warning: _to_pil() failed during error handling for image {i+1}, using empty BBoxes")
                        # 최후의 수단: 빈 BBox로 시도
                        original_bboxes = original_texted_image.bboxes  # 백업
                        original_texted_image.bboxes = []  # 임시로 빈 리스트
                        original_pil, text_pil, mask_pil = original_texted_image._to_pil()
                        original_texted_image.bboxes = original_bboxes  # 복원
                    
                    # 노이즈 이미지를 원본 이미지 크기와 동일하게 생성 (PIL RGB)
                    error_noise_pil = Image.fromarray(
                        (torch.rand(3, original_pil.size[1], original_pil.size[0]) * 255).byte().permute(1,2,0).cpu().numpy(), 
                        "RGB"
                    )
                    
                    # 마스크된 부분에 노이즈를, 마스크되지 않은 부분에 원본을 합성 (PIL Composite)
                    error_inpainted_pil = Image.composite(error_noise_pil, original_pil, mask_pil.convert('L'))
                    
                    # 시각화 (오류 표시)
                    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
                    axes[0].imshow(original_pil); axes[0].set_title("Original"); axes[0].axis("off")
                    axes[1].imshow(text_pil); axes[1].set_title("Text Input (Error)"); axes[1].axis("off")
                    axes[2].imshow(mask_pil, cmap='gray'); axes[2].set_title("Mask"); axes[2].axis("off")
                    axes[3].imshow(error_inpainted_pil); axes[3].set_title("SD3 LoRA Inpainted Result (Error)"); axes[3].axis("off")

                    plt.tight_layout()
                    save_filename = os.path.join(output_dir, f"sd3_lora_inpainted_result_error_{i}.png")
                    plt.savefig(save_filename)
                    plt.close(fig)
                    
                    # 오류 발생 시 에러 이미지를 원본 TextedImage에 저장
                    error_tensor = transforms.ToTensor()(error_inpainted_pil).to(original_texted_image.orig.device)
                    original_texted_image.orig = error_tensor
                    inpainted_images.append(original_texted_image)
            
            # 모든 처리 완료 후 파이프라인 메모리 해제
            del pipe
            gc.collect()
            torch.cuda.empty_cache()
            
            print(f"Inference completed. Results saved to {output_dir}")
            return inpainted_images
        
        except Exception as e:
            print(f"Error in SD3 inference pipeline: {e}")
            # 오류 발생 시 원본 이미지 리스트 반환
            return texted_images_for_model3



