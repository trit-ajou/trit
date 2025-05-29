from torch import nn
import torch
from diffusers import AutoPipelineForInpainting, StableDiffusionXLPipeline
from peft import LoraConfig, get_peft_model
import torch.nn.functional as F
from accelerate import Accelerator
from tqdm import tqdm
from ..datas.TextedImage import TextedImage
from pytorch_msssim import ms_ssim

class Model3(nn.Module):
    """Masked Inpainting Model
    config: 모델 설정 정보 dictionary
    
    """

    def __init__(self, model_config: dict, lora_config: LoraConfig):
        super().__init__()
        self.model_config = model_config
        
    def forward(self, x): # 사용 x
        return x
    
    def ssim_loss(self, pred_img, target_img, mask=None):
        """
        SSIM 기반 손실 계산. 예측된 이미지와 타겟 이미지 간의 (1 - MS-SSIM)을 계산.
        마스크가 제공되면 해당 영역에 대해서만 손실을 계산하거나 가중치를 둘 수 있음.
        (여기서는 전체 이미지에 대해 계산 후 마스크된 영역에 집중하는 방식으로 아래 loss 계산부에서 처리)
        pred_img와 target_img는 [0, 1] 범위여야 함.
        """
        # 데이터 범위를 pytorch_msssim의 기본 기대값인 [0, 1]로 조정
        # VAE 디코딩 후 [-1, 1] 범위라면 (x + 1) / 2 로 변환 필요
        pred_img_0_1 = (pred_img.clamp(-1, 1) + 1) / 2
        target_img_0_1 = (target_img.clamp(-1, 1) + 1) / 2

        # size_average=True (최신 버전에서는 reduction='elementwise_mean') 대신 배치 각 아이템에 대해 계산 후 평균
        # ms_ssim의 reduction 파라미터는 최신 버전에 없을 수 있음 -> 직접 mean()
        ssim_val_per_item = ms_ssim(pred_img_0_1, target_img_0_1, data_range=1.0, size_average=False) # (B,)
        return 1.0 - ssim_val_per_item.mean() # 배치 평균된 (1-SSIM) 값
    
    
    def lora_train(self, texted_images_for_model3: list[TextedImage], lora_config: LoraConfig):
        accelerator = Accelerator(
            mixed_precision="fp16",
            gradient_accumulation_steps= self.model_config["gradient_accumulation_steps"],
        )
        # 모델 파이프라인 로드
        self.pipe = AutoPipelineForInpainting.from_pretrained(self.model_config["model_id"], 
                                                              torch_dtype=torch.float16, 
                                                              variant="fp16", 
                                                              # low_cpu_mem_usage=True # 메모리 부족 시 고려
                                                              ).to(accelerator.device)
        
        
        # VAE, UNet, Text Encoder, Scheduler, Tokenizer를 파이프라인에서 분리
        vae = self.pipe.vae
        unet = self.pipe.unet
        text_encoder = self.pipe.text_encoder # SDXL은 2개의 Text Encoder를 가질 수 있음 (pipe.text_encoder, pipe.text_encoder_2)
        text_encoder_2 = getattr(self.pipe, 'text_encoder_2', None)
        tokenizer = self.pipe.tokenizer # SDXL은 2개의 Tokenizer를 가질 수 있음 (pipe.tokenizer, pipe.tokenizer_2)
        tokenizer_2 = getattr(self.pipe, 'tokenizer_2', None)
        scheduler = self.pipe.scheduler
        
        #Lora 적용
        unet.requires_grad_(False) # LoRA 적용 전 UNet 동결
        unet = get_peft_model(unet, lora_config)
        if accelerator.is_main_process:
            unet.print_trainable_parameters()
        
        # 옵티마이저 설정
        optimizer = torch.optim.AdamW(self.pipe.unet.parameters(), lr=self.model_config["lr"])
        
        # Accelerator로 모델, 옵티마이저 준비
        unet, optimizer = accelerator.prepare(unet, optimizer)
        if text_encoder_2:
            text_encoder, text_encoder_2, vae = accelerator.prepare(text_encoder, text_encoder_2, vae)
        else:
            text_encoder, vae = accelerator.prepare(text_encoder, vae)
        
        
        # 학습 모드 설정
        unet.train()
        text_encoder.eval() # LoRA가 텍스트 인코더에도 적용된다면 train 모드, 아니면 eval 모드
        if text_encoder_2: text_encoder_2.eval()
        vae.eval()
        
        global_step = 0
        
        for epoch in range(self.model_config["epochs"]):
            epoch_total_loss = 0
            epoch_mse_loss_masked = 0
            epoch_ssim_loss_masked = 0
            
            for epoch in range(self.model_config["epochs"]):
                
                num_batches = (len(texted_images_for_model3) + self.model_config["batch_size"] - 1) // self.model_config["batch_size"]
                batch_iterator = tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{self.model_config['epochs']}",
                                      disable=not accelerator.is_main_process)
                
                for batch_idx_iter in batch_iterator:
                    with accelerator.accumulate(unet):
                        start_idx = batch_idx_iter * self.model_config["batch_size"]
                        end_idx = min(start_idx + self.model_config["batch_size"], len(texted_images_for_model3))
                        batch_items = texted_images_for_model3[start_idx:end_idx]
                        
                        if not batch_items: continue
                        
                        # --- 2. 데이터 준비 및 전처리 ---
                        # TextedImage 객체에서 'timg'를 원본 이미지로, 'mask'를 마스크로 사용
                        # 입력 이미지가 [0,1] 범위라고 가정하고, VAE 입력을 위해 [-1,1]로 변환
                        original_pixel_values = torch.stack([item.timg for item in batch_items]).to(accelerator.device, dtype=torch.float32)
                        original_pixel_values = original_pixel_values * 2.0 - 1.0 # [0,1] -> [-1,1]
                        
                        # 마스크 (B, 1, H, W), 1이 복원 대상 영역 (TextedImage.mask가 이미 이 형태라고 명시됨)
                        masks = torch.stack([item.mask for item in batch_items]).to(accelerator.device, dtype=torch.float32)
                        
                        with torch.no_grad():
                            original_latents = vae.encode(original_pixel_values).latent_dist.sample()
                        original_latents = original_latents * vae.config.scaling_factor
                        
                        noise = torch.randn_like(original_latents)
                        bsz = original_latents.shape[0]
                        timesteps = torch.randint(
                            0,
                            scheduler.config.num_train_timesteps,
                            (bsz,),
                            device=original_latents.device,
                        ).long()
                        noisy_latents = scheduler.add_noise(original_latents, noise, timesteps)
                        
                        # 텍스트 프롬프트
                        prompts = self.model_config["prompts"] # 임시 프롬프트
                        
                        with torch.no_grad(): # Text encoder는 학습하지 않음
                            if text_encoder_2:
                                tokenized_prompt1 = tokenizer(
                                    prompts, padding="max_length", max_length=tokenizer.model_max_length,
                                    truncation=True, return_tensors="pt", add_special_tokens=True
                                ).input_ids.to(accelerator.device)
                                tokenized_prompt2 = tokenizer_2(
                                    prompts, padding="max_length", max_length=tokenizer_2.model_max_length,
                                    truncation=True, return_tensors="pt", add_special_tokens=True
                                ).input_ids.to(accelerator.device)
                                
                                prompt_embeds1_out = text_encoder(tokenized_prompt1, output_hidden_states=True)
                                prompt_embeds2_out = text_encoder_2(tokenized_prompt2, output_hidden_states=True)
                                
                                prompt_embeds = torch.cat(
                                    [prompt_embeds1_out.hidden_states[-2], prompt_embeds2_out.hidden_states[-2]], dim=-1
                                )
                                pooled_prompt_embeds = prompt_embeds2_out.pooler_output # 또는 prompt_embeds2_out[0]
                                
                                # SDXL add_time_ids 준비 (원본 이미지 크기 기반)
                                orig_H, orig_W = original_pixel_values.shape[2], original_pixel_values.shape[3]
                                add_time_ids = self.pipe._get_add_time_ids(
                                    (orig_H, orig_W), (0,0), (orig_H, orig_W), dtype=prompt_embeds.dtype, text_embeds_shape=None
                                ).to(accelerator.device)
                                add_time_ids = add_time_ids.repeat(bsz, 1)
                                
                                
                                encoder_hidden_states = prompt_embeds
                                added_cond_kwargs = {"text_embeds": pooled_prompt_embeds, "time_ids": add_time_ids}
                            else: # 일반 Stable Diffusion
                                tokenized_prompt = tokenizer(
                                    prompts, padding="max_length", max_length=tokenizer.model_max_length,
                                    truncation=True, return_tensors="pt"
                                ).input_ids.to(accelerator.device)
                                encoder_hidden_states = text_encoder(tokenized_prompt)[0]
                                added_cond_kwargs = None
                                
                        # --- 3. 모델 예측 (UNet이 노이즈를 예측하도록) ---
                        model_pred_noise = unet(
                            noisy_latents,
                            timesteps,
                            encoder_hidden_states=encoder_hidden_states,
                            added_conditional_kwargs=added_cond_kwargs if text_encoder_2 else None # SDXL은 added_conditional_kwargs
                        ).sample
                        
                        # --- 4. 손실 계산 (마스크된 영역에 집중) ---
                        # 마스크(masks)는 픽셀 공간 기준이므로, latent 공간 크기에 맞게 리사이징
                        # 마스크가 (B, 1, H, W)이고, 1이 복원 영역
                        resized_masks_for_latents = F.interpolate(masks, size=original_latents.shape[2:], mode='nearest')
                        
                        # 4.1 MSE 손실: 모델이 예측한 노이즈 vs 실제 노이즈 (마스크된 영역에서만)
                        mse_loss = F.mse_loss(
                            model_pred_noise * resized_masks_for_latents, # 마스크된 영역의 예측 노이즈
                            noise * resized_masks_for_latents,           # 마스크된 영역의 실제 노이즈
                            reduction="sum"
                        ) / resized_masks_for_latents.sum().clamp(min=1e-6) # 0으로 나누는 것 방지
                        
                        # 4.2 SSIM 손실: 복원된 이미지 vs 원본 이미지 (마스크된 영역에서만)
                        # 예측된 노이즈로부터 원본 latent 복원 시도 (스케줄러의 alpha, beta 값 필요)
                        # scheduler.step()은 추론용. 학습 시에는 직접 계산하거나, DDPM 논문의 단순화된 L_simple 사용.
                        # 여기서는 간단히 model_pred_noise가 실제 노이즈와 얼마나 다른지를 MSE로 측정했으므로,
                        # SSIM은 복원된 이미지 품질을 보기 위해 사용.
                        
                        pred_original_latents = scheduler.step(model_pred_noise, timesteps, noisy_latents).pred_original_sample
                        
                        with torch.no_grad(): # VAE 디코딩은 학습에 영향 X
                            pred_pixel_values = vae.decode(pred_original_latents / vae.config.scaling_factor).sample
                        # target_pixel_values는 original_pixel_values (이미 [-1,1] 범위)
                        
                        # SSIM 계산 (마스크된 영역의 픽셀 값에 대해)
                        # pred_pixel_values와 original_pixel_values는 이미 [-1,1] 범위
                        # ssim_loss_fn은 내부적으로 [0,1]로 변환해서 계산
                        # 마스크를 픽셀 공간 이미지에 적용 (곱하기)
                        ssim_loss = self.ssim_loss(
                            pred_pixel_values * masks, # 마스크된 영역의 예측 픽셀
                            original_pixel_values * masks  # 마스크된 영역의 원본 픽셀
                        )
                        
                        # 최종 손실
                        lambda_ssim = self.model_config["lambda_ssim"]
                        loss = mse_loss + lambda_ssim * ssim_loss
                        
                        # --- 5. 역전파 및 업데이트 ---
                        accelerator.backward(loss)
                        if accelerator.sync_gradients:
                            accelerator.clip_grad_norm_(unet.parameters(), self.model_config.get("max_grad_norm", 1.0))
                        optimizer.step()
                        optimizer.zero_grad()
                        
                        global_step += 1
                        
                        # --- 6. 에포크 종료 후 로깅 ---
                        if accelerator.is_main_process:
                            avg_loss = loss.item() / num_batches
                            avg_mse_loss = mse_loss.item() / num_batches
                            avg_ssim_loss = ssim_loss.item() / num_batches
                            
                            tqdm.write(
                                f"Epoch {epoch+1} Summary: Avg Loss: {avg_loss:.4f}, "
                                f"Avg MSE Loss: {avg_mse_loss:.4f}, "
                                f"Avg SSIM Loss: {avg_ssim_loss:.4f}"
                            )
                            
                    # --- 학습 완료 후 LoRA 가중치 저장 (메인 프로세스에서만) ---
                    if accelerator.is_main_process:
                        unwrapped_unet = accelerator.unwrap_model(unet)
                        unwrapped_unet.save_pretrained(self.model_config["lora_path"])
                        accelerator.print(f"Model3 LoRA weights saved to {self.model_config['lora_path']}")
                            
                            
                            
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                                
                                
                            
                        
                        
                        
            
    
    def inference(self, texted_images_for_model3: list[TextedImage]):
        self.pipe.load_lora_weights(self.model_config["lora_path"], weight_name=self.model_config["lora_weight_name"])
        
        return
            
            
            
            
