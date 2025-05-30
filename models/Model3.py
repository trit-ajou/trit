from torch import nn
import torch
from diffusers import AutoPipelineForInpainting, StableDiffusionXLPipeline
from peft import LoraConfig, get_peft_model
import torch.nn.functional as F
from accelerate import Accelerator
from tqdm import tqdm
from ..datas.TextedImage import TextedImage
from pytorch_msssim import ms_ssim
from PIL import Image
import torchvision.transforms as T
import os
import matplotlib.pyplot as plt
import inspect

class Model3(nn.Module):
    """Masked Inpainting Model
    config: 모델 설정 정보 dictionary
    
    """

    def __init__(self, model_config: dict):
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
    
    def _visualize_results(self, texted_image: TextedImage, output_tensor, index: int):
        # 원본, 텍스트, 마스크, 결과 이미지를 함께 표시
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        # 텐서를 PIL 이미지로 변환
        orig_img, text_img, mask_img = texted_image._to_pil()
        
        # PIL 이미지인 경우 처리
        if isinstance(output_tensor, torch.Tensor):
            output_img = Image.fromarray(
                (output_tensor.detach().cpu().permute(1, 2, 0) * 255).numpy().astype('uint8')
            )
        else:
            output_img = output_tensor  # 이미 PIL 이미지인 경우
        
        # 이미지 표시
        axes[0].imshow(orig_img)
        axes[0].set_title("Original")
        axes[0].axis("off")
        
        axes[1].imshow(text_img)
        axes[1].set_title("Text Image")
        axes[1].axis("off")
        
        axes[2].imshow(mask_img, cmap="gray")
        axes[2].set_title("Mask")
        axes[2].axis("off")
        
        axes[3].imshow(output_img)
        axes[3].set_title("Inpainted Result")
        axes[3].axis("off")
        
        # 저장 경로
        save_dir = self.model_config["output_dir"]
        os.makedirs(save_dir, exist_ok=True)
        
        # 결과 저장
        plt.tight_layout()
        plt.savefig(f"{save_dir}/inpainted_result_{index}.png")
        plt.close(fig)
        
        print(f"Visualization saved to {save_dir}/inpainted_result_{index}.png")
    
    def lora_train(self, texted_images_for_model3: list[TextedImage], accelerator: Accelerator):
        # 1. 파이프라인 로드 (CPU에 먼저 로드하여 메모리 부담 분산 시도)
        print("Loading pipeline components to CPU first...")
        pipe = StableDiffusionXLPipeline.from_pretrained(
            self.model_config["model_id"],
            torch_dtype=torch.float16, # dtype은 유지하되, device는 지정 안함
            variant="fp16"
        )
        
        # 2. 필요한 구성 요소 추출
        print("Extracting pipeline components...")
        vae = pipe.vae
        unet = pipe.unet
        text_encoder = pipe.text_encoder
        text_encoder_2 = getattr(pipe, 'text_encoder_2', None)
        tokenizer = pipe.tokenizer
        tokenizer_2 = getattr(pipe, 'tokenizer_2', None)
        scheduler = pipe.scheduler
        
        # 3. 원본 파이프라인 객체 삭제 (메모리 절약 시도)
        print("Deleting original pipeline object...")
        del pipe 
        accelerator.wait_for_everyone() # 모든 프로세스에서 pipe 객체 삭제 동기화
        torch.cuda.empty_cache() # 캐시 비우기
        print("CUDA cache cleared after deleting pipeline object.")

        # 4. 각 구성요소를 accelerator.device로 이동
        print("Moving VAE to device...")
        vae = vae.to(accelerator.device)
        print("Moving Text Encoder 1 to device...")
        text_encoder = text_encoder.to(accelerator.device)
        if text_encoder_2:
            print("Moving Text Encoder 2 to device...")
            text_encoder_2 = text_encoder_2.to(accelerator.device)
        # unet은 accelerator.prepare에서 처리됨
        # scheduler는 일반적으로 device에 직접 올릴 필요는 없으나, 상태를 가진다면 고려
        print("Components moved to device (UNet will be moved by accelerator.prepare).")
        
        vae.eval()
        text_encoder.eval()
        if text_encoder_2:
            text_encoder_2.eval()
            # text_encoder_2의 projection_dim 확인 및 기본값 설정 (임시 방편)
            if getattr(text_encoder_2.config, 'projection_dim', None) is None:
                print("Warning: text_encoder_2.config.projection_dim is None. Setting to 1280 as a fallback.")
                text_encoder_2.config.projection_dim = 1280 # SDXL 기본값
        
        # unet.requires_grad_(False) # PEFT 사용 시 이 부분은 get_peft_model이 처리

        # LoRA Config 생성
        # target_modules는 SDXL UNet에 맞게 정확히 지정해야 합니다.
        # 예시: ["to_q", "to_k", "to_v", "to_out.0"] 또는 Attention 프로세서 내부의 Linear 레이어들
        # 정확한 모듈 이름은 unet.named_modules()를 통해 확인하거나,
        # 좀 더 안전한 방법은 모든 Linear 레이어를 타겟팅 하거나, 특정 패턴으로 찾는 것입니다.
        # target_modules_example = [
        #     "attn1.to_q", "attn1.to_k", "attn1.to_v", "attn1.to_out.0",  # Self-attention in BasicTransformerBlock
        #     "attn2.to_q", "attn2.to_k", "attn2.to_v", "attn2.to_out.0",  # Cross-attention in BasicTransformerBlock
        # ]

        config_target_modules = ["to_q", "to_k", "to_v", "to_out.0"]
        # 만약 위 모듈이 없다면, "to_q", "to_k", "to_v", "to_out.0" (또는 to_out[0]) 등을 시도해야 합니다.
        # 또는, unet.named_modules()를 통해 전체 모듈 이름을 보고, 해당 경로의 마지막 부분만 사용합니다.
        # (예: 'down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q' -> 'to_q')

        lora_config = LoraConfig(
            r=self.model_config["lora_rank"],
            lora_alpha=self.model_config["lora_alpha"],
            target_modules=config_target_modules, # ["q_proj", "k_proj", "v_proj", "out_proj"], # **매우 중요, 모델에 맞게 수정 필요**
            lora_dropout=0.05,  # 예시 값
            bias="none",        # "none", "all", or "lora_only"
            # task_type="CAUSAL_LM" # UNet에는 해당되지 않을 수 있음, 필요시 제거
        )

        unet = get_peft_model(unet, lora_config)
        unet.print_trainable_parameters() # 학습 가능한 파라미터 수 확인

        # 옵티마이저 설정 (PEFT가 적용된 unet의 파라미터를 전달)
        optimizer = torch.optim.AdamW(
            unet.parameters(), # PEFT가 학습 대상 파라미터만 requires_grad=True로 설정함
            lr=self.model_config["lr"]
        )

        unet, optimizer = accelerator.prepare(unet, optimizer)
        
        # 학습 루프
        global_step = 0
        best_loss = float('inf')
        
        # 학습 루프 시작 전 캐시 비우기
        accelerator.wait_for_everyone()
        torch.cuda.empty_cache()
        print("CUDA cache cleared before starting training loop.")
        
        for epoch in range(self.model_config["epochs"]):
            # 매 에포크 시작 시 캐시 비우기 (선택적, 하지만 OOM 발생 시 유용)
            accelerator.wait_for_everyone()
            torch.cuda.empty_cache()
            print(f"CUDA cache cleared at the beginning of epoch {epoch+1}.")
            
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
                    masks_for_pixel_space = torch.stack([item.mask for item in batch_items]).to(accelerator.device, dtype=torch.float32)
                    
                    masked_pixel_values = original_pixel_values * (1 - masks_for_pixel_space) 
                    
                    with torch.no_grad():
                        gt_latents = vae.encode(original_pixel_values.to(vae.dtype)).latent_dist.sample()
                        gt_latents = gt_latents * vae.config.scaling_factor

                        masked_image_latents = vae.encode(masked_pixel_values.to(vae.dtype)).latent_dist.sample()
                        masked_image_latents = masked_image_latents * vae.config.scaling_factor

                    latent_H, latent_W = gt_latents.shape[2], gt_latents.shape[3]
                    resized_masks_for_latents = F.interpolate(masks_for_pixel_space, size=(latent_H, latent_W), mode='nearest')

                    noise = torch.randn_like(gt_latents)
                    bsz = gt_latents.shape[0]
                    timesteps = torch.randint(
                        0,
                        scheduler.config.num_train_timesteps,
                        (bsz,),
                        device=gt_latents.device,
                    ).long()
                    noisy_gt_latents = scheduler.add_noise(gt_latents, noise, timesteps) 

                    unet_input_latents = torch.cat([noisy_gt_latents, resized_masks_for_latents, masked_image_latents], dim=1)

                    # --- 텍스트 임베딩 및 추가 조건 준비 --- 
                    with torch.no_grad(): 
                        prompts = self.model_config["prompts"]
                        added_cond_kwargs = {} 
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
                            if hasattr(prompt_embeds2_out, 'text_embeds'):
                                pooled_prompt_embeds = prompt_embeds2_out.text_embeds
                            elif hasattr(prompt_embeds2_out, 'last_hidden_state'):
                                pooled_prompt_embeds = prompt_embeds2_out.last_hidden_state[:, 0] 
                            else:
                                raise AttributeError("CLIPTextModelOutput does not have 'text_embeds' or 'last_hidden_state' for pooling.")
                            orig_H, orig_W = original_pixel_values.shape[2], original_pixel_values.shape[3]
                            text_proj_dim = None
                            if hasattr(text_encoder_2.config, 'projection_dim'):
                                text_proj_dim = text_encoder_2.config.projection_dim
                            add_time_ids = self.pipe._get_add_time_ids(
                                (orig_H, orig_W), (0,0), (orig_H, orig_W), 
                                dtype=prompt_embeds.dtype,
                                text_encoder_projection_dim=text_proj_dim 
                            ).to(accelerator.device)
                            add_time_ids = add_time_ids.repeat(bsz, 1)
                            encoder_hidden_states = prompt_embeds
                            added_cond_kwargs["text_embeds"] = pooled_prompt_embeds
                            added_cond_kwargs["time_ids"] = add_time_ids
                        else: 
                            tokenized_prompt = tokenizer(
                                prompts, padding="max_length", max_length=tokenizer.model_max_length,
                                truncation=True, return_tensors="pt"
                            ).input_ids.to(accelerator.device)
                            encoder_hidden_states = text_encoder(tokenized_prompt)[0]
                          
                    # --- 3. 모델 예측 (UNet이 노이즈를 예측하도록) ---
                    model_pred_noise = unet(
                        unet_input_latents,
                        timesteps,
                        encoder_hidden_states=encoder_hidden_states,
                        added_cond_kwargs=added_cond_kwargs
                    ).sample
                    
                    # --- 4. 손실 계산 (마스크된 영역에 집중) ---
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
                    
                    # scheduler.step 전에 model_pred_noise 스케일링
                    scaled_model_pred_noise = scheduler.scale_model_input(model_pred_noise, timesteps)
                    pred_original_latents = scheduler.step(scaled_model_pred_noise, timesteps, noisy_gt_latents).pred_original_sample
                    
                    with torch.no_grad(): # VAE 디코딩은 학습에 영향 X
                        # vae.decode에 전달하기 전에 dtype 맞춤
                        latents_for_decode = pred_original_latents / vae.config.scaling_factor
                        pred_pixel_values = vae.decode(latents_for_decode.to(vae.dtype)).sample
                    # target_pixel_values는 original_pixel_values (이미 [-1,1] 범위)
                    
                    # SSIM 계산 (마스크된 영역의 픽셀 값에 대해)
                    # pred_pixel_values와 original_pixel_values는 이미 [-1,1] 범위
                    # ssim_loss_fn은 내부적으로 [0,1]로 변환해서 계산
                    # 마스크를 픽셀 공간 이미지에 적용 (곱하기)
                    ssim_loss = self.ssim_loss(
                        pred_pixel_values * masks_for_pixel_space, # 마스크된 영역의 예측 픽셀
                        original_pixel_values * masks_for_pixel_space  # 마스크된 영역의 원본 픽셀
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
                        
                    # --- 학습 후 LoRA 가중치 저장 ---
                    if accelerator.is_main_process and loss < best_loss:
                        best_loss = loss
                        save_path = os.path.join(self.model_config["lora_path"], f"checkpoint-{global_step}")
                        # accelerator.unwrap_model(unet).save_pretrained(save_path) # unwrap 필요할 수 있음
                        unet.save_pretrained(save_path) # PEFT 모델의 저장 메서드 사용
                        print(f"LoRA weights saved to {save_path}")

    def inference(self, texted_images_for_model3: list[TextedImage]):
        self.pipe = AutoPipelineForInpainting.from_pretrained(
            self.model_config["model_id"],
            torch_dtype=torch.float16,
            variant="fp16"
        ).to("cuda")

            
            
            
