import torch
import os
import numpy as np
import torch.nn.functional as F
from ..datas.TextedImage import TextedImage
from torch import nn
from diffusers import SD3Transformer2DModel, StableDiffusion3Pipeline
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from accelerate import Accelerator
from tqdm import tqdm
from pytorch_msssim import ms_ssim
from peft import get_peft_model, LoraConfig, PeftModel
import gc
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from torchvision import transforms # 이미지 전처리를 위해 추가 임포트
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
        

    def inference(self, texted_images_for_model3: list[TextedImage]):
            if not self.accelerator.is_main_process: return
            print("Loading SD3 pipeline for inference...")

            weight_dtype = torch.float16
            model_id = self.model_config["model_id"]
            lora_weights_path = self.model_config["lora_path"]

            try:
                base_transformer = SD3Transformer2DModel.from_pretrained(
                    model_id, 
                    subfolder="transformer", 
                    torch_dtype=weight_dtype
                )

                if not os.path.isdir(lora_weights_path):
                    print(f"Error: LoRA weights path '{lora_weights_path}' is not a valid directory. Ensure it points to the directory where save_pretrained() saved the LoRA weights.")
                    raise FileNotFoundError(f"LoRA directory not found: {lora_weights_path}")

                lora_transformer = PeftModel.from_pretrained(base_transformer, lora_weights_path)
                
                if self.model_config.get("fuse_lora_on_inference", True):
                    print("Fusing LoRA weights into the base transformer model for inference...")
                    lora_transformer = lora_transformer.merge_and_unload()
                
                print(f"PEFT LoRA weights from {lora_weights_path} loaded and applied to transformer.")

                pipe = StableDiffusion3Pipeline.from_pretrained(
                    model_id,
                    transformer=lora_transformer,
                    torch_dtype=weight_dtype,
                )
                
                current_device = self.accelerator.device
                pipe.to(current_device)

                print(f"SD3 pipeline with custom LoRA transformer loaded successfully to {current_device}.")

            except Exception as e:
                print(f"Error loading SD3 pipeline with LoRA for inference: {e}")
                return
            
            output_dir = self.model_config.get("output_dir", "datas/images/output/model3_sd3_inference")
            os.makedirs(output_dir, exist_ok=True)
            
            prompt = self.model_config["prompts"]
            negative_prompt = self.model_config["negative_prompt"] # <--- 여기에 정의

            print("Starting SD3 LoRA Inference (Latent Inpainting Mode)...")
            for i, original_texted_image in enumerate(tqdm(texted_images_for_model3, desc="SD3 LoRA Inference")):
                with torch.no_grad():
                    try:
                        # 입력 텐서 정규화 및 장치 이동
                        # orig_tensor는 (C, H, W) 형태이므로, VAE 입력 (B, C, H, W)를 위해 unsqueeze(0) 필요
                        orig_tensor_norm = (original_texted_image.orig.to(current_device, dtype=weight_dtype) * 2.0 - 1.0).unsqueeze(0)
                        # mask_tensor는 (1, H, W) 형태라고 했으므로, 이미 배치 차원 1을 가짐. 추가 unsqueeze(0) 불필요.
                        mask_tensor = original_texted_image.mask.to(current_device, dtype=torch.float32) 
                        
                        latent_space_resolution = 512 # 또는 1024
                        orig_tensor_norm = F.interpolate(orig_tensor_norm, size=(latent_space_resolution, latent_space_resolution), mode="bicubic", align_corners=True)
                        mask_tensor_resized_for_vae_input = F.interpolate(mask_tensor.unsqueeze(0), size=(latent_space_resolution, latent_space_resolution), mode="nearest").squeeze(0)
                        mask_tensor = mask_tensor_resized_for_vae_input # 이후 Latent Masking에 사용


                        pipe.vae.to(current_device)
                        # Latent Space 인코딩
                        latent_image = pipe.vae.encode(orig_tensor_norm).latent_dist.sample() * pipe.vae.config.scaling_factor
                        latent_image = latent_image.to(weight_dtype) # Latent 이미지 dtype 통일
                        
                        # 마스크도 Latent Space 크기에 맞춰 다운샘플링 및 dtype 통일
                        # VAE의 다운샘플링 팩터는 보통 8
                        latent_mask_size = (latent_image.shape[2], latent_image.shape[3]) 
                        latent_mask = F.interpolate(mask_tensor.unsqueeze(0), size=latent_mask_size, mode="nearest").squeeze(0) # mask_tensor는 (1,H,W), interpolate는 (B,C,H,W) 기대
                        latent_mask = (latent_mask > 0.5).float().to(weight_dtype) # 0 또는 1 (float), weight_dtype으로 캐스팅

                        pipe.vae.to("cpu")
                        
                        # Inpainting을 위한 초기 Latent 생성
                        noise = torch.randn_like(latent_image, device=current_device, dtype=weight_dtype)
                        initial_latent = latent_image * (1 - latent_mask) + noise * latent_mask 
                        
                        pipe.text_encoder.to(current_device)
                        if pipe.text_encoder_2: pipe.text_encoder_2.to(current_device)
                        if pipe.text_encoder_3: pipe.text_encoder_3.to(current_device)
                        
                        # 텍스트 인코딩
                        prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = pipe.encode_prompt(
                            prompt=prompt,
                            prompt_2=prompt if pipe.text_encoder_2 else None,
                            prompt_3=prompt if pipe.text_encoder_3 else None,
                            device=current_device,
                            num_images_per_prompt=1, # 배치 크기 1
                            do_classifier_free_guidance=True,
                            negative_prompt=negative_prompt,
                            negative_prompt_2=negative_prompt if pipe.text_encoder_2 else None,
                            negative_prompt_3=negative_prompt if pipe.text_encoder_3 else None,
                        )

                        # 모든 임베딩 텐서를 weight_dtype으로 명시적 캐스팅 후 concat
                        prompt_embeds_full = torch.cat([negative_prompt_embeds.to(weight_dtype), prompt_embeds.to(weight_dtype)], dim=0)
                        pooled_embeddings_full = torch.cat([negative_pooled_prompt_embeds.to(weight_dtype), pooled_prompt_embeds.to(weight_dtype)], dim=0) 
                        
                        pipe.text_encoder.to("cpu")
                        if pipe.text_encoder_2: pipe.text_encoder_2.to("cpu")
                        if pipe.text_encoder_3: pipe.text_encoder_3.to("cpu")
                        
                        # 디퓨전 루프
                        num_inference_steps = self.model_config.get("inference_steps", 28)
                        guidance_scale = self.model_config.get("guidance_scale", 7.0)
                        pipe.scheduler.set_timesteps(num_inference_steps, device=current_device)
                        timesteps = pipe.scheduler.timesteps
                        
                        latents = initial_latent.clone() 
                        
                        for t in timesteps:
                            # latents는 이미 weight_dtype일 가능성이 높지만, 명시적으로 to(weight_dtype)를 적용.
                            # CFG를 위한 latents 확장 시에도 dtype 유지
                            latent_model_input = latents.repeat(2, 1, 1, 1).to(weight_dtype) if guidance_scale > 1 else latents.to(weight_dtype)
                            
                            # t_expanded는 torch.long 타입이어야 하므로 dtype 변경은 하지 않고 device만 맞춤.
                            # 다만, 모델이 float 텐서를 기대할 수 있으므로, .float()로 변환 후 weight_dtype으로 캐스팅.
                            t_expanded = t.expand(latent_model_input.shape[0]).to(current_device).to(weight_dtype) # <--- 여기에 .to(weight_dtype) 추가

                            with self.accelerator.autocast():
                                noise_pred = pipe.transformer(
                                    hidden_states = latent_model_input, # 이미 weight_dtype
                                    timestep = t_expanded, # 이제 weight_dtype
                                    encoder_hidden_states = prompt_embeds_full, # 이미 weight_dtype
                                    pooled_projections = pooled_embeddings_full, # 이미 weight_dtype
                                    return_dict=False
                                )[0]
                                
                            if guidance_scale > 1.0:
                                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                                
                            # scheduler.step의 출력도 weight_dtype으로 유지
                            latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0].to(weight_dtype) # <--- 여기에 .to(weight_dtype) 추가
                            
                            # Inpainting 핵심: 마스크되지 않은 영역을 원본 이미지의 latent로 강제 유지
                            # latent_image, latent_mask는 이미 weight_dtype임.
                            latents = latent_image * (1 - latent_mask) + latents * latent_mask 
                            # --- 여기까지 수정 ---
                            
                        # 최종 잠재 공간을 이미지로 디코딩
                        pipe.vae.to(current_device)
                        # VAE 디코더는 float32를 선호할 수 있지만, autocast가 처리.
                        # 출력 샘플은 weight_dtype일 것임.
                        final_image_tensor_norm = pipe.vae.decode(latents / pipe.vae.config.scaling_factor).sample
                        pipe.vae.to("cpu") 
                        
                        # 이미지 후처리: -1~1 범위 텐서를 0~1 범위로 변환, PIL Image로 변환
                        final_image_tensor_0_1 = (final_image_tensor_norm / 2 + 0.5).clamp(0, 1)
                        
                        # 배치 차원 제거
                        final_image_tensor_0_1 = final_image_tensor_0_1.squeeze(0) 
                        
                        # 필요한 경우 원본 이미지의 최종 크기(256x256)로 다시 다운샘플링 (VAE 입력 크기가 컸다면)
                        if final_image_tensor_0_1.shape[-2:] != original_texted_image.orig.shape[-2:]:
                            final_image_tensor_0_1 = F.interpolate(final_image_tensor_0_1.unsqueeze(0), size=original_texted_image.orig.shape[-2:], mode="bicubic", align_corners=True).squeeze(0)
                        
                        # PIL 이미지로 변환
                        final_inpainted_pil = Image.fromarray((final_image_tensor_0_1.cpu().permute(1,2,0).numpy() * 255).astype(np.uint8), "RGB") # np.uint8 사용
                        
                        # 시각화를 위한 원본 PIL 이미지들 준비
                        original_pil, text_pil, mask_pil = original_texted_image._to_pil()
                        
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

                    except Exception as e:
                        print(f"Error during SD3 inference for item {i}: {e}")
                        # 오류 발생 시 원본 이미지의 마스크 영역에 노이즈를 채워 시각적으로 표시
                        original_pil, text_pil, mask_pil = original_texted_image._to_pil()
                        
                        # 노이즈 이미지를 원본 이미지 크기와 동일하게 생성 (PIL RGB)
                        error_noise_pil = Image.fromarray(
                            (torch.rand(3, original_pil.size[1], original_pil.size[0]) * 255).byte().permute(1,2,0).cpu().numpy(), "RGB"
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
                
            print(f"All SD3 LoRA inference results saved to {output_dir}")
            del pipe
            gc.collect(); torch.cuda.empty_cache()
            return