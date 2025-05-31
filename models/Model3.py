import torch
import os
import torch.nn.functional as F
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

    def lora_train(self, texted_images_for_model3: list["TextedImage"]):
        if self.accelerator.is_main_process:
            print("\n[Model3.lora_train] Entered lora_train method.")
            print(f"[Model3.lora_train] Using accelerator for training setup, including mixed precision and device management.")

        model_id = self.model_config["model_id"]
        lr = self.model_config["lr"]
        weight_decay = self.model_config.get("weight_decay", 3e-4)
        lora_rank = self.model_config["lora_rank"]
        lora_alpha = self.model_config["lora_alpha"]
        lambda_ssim = self.model_config.get("lambda_ssim", 0.8)
        lora_path = self.model_config["lora_path"]
        num_epochs = self.model_config["epochs"]
        prompts_config_str = self.model_config["prompts"]
        negative_prompts_config_str = self.model_config["negative_prompt"]
        
        disable_text_encoder_2 = self.model_config.get("disable_text_encoder_2", False)
        disable_text_encoder_3 = self.model_config.get("disable_text_encoder_3", False)
        enable_gradient_checkpointing = self.model_config.get("gradient_checkpointing", True)

        if self.accelerator.is_main_process:
            print(f"[Model3.lora_train] Using model_id: {model_id}")
            print(f"[Model3.lora_train] Text Encoder 2 disabling: {disable_text_encoder_2}")
            print(f"[Model3.lora_train] Text Encoder 3 disabling: {disable_text_encoder_3}")
            print(f"[Model3.lora_train] Gradient Checkpointing config: {enable_gradient_checkpointing}")

        base_model_nf4 = SD3Transformer2DModel.from_pretrained(
            model_id, subfolder="transformer", torch_dtype=torch.float16
        )
        if enable_gradient_checkpointing:
            try:
                if hasattr(base_model_nf4, 'enable_gradient_checkpointing'):
                    base_model_nf4.enable_gradient_checkpointing()
                    if self.accelerator.is_main_process: print("[Model3.lora_train] Gradient checkpointing enabled for base UNet (before PEFT) via enable_gradient_checkpointing().")
                elif hasattr(base_model_nf4, 'gradient_checkpointing_enable'): 
                    base_model_nf4.gradient_checkpointing_enable()
                    if self.accelerator.is_main_process: print("[Model3.lora_train] Gradient checkpointing enabled for base UNet (before PEFT) via gradient_checkpointing_enable().")
                else:
                    if self.accelerator.is_main_process: print("[Model3.lora_train] Base UNet does not have a direct gradient_checkpointing_enable method. Checkpointing may not be applied directly to base model.")
            except Exception as e:
                if self.accelerator.is_main_process: print(f"[Model3.lora_train] Error enabling gradient checkpointing on base UNet: {e}")

        lora_config = LoraConfig(
            r=lora_rank, lora_alpha=lora_alpha, target_modules=["to_k", "to_q", "to_v", "to_out.0"],
            lora_dropout=0.1, bias="none", init_lora_weights="gaussian"
        )
        model_nf4_lora = get_peft_model(base_model_nf4, lora_config) 
        del base_model_nf4; gc.collect(); torch.cuda.empty_cache()
        
        if self.accelerator.is_main_process: print("[Model3.lora_train] LoRA UNet model prepared for accelerator.prepare.")


        if self.accelerator.is_main_process: print(f"[Model3.lora_train] Loading full StableDiffusion3Pipeline for utility components from '{model_id}'...")
        
        try:
            utility_pipeline = StableDiffusion3Pipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )
            utility_pipeline.vae.to("cpu").eval()
            utility_pipeline.text_encoder.to("cpu").eval()
            if utility_pipeline.text_encoder_2: utility_pipeline.text_encoder_2.to("cpu").eval()
            if utility_pipeline.text_encoder_3: utility_pipeline.text_encoder_3.to("cpu").eval()
            utility_pipeline.transformer.to("cpu").eval() 
            
            if self.accelerator.is_main_process: print("[Model3.lora_train] Full StableDiffusion3Pipeline loaded and utility components moved to CPU.")
        except Exception as e:
            if self.accelerator.is_main_process: print(f"[Model3.lora_train] Error loading full StableDiffusion3Pipeline: {e}")
            raise 

        if self.accelerator.is_main_process: print(f"[Model3.lora_train] Scheduler loaded: {type(utility_pipeline.scheduler).__name__}")
        
        gc.collect(); torch.cuda.empty_cache()


        if self.accelerator.is_main_process: model_nf4_lora.print_trainable_parameters()
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model_nf4_lora.parameters()),
            lr=lr, weight_decay=weight_decay
        )
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.95)
        
        train_loader = torch.utils.data.DataLoader(
            texted_images_for_model3, batch_size=self.model_config["batch_size"],
            shuffle=True, num_workers=0, collate_fn=identity_collate
        )
        
        model_nf4_lora, optimizer, train_loader, lr_scheduler = self.accelerator.prepare(
            model_nf4_lora, optimizer, train_loader, lr_scheduler
        )
        if self.accelerator.is_main_process: print("[Model3.lora_train] Model, optimizer, dataloader, lr_scheduler prepared by accelerator.")

        best_epoch_loss = float("inf")
        model_nf4_lora.train() 

        for epoch in range(num_epochs):
            epoch_loss_sum = 0.0
            if self.accelerator.is_main_process: print(f"\n[Model3.lora_train] Epoch {epoch+1}/{num_epochs} 시작")
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", disable=not self.accelerator.is_main_process)

            for step, batch in enumerate(progress_bar):
                current_batch_size = len(batch)
                
                prompts_list_str = [prompts_config_str] * current_batch_size
                negative_prompts_list_str = [negative_prompts_config_str] * current_batch_size

                with torch.no_grad():
                    inputs = torch.stack([x.timg for x in batch]).to(self.accelerator.device, dtype=torch.float16, non_blocking=True)
                    targets = torch.stack([x.orig for x in batch]).to(self.accelerator.device, dtype=torch.float16, non_blocking=True)
                    masks = torch.stack([x.mask for x in batch]).to(self.accelerator.device, dtype=torch.float32, non_blocking=True)
                    inputs = (inputs * 2.0) - 1.0; targets = (targets * 2.0) - 1.0
                    orig_size = inputs.shape[-2:]
                    vae_input_images = F.interpolate(inputs, size=(self.model_config["input_size"]), mode="bicubic", align_corners=True)

                    utility_pipeline.text_encoder.to(self.accelerator.device)
                    if utility_pipeline.text_encoder_2: utility_pipeline.text_encoder_2.to(self.accelerator.device)
                    if utility_pipeline.text_encoder_3: utility_pipeline.text_encoder_3.to(self.accelerator.device)

                    actual_prompt_embeds, actual_negative_prompt_embeds, \
                    actual_pooled_prompt_embeds, actual_negative_pooled_prompt_embeds = utility_pipeline.encode_prompt(
                        prompt=prompts_list_str, 
                        prompt_2=prompts_list_str if utility_pipeline.text_encoder_2 else None, 
                        prompt_3=prompts_list_str if utility_pipeline.text_encoder_3 else None,
                        device=self.accelerator.device, num_images_per_prompt=1, do_classifier_free_guidance=True,
                        negative_prompt=negative_prompts_list_str, 
                        negative_prompt_2=negative_prompts_list_str if utility_pipeline.text_encoder_2 else None, 
                        negative_prompt_3=negative_prompts_list_str if utility_pipeline.text_encoder_3 else None
                    )
                    utility_pipeline.text_encoder.to("cpu")
                    if utility_pipeline.text_encoder_2: utility_pipeline.text_encoder_2.to("cpu")
                    if utility_pipeline.text_encoder_3: utility_pipeline.text_encoder_3.to("cpu")
                
                prompt_embeds_full = torch.cat([actual_negative_prompt_embeds, actual_prompt_embeds], dim=0)
                pooled_embeddings_full = torch.cat([actual_negative_pooled_prompt_embeds, actual_pooled_prompt_embeds], dim=0)
                del actual_prompt_embeds, actual_negative_prompt_embeds, actual_pooled_prompt_embeds, actual_negative_pooled_prompt_embeds
                gc.collect(); torch.cuda.empty_cache()

                with torch.no_grad(): 
                    utility_pipeline.vae.to(self.accelerator.device) 
                    vae_out = utility_pipeline.vae.encode(vae_input_images)
                    latent_single = (vae_out.latent_dist.sample() * utility_pipeline.vae.config.scaling_factor).to(torch.float16)
                    del vae_out, vae_input_images 
                    gc.collect(); torch.cuda.empty_cache()
                
                max_train_timesteps = self.model_config["max_train_timesteps"]
                t_single = torch.randint(0, max_train_timesteps, (current_batch_size,), device=self.accelerator.device, dtype=torch.long)
                repeat_factor = pooled_embeddings_full.shape[0] // current_batch_size if current_batch_size > 0 else 1
                
                latent_for_unet = latent_single.repeat(repeat_factor, 1, 1, 1)
                t_for_unet = t_single.repeat(repeat_factor)
                del latent_single, t_single
                gc.collect(); torch.cuda.empty_cache()
                
                with self.accelerator.autocast(): 
                    outputs = model_nf4_lora(
                        hidden_states=latent_for_unet, timestep=t_for_unet,         
                        encoder_hidden_states=prompt_embeds_full, pooled_projections=pooled_embeddings_full,
                        return_dict=True
                    )
                    output_tensor = outputs.sample if hasattr(outputs, "sample") else outputs[0]
                
                del latent_for_unet, t_for_unet, prompt_embeds_full, outputs
                gc.collect(); torch.cuda.empty_cache()
                
                decoded_latents = (output_tensor / utility_pipeline.vae.config.scaling_factor)
                del output_tensor; gc.collect(); torch.cuda.empty_cache()

                with self.accelerator.autocast():
                    decoded_images_vae_size = utility_pipeline.vae.decode(decoded_latents).sample
                
                del decoded_latents ; gc.collect(); torch.cuda.empty_cache()
                
                decoded_images_orig_size = F.interpolate(
                    decoded_images_vae_size, size=orig_size, mode="bicubic", align_corners=True
                )
                del decoded_images_vae_size, orig_size ; gc.collect(); torch.cuda.empty_cache()

                cond_output_start_index = current_batch_size * (repeat_factor - 1) if repeat_factor > 1 else 0
                decoded_cond_images = decoded_images_orig_size[cond_output_start_index : cond_output_start_index + current_batch_size]
                del decoded_images_orig_size; gc.collect(); torch.cuda.empty_cache()
                
                decoded_cond_f32 = decoded_cond_images.to(torch.float32)
                targets_f32 = targets.to(torch.float32) 
                masks_f32 = masks 

                loss_recon = F.mse_loss(decoded_cond_f32 * masks_f32, targets_f32 * masks_f32)
                loss_ssim = self.ssim_loss(decoded_cond_images, targets) 
                total_loss = loss_recon + lambda_ssim * loss_ssim
                
                total_loss_item = total_loss.item() 
                del decoded_cond_images, decoded_cond_f32, targets_f32, masks_f32, targets, masks, loss_recon, loss_ssim
                gc.collect(); torch.cuda.empty_cache()

                self.accelerator.backward(total_loss)
                optimizer.step()
                optimizer.zero_grad()
                
                utility_pipeline.vae.to("cpu") 
                
                del total_loss
                gc.collect(); torch.cuda.empty_cache()

                epoch_loss_sum += total_loss_item 

            lr_scheduler.step()
            avg_epoch_loss = epoch_loss_sum / len(train_loader) if len(train_loader) > 0 else 0.0
            if self.accelerator.is_main_process:
                print(f"[Model3.lora_train] Epoch {epoch+1}/{num_epochs} Avg Loss: {avg_epoch_loss:.4f}, LR: {lr_scheduler.get_last_lr()[0]:.2e}")
                if avg_epoch_loss < best_epoch_loss and len(train_loader) > 0 :
                    best_epoch_loss = avg_epoch_loss
                    unwrapped_model = self.accelerator.unwrap_model(model_nf4_lora) 
                    actual_save_directory = lora_path
                    os.makedirs(actual_save_directory, exist_ok=True)
                    self.accelerator.wait_for_everyone() 
                    if self.accelerator.is_main_process: 
                        unwrapped_model.save_pretrained(actual_save_directory)
                        print(f"  → LoRA fine-tuned model saved to {actual_save_directory}")
            
            gc.collect(); torch.cuda.empty_cache()

        if self.accelerator.is_main_process: print("[Model3.lora_train] Training complete.")
        
        utility_pipeline.vae.cpu() 
        utility_pipeline.text_encoder.cpu()
        if utility_pipeline.text_encoder_2: utility_pipeline.text_encoder_2.cpu()
        if utility_pipeline.text_encoder_3: utility_pipeline.text_encoder_3.cpu()
        utility_pipeline.transformer.cpu() 
        del utility_pipeline 
        del model_nf4_lora 
        gc.collect(); torch.cuda.empty_cache()

    def inference(self, texted_images_for_model3: list["TextedImage"]):
        if not self.accelerator.is_main_process: return
        print("Loading SD3 pipeline for inference...")

        weight_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
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

            # 전체 파이프라인 로드 (transformer를 LoRA 적용된 것으로 주입)
            pipe = StableDiffusion3Pipeline.from_pretrained(
                model_id,
                transformer=lora_transformer,
                torch_dtype=weight_dtype,
                # safety_checker=None # 필요시 safety_checker 비활성화
            )
            
            current_device = self.accelerator.device
            pipe.to(current_device)

            print(f"SD3 pipeline with custom LoRA transformer loaded successfully to {current_device}.")

        except Exception as e:
            print(f"Error loading SD3 pipeline with LoRA for inference: {e}")
            return
        
        output_dir = self.model_config.get("output_dir", "datas/images/output/model3_sd3_inference")
        os.makedirs(output_dir, exist_ok=True)

        for i, texted_image_item in enumerate(tqdm(texted_images_for_model3, desc="Inference Progress", disable=not self.accelerator.is_main_process)):
            with torch.no_grad(): # 추론 시에는 그래디언트 계산 불필요
                try:
                    original_pil, text_pil, mask_pil = texted_image_item._to_pil()

                    # 1. 원본 이미지와 마스크를 PyTorch 텐서로 변환 및 정규화
                    # VAE 입력 크기 (예: 256x256)에 맞춰 이미지와 마스크 크기 조정
                    # diffusers의 VAE는 보통 8배 다운샘플링하므로, 인페인팅 Latent는 (H/8, W/8)이 됨.
                    image_transforms = transforms.Compose([
                        transforms.ToTensor(), # PIL to Tensor (0-1 range)
                        transforms.Normalize([0.5], [0.5]), # Normalize to -1 to 1 range
                    ])
                    mask_transforms = transforms.Compose([
                        transforms.ToTensor(), # PIL to Tensor (0-1 range)
                        # 마스크는 0과 1 값만 필요
                    ])

                    # 입력 이미지와 마스크를 VAE가 처리할 수 있는 형태의 텐서로 변환
                    # pipe.vae.config.sample_size는 VAE가 기대하는 이미지 크기일 수 있음
                    vae_input_size = pipe.vae.config.sample_size if hasattr(pipe.vae.config, 'sample_size') else 256
                    
                    original_tensor = F.interpolate(
                        image_transforms(original_pil).unsqueeze(0).to(current_device, dtype=weight_dtype),
                        size=(vae_input_size, vae_input_size), mode="bicubic", align_corners=True
                    )
                    mask_tensor = F.interpolate(
                        mask_transforms(mask_pil).unsqueeze(0).to(current_device, dtype=torch.float32), # 마스크는 float32
                        size=(vae_input_size, vae_input_size), mode="nearest" # 마스크는 nearest (이진 값 유지)
                    )
                    
                    # 2. 원본 이미지와 마스크를 Latent Space로 인코딩
                    # VAE를 GPU로 옮겨서 인코딩 수행
                    pipe.vae.to(current_device)
                    # 입력 이미지는 -1 to 1 범위, 마스크는 0-1 (0은 가려짐, 1은 보임)
                    latent_image = pipe.vae.encode(original_tensor).latent_dist.sample() * pipe.vae.config.scaling_factor
                    # 마스크도 Latent Space 크기에 맞춰 조정 (VAE는 8배 다운샘플링이 일반적)
                    mask_downscaled_size = (vae_input_size // 8, vae_input_size // 8) # SD3의 잠재 공간 크기에 맞춤
                    latent_mask = F.interpolate(mask_tensor, size=mask_downscaled_size, mode="nearest")
                    
                    # 3. 마스크를 Latent Space에 적용 (Inpainting)
                    # 마스크가 1이면 원본 유지, 0이면 노이즈 (채워야 할 부분)
                    # 마스크가 반대로 (0이 채울 부분) 되어 있다면 1-mask_tensor 사용
                    # 주어진 마스크는 'Binay pixel-wise mask (1, H, W)' 이며, 이미지에서 흰색 글자, 검은색 배경으로 보입니다.
                    # 즉, 흰색 (1) 부분이 마스크되어 채워져야 하는 영역으로 보입니다.
                    # 따라서 마스크를 반전할 필요 없이 그대로 사용하거나,
                    # 만약 마스크가 채워야 할 부분이면 그대로 사용하고, 보존할 부분이면 반전합니다.
                    # 일반적으로 0이 마스크 영역(채울 부분), 1이 보존할 영역입니다.
                    # 주어진 이미지에서 글자 (흰색) 부분이 마스크로 보입니다. 따라서 글자 부분을 채울 것이므로,
                    # 마스크 텐서에서 1인 부분이 채워질 부분이라고 가정하고 `masked_latent = latent_image * (1 - latent_mask)` 처럼 사용.
                    # 아니면, `mask_pil`이 흰색이 마스크 영역이면 `(1-mask_tensor)`를 Latent 마스크로 사용해야 합니다.
                    # 현재 마스크 이미지는 흰 글자 (1) 검은 배경 (0)이므로, 흰 글자를 채울 것이면 그대로 사용.
                    # Diffusers Inpainting 파이프라인은 보통 0이 노이즈를 추가할 영역 (마스크된 영역), 1이 원본 유지 영역입니다.
                    # 따라서 마스크를 반전하여 0이 글자 영역이 되도록 합니다.

                    # 마스크 (0-1)를 0 또는 1로 명확히
                    latent_mask = (latent_mask > 0.5).float() # 이진 마스크로

                    # 노이즈 생성
                    noise = torch.randn_like(latent_image, device=current_device, dtype=weight_dtype)

                    # 마스크 영역은 노이즈로, 마스크되지 않은 영역은 원본 Latent 이미지로
                    # latent_mask는 채워야 할 영역이 1인 마스크라고 가정합니다.
                    # Stable Diffusion Inpainting 공식 문서에 따르면, 0은 마스크, 1은 원본
                    # 따라서 1-latent_mask를 사용하여 노이즈가 마스크된 영역에 들어가게 합니다.
                    masked_latent = latent_image * (1 - latent_mask) + noise * latent_mask
                    
                    # 마스크 이미지의 채널 수 (SD3 Transformer는 16채널 mask_image_input을 받음)
                    # Stable Diffusion XL과 유사하게, 마스크와 마스크된 이미지를 조건으로 사용
                    # SD3의 경우, `transformer`의 `image_embeddings` 인자가 있을 수 있음.
                    # 여기서는 `transformer`의 `image_embeddings` 인자가 Latent Image + Latent Mask의 concat을 받는다고 가정
                    # (SD3의 `transformer` 입력 형태는 SDXL의 `unet`과 다를 수 있으므로 공식 문서 확인 필요)

                    # Stable Diffusion 3는 기본적으로 Inpainting을 위해 image_latents를 직접 받지 않음
                    # 즉, UNet에 'masked_latent'를 직접 전달하는 방식이 아닐 수 있음.
                    # SD3는 `pooled_projections`와 `encoder_hidden_states`를 통해 텍스트 및 이미지 임베딩을 받음.
                    # Inpainting을 위한 Latent Conditioning은 SD3 아키텍처에 맞춰 재설계되어야 함.
                    # 하지만 지금으로서는 "이미지 전체의 맥락"을 반영하는 가장 간단한 방법으로 시도.
                    # **주의: 이 부분은 SD3의 실제 Inpainting 구현과 다를 수 있습니다.**
                    # 임시 해결책으로, 입력 이미지를 임베딩으로 변환하여 UNet에 전달하는 방식을 시도해 봅니다.
                    
                    # 텍스트 인코더와 VAE를 GPU로 옮김 (추론에 필요)
                    pipe.text_encoder.to(current_device)
                    if pipe.text_encoder_2: pipe.text_encoder_2.to(current_device)
                    if pipe.text_encoder_3: pipe.text_encoder_3.to(current_device)
                    # pipe.vae는 이미 current_device에 있을 가능성 높음

                    prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = pipe.encode_prompt(
                        prompt=prompt,
                        prompt_2=prompt if pipe.text_encoder_2 else None,
                        prompt_3=prompt if pipe.text_encoder_3 else None,
                        device=current_device,
                        num_images_per_prompt=1,
                        do_classifier_free_guidance=True,
                        negative_prompt=negative_prompt,
                        negative_prompt_2=negative_prompt if pipe.text_encoder_2 else None,
                        negative_prompt_3=negative_prompt if pipe.text_encoder_3 else None,
                    )
                    prompt_embeds_full = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
                    pooled_embeddings_full = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

                    # UNet에 전달할 Latent Input (마스크된 Latent)
                    # SD3의 `transformer` (UNet)의 입력 채널은 16개 (기존 SD는 4개).
                    # `masked_latent`의 채널은 4개.
                    # `mask_image_input` (16채널)이 무엇인지 확인 필요.
                    # SD3의 `transformer`는 `hidden_states`, `timestep`, `encoder_hidden_states`, `pooled_projections`를 받음.
                    # Inpainting을 위한 추가 conditioning은 `pooled_projections`나 `encoder_hidden_states`에 통합될 수 있음.
                    # 이 부분을 현재 `StableDiffusion3Pipeline`의 `.prepare_latents()`나 유사 메서드에서 처리해야 함.

                    # 임시방편으로 `masked_latent`를 `hidden_states`로 사용하고,
                    # `transformer` (UNet)의 `image_embeddings` 또는 `pixel_values`에 마스크된 원본 이미지를 전달하는 방식은
                    # Stable Diffusion 3 모델 아키텍처와 호환되지 않을 가능성이 높습니다.
                    # 가장 간단한 해결책은 기존 SD1.x/2.x 파이프라인처럼 Latent space에서 noise를 추가하는 것.

                    # pipe의 scheduler를 사용하여 노이즈 추가
                    num_inference_steps = self.model_config.get("inference_steps", 28)
                    pipe.scheduler.set_timesteps(num_inference_steps, device=current_device)
                    timesteps = pipe.scheduler.timesteps
                    
                    # 마스크된 Latent에 노이즈 스케일링
                    # masked_latent_noisy = pipe.scheduler.add_noise(masked_latent, noise, timesteps[:1]) # 첫 번째 스텝만 노이즈 추가
                    # 이 방식은 일반적인 DDIM/DDPM 추론 과정에서 사용
                    
                    # Inpainting Latent 생성 (SDXL Inpainting 참고)
                    # SDXL에서는 `pipe.prepare_latents`에서 `image`와 `mask_image`를 처리하여 `latents`를 반환
                    # SD3에는 이와 같은 `prepare_latents`가 아직 없을 수 있음.
                    # 임시로, 마스크된 영역에 노이즈를 적용한 Latent를 사용하여 추론 진행

                    # `pipe.transformer`에 전달할 latent_model_input 생성
                    # `latent_image`에 noise를 주입하는 과정은 scheduler가 담당
                    # 여기서 `masked_latent`는 초기 latent로서 사용됨.
                    
                    # UNet에 전달하기 위해 Latent를 준비
                    # CFG를 위한 확장 (conditional + unconditional)
                    # Latent space dimensions are (batch_size, 4, H/8, W/8)
                    # if guidance_scale > 1, batch_size becomes 2 * original_batch_size
                    
                    guidance_scale = self.model_config.get("guidance_scale", 7.0)
                    
                    # pipe의 _get_add_time_ids, _encode_prompt 등을 직접 호출해야 할 수도 있음.
                    # 여기서는 pipeline의 `__call__` 메서드에 인페인팅 파라미터를 넘기는 것으로 시도 (만약 지원한다면)
                    # 하지만 이전처럼 TypeError를 낼 것이므로, SD3는 아직 직접 지원하지 않는 것으로 보고
                    # `image`와 `mask_image`를 인자로 넣지 않고 텍스트 프롬프트 기반으로만 생성
                    # 즉, 기존의 Text-to-Image 방식으로 생성하고, 이후에 수동 합성하는 방식으로 전환
                    
                    # ***** 가장 간단한 Inpainting 워크어라운드 (Masked Composite) *****
                    # 1. 텍스트 프롬프트로 이미지를 생성
                    generated_image_pil = pipe(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        generator=torch.Generator(device=current_device).manual_seed(42 + i),
                    ).images[0]

                    # 2. 마스크를 사용하여 원본 이미지와 생성된 이미지를 합성
                    # 마스크는 흰색 글자, 검은색 배경 (흰색 부분이 채울 영역)
                    # PIL.Image.composite(image1, image2, mask)
                    # image1: 마스크된 영역 (여기서는 generated_image_pil)
                    # image2: 마스크되지 않은 영역 (여기서는 original_pil)
                    # mask: 0 (검정)은 image1에서, 255 (흰색)는 image2에서 가져옴.
                    # 따라서 마스크가 흰색 글자, 검은색 배경이라면, 
                    # generated_image_pil은 글자 영역으로, original_pil은 글자 없는 영역으로.
                    # 마스크를 `invert`해서 글자 없는 영역이 255 (흰색)이 되도록 해야 합니다.
                    from PIL import ImageChops, ImageOps # ImageOps for invert

                    # mask_pil은 (H, W, 1) 이진 마스크
                    # Image.composite는 (H,W)의 L (luminance) 모드 마스크를 기대함
                    # mask_pil은 이미 (H, W) 형태의 단일 채널 이미지로 간주.
                    # 0은 검은색, 255는 흰색. 흰색 글자 영역은 255
                    # `Image.composite`의 마스크는 255가 foreground, 0이 background
                    # 만약 마스크 흰색 영역을 생성된 이미지로 채우고 싶으면 (마스크가 텍스트인 경우):
                    # `mask_pil` (흰색 글자) -> 이 흰색 글자 부분에 generated_image_pil을 적용하고 싶다면.
                    # `Image.composite(generated_image_pil, original_pil, mask_pil.convert('L'))`
                    # 위 코드의 의미: mask_pil의 흰색(글자) 부분은 generated_image_pil에서, 검은색(배경) 부분은 original_pil에서 가져옴.
                    # 즉, 글자가 생성된 이미지로 교체되는 효과.
                    # 이것이 우리가 원하는 인페인팅 효과입니다!

                    inpainted_image_pil = Image.composite(generated_image_pil, original_pil, mask_pil.convert('L'))


                except Exception as e:
                    print(f"Error during SD3 inference for image {i}: {e}")
                    inpainted_image_pil = original_pil # 오류 시 원본 이미지로 대체
                    
            fig, axes = plt.subplots(1, 4, figsize=(20, 5))
            axes[0].imshow(original_pil); axes[0].set_title("Original"); axes[0].axis("off")
            axes[1].imshow(text_pil); axes[1].set_title("Text Input"); axes[1].axis("off")
            axes[2].imshow(mask_pil, cmap='gray'); axes[2].set_title("Mask"); axes[2].axis("off")
            axes[3].imshow(inpainted_image_pil); axes[3].set_title("SD3 LoRA Inpainted Result"); axes[3].axis("off") # 제목 변경

            plt.tight_layout()
            save_filename = os.path.join(output_dir, f"sd3_lora_inference_result_{i}.png")
            plt.savefig(save_filename)
            plt.close(fig)

        print(f"All SD3 LoRA inference results saved to {output_dir}")
        return