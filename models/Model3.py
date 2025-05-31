import torch
import os
import torch.nn.functional as F
from torch import nn
from diffusers import SD3Transformer2DModel, StableDiffusion3Pipeline
from accelerate import Accelerator
from tqdm import tqdm
from pytorch_msssim import ms_ssim
from peft import get_peft_model, LoraConfig
# TextedImage 클래스는 현재 컨텍스트에서 사용 가능하다고 가정합니다.
# from ..datas.TextedImage import TextedImage


def identity_collate(batch):
    return batch

class Model3(nn.Module):
    def __init__(self, model_config: dict):
        super().__init__()
        self.model_config = model_config

    def ssim_loss(self, pred_img, target_img):
        pred_img_0_1 = (pred_img.clamp(-1, 1) + 1) / 2
        target_img_0_1 = (target_img.clamp(-1, 1) + 1) / 2
        pred_img_f = pred_img_0_1.to(torch.float32)
        target_img_f = target_img_0_1.to(torch.float32)
        ssim_val = ms_ssim(pred_img_f, target_img_f, data_range=1.0, size_average=False)
        return 1.0 - ssim_val.mean()

    def lora_train(self, texted_images_for_model3: list["TextedImage"], accelerator: Accelerator):
        if accelerator.is_main_process:
            print("\n[Model3.lora_train] Entered lora_train method.")

        # ───────────────────────────────────────────────
        # STEP 1. 모델/파이프라인 로드 및 사전 설정
        # ───────────────────────────────────────────────
        model_id = self.model_config["model_id"]
        batch_size = self.model_config["batch_size"]
        lr = self.model_config["lr"]
        weight_decay = self.model_config.get("weight_decay", 3e-4)
        lora_rank = self.model_config["lora_rank"]
        lora_alpha = self.model_config["lora_alpha"]
        lambda_ssim = self.model_config.get("lambda_ssim", 0.5)
        lora_path = self.model_config["lora_path"]
        num_epochs = self.model_config["epochs"]
        
        prompts_config_str = self.model_config["prompts"]
        negative_prompts_config_str = self.model_config["negative_prompt"]

        model_nf4: SD3Transformer2DModel = SD3Transformer2DModel.from_pretrained(
            model_id,
            subfolder="transformer",
            torch_dtype=torch.float16
        )

        pipeline: StableDiffusion3Pipeline = StableDiffusion3Pipeline.from_pretrained(
            model_id,
            transformer=model_nf4,
            torch_dtype=torch.float16
        )

        text_encoder_1 = pipeline.text_encoder
        text_encoder_2 = pipeline.text_encoder_2
        text_encoder_3 = pipeline.text_encoder_3

        if text_encoder_1 is None or text_encoder_2 is None or text_encoder_3 is None:
            missing_encoders = []
            if text_encoder_1 is None: missing_encoders.append("text_encoder_1 (CLIP-ViT/G)")
            if text_encoder_2 is None: missing_encoders.append("text_encoder_2 (CLIP-ViT/L)")
            if text_encoder_3 is None: missing_encoders.append("text_encoder_3 (T5-XXL)")
            raise RuntimeError(
                f"[ERROR] The following text encoders are None: {', '.join(missing_encoders)}. "
                "StableDiffusion3Pipeline.from_pretrained likely did not load them correctly."
            )
        
        vae = accelerator.prepare(pipeline.vae)
        text_encoder_1 = text_encoder_1.to(accelerator.device, dtype=torch.float16).eval()
        text_encoder_2 = text_encoder_2.to(accelerator.device, dtype=torch.float16).eval()
        text_encoder_3 = text_encoder_3.to(accelerator.device, dtype=torch.float16).eval()

        # ───────────────────────────────────────────────
        # STEP 2. UNet(Transformer) 파라미터 freeze + LoRA 적용 + Optimizer/Scheduler 준비
        # ───────────────────────────────────────────────
        for p in model_nf4.parameters():
            p.requires_grad = False

        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
            lora_dropout=0.1,
            bias="none",
            init_lora_weights="gaussian"
        )
        
        model_nf4 = get_peft_model(model_nf4, lora_config)
        if accelerator.is_main_process:
            model_nf4.print_trainable_parameters()

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model_nf4.parameters()),
            lr=lr,
            weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.95)
        optimizer, scheduler = accelerator.prepare(optimizer, scheduler)

        # ───────────────────────────────────────────────
        # STEP 3. DataLoader 생성
        # ───────────────────────────────────────────────
        train_loader = torch.utils.data.DataLoader(
            texted_images_for_model3,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=identity_collate
        )
        train_loader = accelerator.prepare(train_loader)

        # ───────────────────────────────────────────────
        # STEP 4. LoRA 적용된 UNet을 accelerator로 옮기기 (GPU)
        # ───────────────────────────────────────────────
        model_nf4 = accelerator.prepare(model_nf4)

        best_epoch_loss = float("inf")

        # ───────────────────────────────────────────────
        # STEP 5. Training Loop (float16 연산)
        # ───────────────────────────────────────────────
        model_nf4.train()
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            if accelerator.is_main_process:
                print(f"\n[STEP 5] Epoch {epoch+1}/{num_epochs} 시작")

            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", disable=not accelerator.is_main_process)

            for step, batch in enumerate(progress_bar):
                current_batch_size = len(batch)
                prompts_list = [prompts_config_str] * current_batch_size
                negative_prompts_list = [negative_prompts_config_str] * current_batch_size

                inputs = torch.stack([x.timg for x in batch])
                targets = torch.stack([x.orig for x in batch])
                masks = torch.stack([x.mask for x in batch])

                inputs = (inputs * 2.0) - 1.0
                targets = (targets * 2.0) - 1.0

                inputs = inputs.to(accelerator.device, dtype=torch.float16)
                targets = targets.to(accelerator.device, dtype=torch.float16)
                masks = masks.to(accelerator.device, dtype=torch.float32)
                
                orig_size = inputs.shape[-2:]
                vae_input_images = F.interpolate(
                    inputs,
                    size=(512, 512),
                    mode="bicubic",
                    align_corners=True
                )

                # 5.2. Text Prompt 준비 & encode_prompt 호출
                with torch.no_grad():
                    actual_prompt_embeds, actual_negative_prompt_embeds, \
                    actual_pooled_prompt_embeds, actual_negative_pooled_prompt_embeds = pipeline.encode_prompt(
                        prompt=prompts_list,
                        prompt_2=prompts_list,
                        prompt_3=prompts_list,
                        device=accelerator.device,
                        num_images_per_prompt=1,
                        do_classifier_free_guidance=True,
                        negative_prompt=negative_prompts_list,
                        negative_prompt_2=negative_prompts_list,
                        negative_prompt_3=negative_prompts_list
                    )

                cond_prompt_embeds = actual_prompt_embeds
                uncond_prompt_embeds = actual_negative_prompt_embeds
                cond_pooled = actual_pooled_prompt_embeds
                uncond_pooled = actual_negative_pooled_prompt_embeds
                
                if accelerator.is_main_process and step == 0:
                    print(f"      [5.2] Shapes after encode_prompt:")
                    print(f"            cond_prompt_embeds.shape   = {cond_prompt_embeds.shape}, dtype={cond_prompt_embeds.dtype}")
                    print(f"            uncond_prompt_embeds.shape = {uncond_prompt_embeds.shape}, dtype={uncond_prompt_embeds.dtype}")
                    print(f"            cond_pooled.shape          = {cond_pooled.shape}, dtype={cond_pooled.dtype}")
                    print(f"            uncond_pooled.shape        = {uncond_pooled.shape}, dtype={uncond_pooled.dtype}")

                prompt_embeds_full = torch.cat([uncond_prompt_embeds, cond_prompt_embeds], dim=0)
                pooled_embeddings_full = torch.cat([uncond_pooled, cond_pooled], dim=0)
                
                if accelerator.is_main_process and step == 0:
                    print(f"      [5.3] Shapes after concatenation for CFG:")
                    print(f"            prompt_embeds_full.shape   = {prompt_embeds_full.shape}")
                    print(f"            pooled_embeddings_full.shape= {pooled_embeddings_full.shape}")

                # 5.4. VAE 인코딩 (no_grad)
                with torch.no_grad():
                    vae_out = vae.encode(vae_input_images)
                    # vae_out.latent_dist.sample() (float16) * vae.config.scaling_factor (Python float) -> float32
                    # UNet에 float16 입력을 보장하기 위해 명시적으로 .to(torch.float16) 캐스팅
                    latent = (vae_out.latent_dist.sample() * vae.config.scaling_factor).to(torch.float16)
                
                if accelerator.is_main_process and step == 0: # latent 타입 확인용 디버그 프린트 (선택사항)
                    print(f"      [5.4] latent.dtype after VAE encoding and cast = {latent.dtype}")


                max_train_timesteps = self.model_config.get("max_train_timesteps", 1000)
                t = torch.randint(0, max_train_timesteps, (current_batch_size,), device=accelerator.device, dtype=torch.long)

                # 5.6. LoRA 적용된 UNet(Transformer) forward
                if accelerator.is_main_process and step == 0:
                    print(f"      [5.6] UNet forward input shapes:")
                    print(f"            latent.shape                 = {latent.shape}")
                    print(f"            t.shape                      = {t.shape}")
                    print(f"            pooled_embeddings_full.shape = {pooled_embeddings_full.shape}")
                    print(f"            prompt_embeds_full.shape     = {prompt_embeds_full.shape}")
                
                outputs = model_nf4(
                    hidden_states=latent, # 이제 latent는 확실히 float16 입니다.
                    timestep=t,
                    encoder_hidden_states=prompt_embeds_full,
                    pooled_projections=pooled_embeddings_full,
                    return_dict=True
                )
                output_tensor = outputs.sample if hasattr(outputs, "sample") else outputs[0]

                if accelerator.is_main_process and step == 0:
                     print(f"      [5.6] UNet forward output_tensor.shape = {output_tensor.shape}, dtype={output_tensor.dtype}")


                # 5.7. VAE 디코딩 & 원래 크기 복원 (torch.no_grad() 없음)
                temp_decoded_latents_float32 = output_tensor / vae.config.scaling_factor
                decoded_latents = temp_decoded_latents_float32.to(torch.float16)

                if accelerator.is_main_process and step == 0:
                    print(f"      [5.7 Pre-Decode] decoded_latents.dtype = {decoded_latents.dtype}")

                decoded_images_vae_size = vae.decode(decoded_latents).sample
                
                decoded_images_orig_size = F.interpolate(
                    decoded_images_vae_size,
                    size=orig_size,
                    mode="bicubic",
                    align_corners=True
                )

                # 5.8. Loss 계산
                decoded_cond_images = decoded_images_orig_size[current_batch_size:]
                
                decoded_cond_f32 = decoded_cond_images.to(torch.float32)
                targets_f32 = targets.to(torch.float32)
                masks_f32 = masks.to(torch.float32)

                loss_recon = F.mse_loss(decoded_cond_f32 * masks_f32, targets_f32 * masks_f32)
                loss_ssim = self.ssim_loss(decoded_cond_images, targets)
                total_loss = loss_recon + lambda_ssim * loss_ssim

                # 5.9. Backward 및 Optimizer step
                accelerator.backward(total_loss)
                optimizer.step()
                optimizer.zero_grad()

                epoch_loss += total_loss.item()
                progress_bar.set_postfix(loss=total_loss.item(), recon=loss_recon.item(), ssim=loss_ssim.item())
            
            # End of epoch
            avg_epoch_loss = epoch_loss / len(train_loader)
            if accelerator.is_main_process:
                print(f"[STEP 5 End] Epoch {epoch+1}/{num_epochs} Avg Loss: {avg_epoch_loss:.4f}")

                if avg_epoch_loss < best_epoch_loss:
                    best_epoch_loss = avg_epoch_loss
                    unwrapped_model = accelerator.unwrap_model(model_nf4)
                    actual_save_directory = lora_path
                    os.makedirs(actual_save_directory, exist_ok=True)
                    unwrapped_model.save_pretrained(actual_save_directory)
                    print(f"→ LoRA fine-tuned model saved to {actual_save_directory}")
            scheduler.step()

        if accelerator.is_main_process:
            print("[Model3.lora_train] Training complete.")

    def inference(self, texted_images_for_model3: list["TextedImage"]):
        # accelerator 인스턴스가 이 메소드 내에서 정의되거나 전달되어야 합니다.
        # if accelerator.is_main_process:
        print("\n[Model3.inference] Entered inference method.")
        # TODO: Implement inference logic
        pass