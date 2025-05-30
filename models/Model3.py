import torch
import os
import matplotlib.pyplot as plt
from torch import nn
from diffusers import StableDiffusionInpaintPipeline
from peft import LoraConfig, get_peft_model
import torch.nn.functional as F
from accelerate import Accelerator
from tqdm import tqdm
from ..datas.TextedImage import TextedImage
from pytorch_msssim import ms_ssim
from PIL import Image
import math
from torch.amp import autocast
import gc

class Model3(nn.Module):
    """Masked Inpainting Model
    config: 모델 설정 정보 dictionary
    
    """
    def __init__(self, model_config: dict):
        super().__init__()
        self.model_config = model_config
        
    @staticmethod
    def _get_add_time_ids(original_size, crops_coords_top_left, target_size, dtype):
        """
        Helper function to create add_time_ids for SDXL.
        Based on diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl.StableDiffusionXLPipeline._get_add_time_ids
        """
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        return add_time_ids
        
    
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
        # 1. 파이프라인 로드 (CPU에 먼저 로드하여 메모리 부담 분산)
        print("Loading pipeline components to CPU first...")
        try:
            # 기본 SD 1.5 모델을 사용하여 각 컴포넌트를 별도로 로드 (인페인팅 모델 대신)
            from diffusers import UNet2DConditionModel, AutoencoderKL
            from transformers import CLIPTextModel, CLIPTokenizer
            from diffusers import DDPMScheduler
            
            # 텍스트 인코더와 토크나이저는 SD 1.5 기본 모델에서 로드
            text_encoder = CLIPTextModel.from_pretrained(
                "stable-diffusion-v1-5/stable-diffusion-v1-5", 
                subfolder="text_encoder",
                torch_dtype=torch.float16
            )
            tokenizer = CLIPTokenizer.from_pretrained(
                "stable-diffusion-v1-5/stable-diffusion-v1-5", 
                subfolder="tokenizer"
            )
            
            # VAE 로드
            vae = AutoencoderKL.from_pretrained(
                "stable-diffusion-v1-5/stable-diffusion-v1-5", 
                subfolder="vae",
                torch_dtype=torch.float16
            )
            
            # UNet 로드 (인페인팅 모델 아닌 기본 UNet)
            print("Loading original UNet for modification...")
            unet = UNet2DConditionModel.from_pretrained(
                self.model_config["model_id"], 
                subfolder="unet",
                torch_dtype=torch.float16
            )
            print(f"Original UNet in_channels: {unet.conv_in.in_channels}")

            # 새로운 conv_in 레이어 생성 (9 채널 입력)
            new_in_channels = 9 # 4 (noisy_latents) + 1 (mask) + 4 (masked_image_latents_for_conditioning)
            original_in_channels = unet.conv_in.in_channels # 보통 4
            out_channels = unet.conv_in.out_channels
            kernel_size = unet.conv_in.kernel_size
            stride = unet.conv_in.stride
            padding = unet.conv_in.padding

            new_conv_in = nn.Conv2d(
                new_in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
            ).to(unet.conv_in.weight.dtype) # 원본 레이어와 동일한 dtype으로 설정

            # 새 conv_in 레이어 가중치 초기화:
            with torch.no_grad():
                # 원본 가중치 복사 (처음 original_in_channels 만큼)
                new_conv_in.weight.data[:, :original_in_channels, :, :] = unet.conv_in.weight.data.clone()
                
                # 추가된 채널에 대한 가중치는 0으로 초기화
                if new_in_channels > original_in_channels:
                    nn.init.zeros_(new_conv_in.weight.data[:, original_in_channels:, :, :])
                
                # Bias 처리
                if unet.conv_in.bias is not None:
                    new_conv_in.bias.data = unet.conv_in.bias.data.clone()
                elif new_conv_in.bias is not None: # 원본에는 bias 없고 새거에 있으면 0으로
                    nn.init.zeros_(new_conv_in.bias.data)
            
            unet.conv_in = new_conv_in
            print(f"Modified UNet in_channels: {unet.conv_in.in_channels}, dtype: {unet.conv_in.weight.dtype}")
            
            # 스케줄러 로드
            scheduler = DDPMScheduler.from_pretrained(
                "stable-diffusion-v1-5/stable-diffusion-v1-5", 
                subfolder="scheduler"
            )
            
            print("모든 컴포넌트를 성공적으로 로드했습니다.")
        except Exception as e:
            print(f"모델 컴포넌트 로드 중 오류 발생: {e}")
            print("대체 모델로 시도합니다...")
            
            # 대체 방법: 완전한 파이프라인 로드 후 컴포넌트 추출
            from diffusers import StableDiffusionPipeline
            
            pipe = StableDiffusionPipeline.from_pretrained(
                "stable-diffusion-v1-5/stable-diffusion-v1-5",  # 모델 ID 변경
                torch_dtype=torch.float16,
                safety_checker=None
            )
            
            # 필요한 구성 요소 추출
            vae = pipe.vae
            unet = pipe.unet
            text_encoder = pipe.text_encoder
            tokenizer = pipe.tokenizer
            scheduler = pipe.scheduler
            
            # 원본 파이프라인 객체 삭제
            del pipe
        
        # 이 지점에서 필요한 모든 컴포넌트(vae, unet, text_encoder, tokenizer, scheduler)가 로드됨
        
        accelerator.wait_for_everyone()
        torch.cuda.empty_cache()
        
        # 스케줄러 부분 수정 (나눗셈 오류 방지)
        # total_steps 최소값 확인
        total_steps = max(2, self.model_config["epochs"] * ((len(texted_images_for_model3) + self.model_config["batch_size"] - 1) // self.model_config["batch_size"]))
        warmup_steps = min(int(total_steps * 0.05), total_steps - 1)  # 워밍업 단계가 total_steps보다 작게
        
        print(f"Training schedule: total_steps={total_steps}, warmup_steps={warmup_steps}")
        
        # 4. 각 구성요소를 accelerator.device로 이동
        print("Moving components to device...")
        vae = vae.to(accelerator.device)
        text_encoder = text_encoder.to(accelerator.device)
        
        # VAE, text encoder 등은 학습하지 않음
        vae.eval()
        text_encoder.eval()
        
        # LoRA Config 생성
        config_target_modules = ["to_q", "to_k", "to_v", "to_out.0"]
        lora_config = LoraConfig(
            r=self.model_config["lora_rank"],
            lora_alpha=self.model_config["lora_alpha"],
            target_modules=config_target_modules,
            lora_dropout=0.05,
            bias="none",
        )
        
        # 기존 UNet에 LoRA 적용
        unet = get_peft_model(unet, lora_config)
        unet.print_trainable_parameters()
        
        # 1. 기본 옵티마이저 설정
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, unet.parameters()),  # 학습 가능한 파라미터만 선택
            lr=self.model_config["lr"],
            betas=(0.9, 0.999),
            weight_decay=1e-2,
            eps=1e-8
        )

        # 중요: accelerator.prepare()로 감싸기 전에 스케줄러의 필요한 값을 저장해둡니다
        # 이렇게 하면 래핑된 스케줄러에서 접근이 어려운 속성을 미리 캐싱할 수 있습니다
        num_train_timesteps = scheduler.config.num_train_timesteps
        original_scheduler = scheduler  # 원본 스케줄러 참조 저장
        
        # 2. 점진적으로 감소하는 학습률 스케줄러
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            else:
                # 0으로 나누는 문제 방지
                return 0.5 * (1.0 + math.cos(math.pi * (current_step - warmup_steps) / max(1, total_steps - warmup_steps)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

        # accelerator로 준비
        unet, optimizer, scheduler = accelerator.prepare(unet, optimizer, scheduler)
        
        # UNet에 gradient checkpointing 활성화 (메모리 효율성 향상)
        unet.train()
        unet.enable_gradient_checkpointing()
        
        # 학습 루프
        global_step = 0
        best_loss = float('inf')
        
        # 학습 루프 시작 전 캐시 비우기
        accelerator.wait_for_everyone()
        torch.cuda.empty_cache()
        
        dataset_size = len(texted_images_for_model3)
        batch_size = self.model_config["batch_size"]
        
        # 학습 메인 루프
        for epoch in range(self.model_config["epochs"]):
            # 에포크 시작 시 캐시 비우기
            accelerator.wait_for_everyone()
            torch.cuda.empty_cache()
            print(f"CUDA cache cleared at the beginning of epoch {epoch+1}/{self.model_config['epochs']}.")
            
            # 이미지 인덱스를 섞어서 에포크마다 다른 순서로 학습
            indices = torch.randperm(dataset_size).tolist()
            
            # 배치 단위로 처리
            num_batches = (dataset_size + batch_size - 1) // batch_size
            batch_iterator = tqdm(range(num_batches), 
                                desc=f"Epoch {epoch+1}/{self.model_config['epochs']}",
                                disable=not accelerator.is_main_process)
            
            epoch_loss = 0.0
            
            for batch_idx in batch_iterator:
                with accelerator.accumulate(unet):
                    # 현재 배치에 해당하는 인덱스 계산
                    start_idx = batch_idx * batch_size
                    end_idx = min(start_idx + batch_size, dataset_size)
                    batch_indices = indices[start_idx:end_idx]
                    
                    # 현재 배치의 이미지만 처리
                    batch_items = [texted_images_for_model3[i] for i in batch_indices]
                    if not batch_items:
                        continue
                    
                    # --- 2. 데이터 준비 및 전처리 ---
                    # 각 이미지를 하나씩 GPU로 이동하고 처리
                    original_pixel_values_list = []
                    masks_for_pixel_space_list = []
                    
                    for item in batch_items:
                        # CPU에서 처리하다가 필요할 때만 GPU로 이동
                        orig = item.orig.to(accelerator.device, dtype=torch.float32)
                        mask = item.mask.to(accelerator.device, dtype=torch.float32)
                        
                        # [0,1] -> [-1,1] 변환
                        orig = orig * 2.0 - 1.0
                        
                        original_pixel_values_list.append(orig)
                        masks_for_pixel_space_list.append(mask)
                    
                    # 배치로 스택
                    original_pixel_values = torch.stack(original_pixel_values_list)
                    masks_for_pixel_space = torch.stack(masks_for_pixel_space_list)
                    
                    # 마스크 적용하여 텍스트 영역을 제거한 이미지 생성
                    masked_pixel_values = original_pixel_values * (1 - masks_for_pixel_space)
                    
                    # VAE 인코딩
                    with torch.no_grad():
                        gt_latents = vae.encode(original_pixel_values.to(vae.dtype)).latent_dist.sample()
                        gt_latents = gt_latents * vae.config.scaling_factor
                        
                        masked_image_latents = vae.encode(masked_pixel_values.to(vae.dtype)).latent_dist.sample()
                        masked_image_latents = masked_image_latents * vae.config.scaling_factor
                    
                    # 잠재 공간에서의 마스크 크기 조정
                    latent_H, latent_W = gt_latents.shape[2], gt_latents.shape[3]
                    resized_masks_for_latents = F.interpolate(masks_for_pixel_space, size=(latent_H, latent_W), mode='nearest')

                    # --- UNet 입력 및 타겟 노이즈 준비 ---
                    # 1. 컨디셔닝을 위한 마스크 처리된 이미지의 잠재 공간 (masked_image_latents는 이미 계산됨)
                    # masked_image_latents 사용

                    # 2. UNet 입력의 첫 4채널: 마스크된 영역은 노이즈로, 나머지는 원본 latents로 채우고, 여기에 스케줄러 노이즈 추가
                    noise_for_masked_area = torch.randn_like(gt_latents) # 마스크 영역을 채울 노이즈
                    initial_latents = gt_latents * (1 - resized_masks_for_latents) + noise_for_masked_area * resized_masks_for_latents
                    
                    # 3. 스케줄러 노이즈 (UNet이 예측해야 할 타겟)
                    target_noise = torch.randn_like(initial_latents) 
                    bsz = initial_latents.shape[0]
                    timesteps = torch.randint(0, num_train_timesteps, (bsz,), device=initial_latents.device).long()
                    noisy_initial_latents = original_scheduler.add_noise(initial_latents, target_noise, timesteps)
                    
                    # 4. UNet 입력 준비 (9 채널)
                    unet_sample_input = torch.cat([noisy_initial_latents, resized_masks_for_latents, masked_image_latents], dim=1)
                    
                    # 텍스트 임베딩 준비
                    with torch.no_grad():
                        prompts = [self.model_config["prompts"]] * bsz
                        tokenized_prompt = tokenizer(
                            prompts, 
                            padding="max_length", 
                            max_length=tokenizer.model_max_length,
                            truncation=True, 
                            return_tensors="pt"
                        ).input_ids.to(accelerator.device)
                        encoder_hidden_states = text_encoder(tokenized_prompt)[0]
                    
                    # --- 3. 모델 예측 ---
                    with autocast('cuda'):
                        model_pred_noise = unet(
                            unet_sample_input, 
                            timesteps,
                            encoder_hidden_states=encoder_hidden_states
                        ).sample
                        
                        # --- 4. 손실 계산 --- (타겟은 target_noise)
                        mse_loss = F.mse_loss(
                            model_pred_noise, 
                            target_noise, # UNet은 스케줄러가 추가한 target_noise를 예측
                            reduction="mean"
                        )
                    
                    # --- 5. 역전파 및 업데이트 ---
                    accelerator.backward(mse_loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(unet.parameters(), self.model_config.get("max_grad_norm", 1.0))
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    
                    global_step += 1
                    epoch_loss += mse_loss.detach().item()
                    
                    # --- 6. 로깅 및 체크포인트 저장 ---
                    if accelerator.is_main_process:
                        batch_loss = mse_loss.item()
                        batch_iterator.set_postfix(loss=f"{batch_loss:.4f}")
                        
                        # 손실이 개선되었을 때 모델 저장 (배치마다 체크)
                        if batch_loss < best_loss:
                            best_loss = batch_loss
                            
                            # 저장 경로 확인 및 생성
                            os.makedirs(self.model_config["lora_path"], exist_ok=True)
                            save_path = os.path.join(self.model_config["lora_path"], "best_model.safetensors")
                            
                            # 언래핑 후 저장
                            unwrapped_unet = accelerator.unwrap_model(unet)
                            unwrapped_unet.save_pretrained(save_path)
                            print(f"LoRA weights saved to {save_path} (step {global_step}, loss: {best_loss:.4f})")
                
                # 배치 처리 후 불필요한 텐서 해제
                del original_pixel_values, masks_for_pixel_space, masked_pixel_values
                del gt_latents, masked_image_latents, resized_masks_for_latents
                del noise_for_masked_area, initial_latents, target_noise
                del noisy_initial_latents, unet_sample_input, model_pred_noise
                torch.cuda.empty_cache()

                if global_step % 10 == 0 and accelerator.is_main_process:
                    current_lr = optimizer.param_groups[0]['lr']
                    print(f"Step {global_step}: lr = {current_lr:.6f}, loss = {batch_loss:.4f}")
            
            # 에포크 종료 후 평균 손실 출력
            avg_epoch_loss = epoch_loss / num_batches
            print(f"Epoch {epoch+1} completed: avg_loss = {avg_epoch_loss:.4f}")
            
            # 더 적극적으로 메모리 확보
            torch.cuda.empty_cache()
            # 가비지 컬렉션 강제 실행
            gc.collect()

    def inference(self, texted_images_for_model3: list[TextedImage]):
        print("Loading pipeline components to CPU first...")
        try:
            # StableDiffusionXLInpaintPipeline 대신 StableDiffusionInpaintPipeline 사용
            pipe = StableDiffusionInpaintPipeline.from_pretrained(
                "stable-diffusion-v1-5/stable-diffusion-v1-5",  # 모델 ID 변경
                torch_dtype=torch.float16, 
                safety_checker=None
            )
        except Exception as e:
            print(f"인페인팅 모델 로드 중 오류 발생: {e}")
            print("기본 SD 모델에서 인페인팅 파이프라인을 생성합니다...")
            
            # 오류 발생 시 대안: 기본 SD 모델로부터 인페인팅 파이프라인 생성
            from diffusers import StableDiffusionPipeline
            
            # 먼저 기본 파이프라인 로드
            base_pipe = StableDiffusionPipeline.from_pretrained(
                "stable-diffusion-v1-5/stable-diffusion-v1-5",  # 모델 ID 변경
                torch_dtype=torch.float16,
                safety_checker=None
            )
            
            # 인페인팅 파이프라인으로 변환
            pipe = StableDiffusionInpaintPipeline(**base_pipe.components)
            del base_pipe
        
        # LoRA 로드 - 여기서 발생하는 "No LoRA keys associated..." 경고는 정상적이며 무시 가능
        try:
            lora_path = os.path.join(self.model_config["lora_path"], self.model_config["lora_weight_name"])
            if os.path.exists(lora_path):
                pipe.load_lora_weights(lora_path)
                # LoRA 가중치를 기본 모델에 융합 (선택적) - 메모리 사용량 감소, 추론 속도 향상
                pipe.fuse_lora()
                print("LoRA 가중치를 성공적으로 로드하고 융합했습니다.")
            else:
                print(f"경고: LoRA 가중치 파일을 찾을 수 없습니다: {lora_path}")
        except Exception as e:
            print(f"LoRA 가중치 로드 중 오류 발생: {e}")
        
        pipe.to("cuda")
        
        # 추론 루프
        # 결과를 저장할 디렉토리 생성 (model_config에 정의된 output_dir 사용)
        output_dir = self.model_config.get("output_dir", "datas/images/output/model3_inference_viz") # 기본값 설정
        os.makedirs(output_dir, exist_ok=True)

        for i, texted_image_item in enumerate(tqdm(texted_images_for_model3, desc="Inference Progress")):
            # TextedImage에서 PIL 이미지 가져오기
            try:
                original_pil = texted_image_item.orig_pil # 원본 PIL (TextedImage에 있다고 가정)
                text_pil = texted_image_item.timg_pil     # 텍스트 합성된 PIL (TextedImage에 있다고 가정)
                mask_pil = texted_image_item.mask_pil     # 마스크 PIL (TextedImage에 있다고 가정)
            except AttributeError:
                # TextedImage에 orig_pil, timg_pil, mask_pil이 직접 없는 경우 _to_pil() 활용
                original_pil, text_pil, mask_pil = texted_image_item._to_pil()

            # 파이프라인 입력 준비
            prompt = self.model_config["prompts"]
            negative_prompt = self.model_config.get("negative_prompt", "")

            # 추론 실행 (GPU에서 실행)
            with torch.no_grad():
                try:
                    # SD 1.5 Inpainting 파이프라인 호출 (SDXL과 파라미터 구성이 다름)
                    inpainted_image_pil = pipe(
                        prompt=prompt,
                        image=original_pil,  # 원본 이미지
                        mask_image=mask_pil, # 마스크 이미지 (1: 인페인팅할 영역, 0: 보존할 영역)
                        negative_prompt=negative_prompt,
                        num_inference_steps=30,  # 성능과 속도의 균형을 위해 30으로 조정
                        guidance_scale=7.5,     # SD 1.5의 일반적인 가이던스 스케일 값
                    ).images[0]
                except Exception as e:
                    print(f"Error during inference for image {i}: {e}")
                    # 오류 발생 시 원본 이미지 사용
                    inpainted_image_pil = original_pil
                    continue

            # 시각화 및 저장
            fig, axes = plt.subplots(1, 4, figsize=(20, 5))

            axes[0].imshow(original_pil)
            axes[0].set_title("Original")
            axes[0].axis("off")

            axes[1].imshow(text_pil)
            axes[1].set_title("Text Image (Input for Model1/2)")
            axes[1].axis("off")

            axes[2].imshow(mask_pil, cmap='gray')
            axes[2].set_title("Mask")
            axes[2].axis("off")

            axes[3].imshow(inpainted_image_pil)
            axes[3].set_title("Inpainted Result")
            axes[3].axis("off")

            plt.tight_layout()
            save_filename = os.path.join(output_dir, f"inpainted_result_{i}.png")
            plt.savefig(save_filename)
            plt.close(fig)
            # print(f"Visualization saved to {save_filename}") # tqdm 사용 시 중복 출력 방지 가능

        print(f"All inference results saved to {output_dir}")
        return
        
            
            
            
        
        

            
            
            
