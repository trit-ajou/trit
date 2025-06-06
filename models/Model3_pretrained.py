import torch
import torch.nn as nn
import os
import numpy as np
import gc
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from ..datas.TextedImage import TextedImage

# 부동소수점 행렬 곱셈 정밀도 설정
torch.set_float32_matmul_precision('high')
print("부동소수점 행렬 곱셈 정밀도를 'high'로 설정")
class Model3_pretrained(nn.Module):
    def __init__(self, model_config: dict, device: str = "cuda"):
        super().__init__()
        self.model_config = model_config
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

    def lora_train(self, texted_images_for_model3: list[TextedImage]):
        """
        Stable Diffusion Inpainting 2.0 은 사전훈련된 모델을 사용하므로 별도 훈련이 필요하지 않습니다.
        이 메서드는 호환성을 위해 유지됩니다.
        """
        print("Stable Diffusion Inpainting 모델은 사전훈련된 모델을 사용합니다.")
        print("별도의 LoRA 훈련이 필요하지 않습니다.")
        return

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
            prompt = self.model_config.get("prompts", "pure black and white manga style image with no color tint, absolute grayscale, contextual manga style")
            negative_prompt = self.model_config.get("negative_prompt", "deformed, distorted, disfigured, poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, mutated hands and fingers, disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation, NSFW, photo, realistic, color, colorful, purple, violet, sepia, any color tint")
            guidance_scale = self.model_config.get("guidance_scale", 7.5)
            num_inference_steps = self.model_config.get("inference_steps", 28)

            # 2. Stable Diffusion 2.0 Inpainting 파이프라인 로드 (고품질 만화 스타일에 적합)
            pipe = StableDiffusionInpaintPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
            )

            # 3. 모델을 GPU로 이동
            pipe.to(self.device)

            # 3.1 VAE만 fp32로 변경 (재구성 품질 향상)
            pipe.vae = pipe.vae.to(dtype=torch.float32)

            # 4. 출력 디렉토리 설정
            output_dir = self.model_config.get("output_dir", "trit/datas/images/output")
            os.makedirs(output_dir, exist_ok=True)

            # 5. 각 패치 처리
            for i, current_patch in enumerate(tqdm(texted_images_to_inpaint, desc="Inpainting patches")):
                try:
                    # VRAM 관리
                    torch.cuda.empty_cache()

                    # 원본 이미지와 마스크를 PIL로 변환
                    to_pil = transforms.ToPILImage()
                    orig_pil = to_pil(current_patch.orig.cpu())
                    mask_pil = to_pil(current_patch.mask.cpu().squeeze(0))

                    # 마스크 이진화 (0 또는 255)
                    mask_np = np.array(mask_pil)
                    mask_binary = (mask_np > 127).astype(np.uint8) * 255
                    mask_binary_pil = Image.fromarray(mask_binary, "L")

                    # 이미지 크기 가져오기
                    width, height = orig_pil.size

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

            # 6. 메모리 정리
            del pipe
            gc.collect()
            torch.cuda.empty_cache()

            print(f"Inference completed. Results saved to {output_dir}")
            return texted_images_to_inpaint

        except Exception as e:
            print(f"Error initializing Stable Diffusion Inpainting pipeline: {e}")
            return texted_images_to_inpaint