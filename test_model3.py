#!/usr/bin/env python3
"""
Stable Diffusion Inpainting 모델 테스트 스크립트
"""

import torch
from diffusers import StableDiffusionInpaintPipeline

def test_sd_inpainting():
    """Stable Diffusion Inpainting 파이프라인 로딩 테스트"""
    try:
        print("Testing Stable Diffusion Inpainting pipeline loading...")

        # GPU 사용 가능 여부 확인
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # Stable Diffusion 2.0 Inpainting 파이프라인 로드 시도
        print("Loading StableDiffusionInpaintPipeline...")
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-inpainting",
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        )

        print("Moving pipeline to device...")
        pipe.to(device)

        print("✅ Stable Diffusion Inpainting pipeline loaded successfully!")

        # 메모리 정리
        del pipe
        torch.cuda.empty_cache() if device.type == "cuda" else None

        return True

    except Exception as e:
        print(f"❌ Error loading Stable Diffusion Inpainting pipeline: {e}")
        return False

if __name__ == "__main__":
    success = test_sd_inpainting()
    if success:
        print("\n🎉 Stable Diffusion Inpainting 모델이 성공적으로 로드되었습니다!")
        print("Model3.py에서 사용할 수 있습니다.")
    else:
        print("\n⚠️  Stable Diffusion Inpainting 모델 로딩에 실패했습니다.")
        print("diffusers 버전을 확인하거나 다른 모델을 사용해보세요.")
