import torch.nn.functional as F
from tqdm import tqdm
import os
import random
import torch
import matplotlib.pyplot as plt


def train_one_epoch_model1(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    num_epochs: int,
    vis_save_dir: str,
    vis_interval: int = 1,
):
    """
    CRAFT(Model1)를 한 에폭(epoch) 동안 학습시키고, 주기적으로 시각화를 수행합니다.

    Args:
        model (nn.Module): 학습할 CRAFT 모델.
        train_loader (DataLoader): 학습 데이터로더.
        optimizer (torch.optim.Optimizer): 옵티마이저.
        device (torch.device): 학습에 사용할 디바이스.
        epoch (int): 현재 에폭 번호 (0-based index).
        num_epochs (int): 총 학습 에폭 수.
        vis_interval (int): 시각화를 수행할 에폭 주기. (예: 1이면 매 에폭, 5면 5 에폭마다)
        vis_save_dir (str): 시각화 결과물을 저장할 디렉토리 경로.

    Returns:
        float: 해당 에폭의 평균 손실(loss).
    """
    # 1. 초기화
    model.train()  # 모델을 학습 모드로 설정
    total_loss = 0.0

    # 2. 배치 단위 학습 루프
    num_batches = len(train_loader)
    batch_iterator = tqdm(
        train_loader,
        desc=f"Epoch {epoch + 1}/{num_epochs} - Model1 Train",
        leave=False,
        dynamic_ncols=True,
    )

    for batch_idx, (images, region_targets, affinity_targets, _, _) in enumerate(
        batch_iterator
    ):
        # --- 데이터 준비 ---
        images = images.to(device)
        region_targets = region_targets.to(device)
        affinity_targets = affinity_targets.to(device)

        # --- 순전파 (Forward Pass) ---
        optimizer.zero_grad()
        # 모델 출력: (B, H, W, 2), 여기서 채널 2는 [region, affinity]
        pred_maps, _ = model(images)

        # 손실 계산을 위해 채널을 앞으로 이동: (B, H, W, 2) -> (B, 2, H, W)
        pred_maps = pred_maps.permute(0, 3, 1, 2)

        pred_region = pred_maps[:, 0:1, :, :]
        pred_affinity = pred_maps[:, 1:2, :, :]

        # --- 손실 계산 (Loss Calculation) ---
        # 불균형한 데이터셋을 위한 가중치(pos_weight) 계산
        pos_count = torch.sum(region_targets)
        neg_count = region_targets.numel() - pos_count
        # 0으로 나누는 것을 방지하기 위해 작은 값(epsilon) 추가
        pos_weight = torch.tensor([neg_count / (pos_count + 1e-6)], device=device)

        loss_region = F.binary_cross_entropy_with_logits(
            pred_region, region_targets, pos_weight=pos_weight
        )
        loss_affinity = F.binary_cross_entropy_with_logits(
            pred_affinity, affinity_targets, pos_weight=pos_weight
        )
        batch_loss = loss_region + loss_affinity

        # --- 역전파 (Backward Pass) ---
        batch_loss.backward()
        optimizer.step()

        # --- 로깅 ---
        total_loss += batch_loss.item()
        batch_iterator.set_postfix(loss=f"{batch_loss.item():.4f}")

    # 3. 에폭 종료 후 처리

    # --- 시각화 ---
    # 시각화 주기가 되었는지 확인
    if (epoch + 1) % vis_interval == 0:
        print(f"\n[Trainer] Visualizing output for epoch {epoch + 1}...")
        # batch_iterator 루프에서 마지막 배치의 데이터를 사용
        # detach()를 사용하여 계산 그래프에서 분리
        visualize_training_progress_model1(
            gt_region_batch=region_targets,
            gt_affinity_batch=affinity_targets,
            pred_region_batch=pred_region.detach(),
            pred_affinity_batch=pred_affinity.detach(),
            epoch=epoch,
            save_dir=vis_save_dir,
        )

    # --- 평균 손실 반환 ---
    avg_epoch_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_epoch_loss


def visualize_training_progress_model1(
    gt_region_batch: torch.Tensor,
    gt_affinity_batch: torch.Tensor,
    pred_region_batch: torch.Tensor,
    pred_affinity_batch: torch.Tensor,
    epoch: int,
    save_dir: str,
    prefix: str = "model1_train",
):
    """
    CRAFT(Model1) 학습 중인 한 배치의 GT와 Prediction을 시각화합니다.
    배치 내에서 랜덤하게 하나의 샘플을 선택하여 저장합니다.

    Args:
        gt_region_batch (torch.Tensor): Region GT 맵 배치 (B, 1, H, W).
        gt_affinity_batch (torch.Tensor): Affinity GT 맵 배치 (B, 1, H, W).
        pred_region_batch (torch.Tensor): 예측된 Region 맵 배치 (B, 1, H, W). (Logits 상태)
        pred_affinity_batch (torch.Tensor): 예측된 Affinity 맵 배치 (B, 1, H, W). (Logits 상태)
        epoch (int): 현재 에폭 번호 (0부터 시작).
        save_dir (str): 이미지를 저장할 디렉토리.
        prefix (str): 저장할 파일명의 접두사.
    """
    # 배치에서 시각화할 샘플 랜덤 선택
    batch_size = gt_region_batch.size(0)
    sample_idx = random.randint(0, batch_size - 1)

    # 선택된 샘플의 GT와 Prediction 추출 및 CPU로 이동
    gt_region = gt_region_batch[sample_idx, 0].cpu().numpy()
    gt_affinity = gt_affinity_batch[sample_idx, 0].cpu().numpy()

    # Prediction은 로짓(logits) 상태이므로 sigmoid를 적용하여 확률로 변환
    pred_region = torch.sigmoid(pred_region_batch[sample_idx, 0]).cpu().numpy()
    pred_affinity = torch.sigmoid(pred_affinity_batch[sample_idx, 0]).cpu().numpy()

    # 1행 4열의 서브플롯 생성
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(f"Epoch {epoch + 1} - Sample from Batch", fontsize=16)

    # GT Region
    im1 = axes[0].imshow(gt_region, cmap="jet", vmin=0, vmax=1)
    axes[0].set_title("GT Region")
    axes[0].axis("off")
    fig.colorbar(im1, ax=axes[0])

    # Pred Region
    im2 = axes[1].imshow(pred_region, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("Pred Region")
    axes[1].axis("off")
    fig.colorbar(im2, ax=axes[1])

    # GT Affinity
    im3 = axes[2].imshow(gt_affinity, cmap="jet", vmin=0, vmax=1)
    axes[2].set_title("GT Affinity")
    axes[2].axis("off")
    fig.colorbar(im3, ax=axes[2])

    # Pred Affinity
    im4 = axes[3].imshow(pred_affinity, cmap="jet", vmin=0, vmax=1)
    axes[3].set_title("Pred Affinity")
    axes[3].axis("off")
    fig.colorbar(im4, ax=axes[3])

    # 이미지 저장
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(
        save_dir, f"{prefix}_epoch{epoch + 1}_sample{sample_idx}.png"
    )
    plt.savefig(save_path)
    plt.close(fig)
    print(f"\n[Visualizer] Saved training visualization to {save_path}")
