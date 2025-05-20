import torch
from torch import nn
from torch import Tensor as img_tensor
from torchvision.ops import nms

from ..datas.Utils import BBox


class Model1(nn.Module):
    """Text Object BBox Detection Model"""

    def __init__(self, input_size: tuple[int, int], max_objects: int):
        super().__init__()
        self.input_size = input_size
        self.max_objects = max_objects
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, max_objects * (4 + 1)),
        )

    def forward(self, x: img_tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.features(x)
        x = x.view(-1, self.max_objects, 5)
        x = torch.sigmoid(x)
        bboxes = torch.sigmoid(x[:, :, :4])  # (batch_size, max_objects, 4)
        scores = x[:, :, 4:]  # (batch_size, max_objects, 1)
        return bboxes, scores

    @staticmethod
    def preprocess(
        bboxes: list[BBox], input_size: tuple[int, int], max_objects: int, device
    ) -> torch.Tensor:
        H, W = input_size
        num_to_process = min(len(bboxes), max_objects)
        target_bboxes_tensor = torch.zeros(
            (max_objects, 4), dtype=torch.float32, device=device
        )
        target_scores_tensor = torch.zeros(
            (max_objects, 1), dtype=torch.float32, device=device
        )
        if num_to_process > 0:
            bboxes_to_convert = [bbox_tuple for bbox_tuple in bboxes[:num_to_process]]
            bboxes_data = torch.tensor(
                bboxes_to_convert, dtype=torch.float32, device=device
            )
            norm_factors = torch.tensor(
                [W, H, W, H], dtype=torch.float32, device=device
            ).view(1, 4)
            normalized_bboxes = bboxes_data / norm_factors
            target_bboxes_tensor[:num_to_process] = normalized_bboxes
            target_scores_tensor[:num_to_process, 0] = 1.0

        return target_bboxes_tensor, target_scores_tensor

    def postprocess(
        self, preds: torch.Tensor, confidence_threshold=0.5, nms_threshold=0.4
    ) -> list[BBox]:
        H, W = self.input_size
        bboxes = preds[:, :, :4]
        scores = preds[:, :, 4]  # squeeze
        # 1. Confidence Thresholding
        filter = scores > confidence_threshold
        bboxes = bboxes[filter]  # (num_filtered, 4)
        scores = scores[filter]  # (num_filtered,)
        if bboxes.numel() == 0:
            return []
        # 2. 좌표 Denormalization
        denormed_bboxes = torch.zeros_like(bboxes)
        denormed_bboxes[:, 0] = bboxes[:, 0] * W
        denormed_bboxes[:, 1] = bboxes[:, 1] * H
        denormed_bboxes[:, 2] = bboxes[:, 2] * W
        denormed_bboxes[:, 3] = bboxes[:, 3] * H
        bboxes = denormed_bboxes
        # 3. NMS (Non-Maximum Suppression)
        # torchvision.ops.nms는 (x1, y1, x2, y2) 포맷을 사용합니다.
        x1 = torch.min(bboxes[:, 0], bboxes[:, 2])
        x2 = torch.max(bboxes[:, 0], bboxes[:, 2])
        y1 = torch.min(bboxes[:, 1], bboxes[:, 3])
        y2 = torch.max(bboxes[:, 1], bboxes[:, 3])
        bboxes = torch.stack([x1, y1, x2, y2], dim=1)
        # nms 적용
        nms_result = nms(bboxes, scores, nms_threshold)
        bboxes = bboxes[nms_result]
        # 4. BBox 객체로 변환
        result_bboxes: list[BBox] = []
        for bbox in bboxes:
            result_bboxes.append(
                BBox(bbox[0].item(), bbox[1].item(), bbox[2].item(), bbox[3].item())
            )
        return result_bboxes


class Model1Loss(nn.Module):
    def __init__(
        self,
        lambda_coord: float = 1.0,
        lambda_obj: float = 1.0,
        beta_smooth_l1: float = 1.0,
    ):
        super().__init__()
        self.lambda_coord = lambda_coord
        self.lambda_obj = lambda_obj
        self.smooth_l1_loss = nn.SmoothL1Loss(reduction="sum", beta=beta_smooth_l1)
        self.bce_loss = nn.BCEWithLogitsLoss(reduction="sum")

    def forward(
        self,
        pred_bboxes: torch.Tensor,  # (B, MAX_OBJECTS, 4) - Normalized [x1, y1, x2, y2]
        pred_scores: torch.Tensor,  # (B, MAX_OBJECTS, 1) - Objectness scores (after sigmoid)
        target_bboxes: torch.Tensor,  # (B, MAX_OBJECTS, 4) - [norm_x1,y1,x2,y2, presence_flag]
        target_scores: torch.Tensor,  # (B, MAX_OBJECTS, 1) - [norm_x1,y1,x2,y2, presence_flag]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # 타겟에서 좌표와 객체 존재 플래그 분리
        gt_objectness_for_bce = target_scores
        # 1. Localization Loss (좌표 회귀 손실)
        # 객체가 실제로 존재하는 위치 (positive samples)에 대해서만 계산
        # positive_mask: (B, MAX_OBJECTS), True인 곳이 객체 존재
        positive_mask = (target_scores == 1.0).squeeze_()
        # 마스크를 사용하여 positive 샘플의 예측 박스와 타겟 박스 선택
        pred_bboxes = pred_bboxes[positive_mask]  # (num_pos_objects, 4)
        target_bboxes = target_bboxes[positive_mask]  # (num_pos_objects, 4)
        num_pos_objects = target_bboxes.size(0)  # 배치 내의 총 실제 객체 수

        if num_pos_objects > 0:
            loc_loss = self.smooth_l1_loss(pred_bboxes, target_bboxes)
            # 실제 객체 수로 정규화 (평균 손실)
            loc_loss_normalized = loc_loss / num_pos_objects
        else:
            loc_loss = torch.tensor(
                0.0, device=pred_bboxes.device, dtype=pred_bboxes.dtype
            )
            loc_loss_normalized = torch.tensor(
                0.0, device=pred_bboxes.device, dtype=pred_bboxes.dtype
            )

        # 2. Objectness Loss (객체 존재 점수 손실)
        # 모든 MAX_OBJECTS 위치에 대해 계산 (positive 및 negative)
        # pred_scores: (B, MAX_OBJECTS, 1)
        # gt_objectness_for_bce: (B, MAX_OBJECTS, 1)
        obj_loss = self.bce_loss(pred_scores, gt_objectness_for_bce)
        # 정규화: 총 예측 수 (B * MAX_OBJECTS)로 나눔 (평균 손실)
        num_total_predictions = pred_scores.numel()  # B * MAX_OBJECTS * 1
        obj_loss_normalized = obj_loss / num_total_predictions
        # 총 손실 (가중치 적용)
        # 여기서 loc_loss_normalized와 obj_loss_normalized를 사용해야 함
        total_loss = (
            self.lambda_coord * loc_loss_normalized
            + self.lambda_obj * obj_loss_normalized
        )
        return total_loss, loc_loss_normalized, obj_loss_normalized
