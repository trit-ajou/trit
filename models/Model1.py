import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from .BaseModel import BaseModel  # Corrected import path based on file structure
from typing import List, Dict, Optional  # For type hinting


class Model1(BaseModel):
    """Text Object BBox Detection Model using Faster R-CNN"""

    def __init__(
        self, num_classes=2, device: Optional[str] = None
    ):  # num_classes = 1 (text) + 1 (background)
        # If device is not specified, BaseModel will handle the default
        super().__init__(device=device)

        # Load a pre-trained Faster R-CNN model
        # Using updated weights API
        weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            weights=weights
        )

        # Get the number of input features for the classifier
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features

        # Replace the pre-trained head with a new one
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        self.model.to(self.device)  # Move model to the device specified in BaseModel

    def forward(
        self,
        images: List[torch.Tensor],
        targets: Optional[List[Dict[str, torch.Tensor]]] = None,
    ):
        """
        Args:
            images (list[Tensor]): images to be processed
            targets (list[Dict[str, Tensor]], optional): ground-truth boxes present in the image.
                                                        Required for training.
        Returns:
            During training, returns a dict[str, Tensor] which contains the losses.
            During inference, returns a list[Dict[str, Tensor]] containing predicted boxes, labels, and scores.
        """
        if self.training and targets is None:
            # In torchvision's Faster R-CNN, targets being None is valid even in training mode.
            # The model then expects images to be a list of Tensors and targets to be None or a list of dicts.
            # However, for clarity in our pipeline, if we are training, we expect targets.
            # For now, let's raise a warning or proceed if torchvision handles it.
            # According to torchvision docs, for training, targets should not be None.
            # So, raising an error is appropriate if our training loop contract expects targets.
            raise ValueError(
                "In training mode, targets should be passed to the forward method."
            )

        # Ensure images are on the correct device
        images = [img.to(self.device) for img in images]

        processed_targets = None
        if targets is not None:
            # Ensure targets are on the correct device
            # Targets is a list of dicts, each dict has 'boxes' and 'labels'
            processed_targets = []
            for t_dict in targets:
                processed_targets.append(
                    {
                        "boxes": t_dict["boxes"].to(self.device),
                        "labels": t_dict["labels"].to(self.device),
                    }
                )

        # The model handles targets being None during inference.
        return self.model(images, processed_targets)
