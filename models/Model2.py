import torch
from torch import nn
import torchvision.models.segmentation as segmentation
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, num_classes=2, pretrained=True):
        # Load pretrained DeepLabv3 with ResNet-50 backbone
        weights = DeepLabV3_ResNet50_Weights.DEFAULT if pretrained else None
        self.deeplabv3 = segmentation.deeplabv3_resnet50(weights=weights)

        # Modify classifier for binary segmentation (background vs text)
        self.deeplabv3.classifier = segmentation.deeplabv3.DeepLabHead(
            in_channels=2048,
            num_classes=num_classes,
        )

        # Auxiliary classifier (if exists) - fix parameter name
        if hasattr(self.deeplabv3, "aux_classifier") and self.deeplabv3.aux_classifier is not None:
            self.deeplabv3.aux_classifier = segmentation.fcn.FCNHead(
                in_channels=1024,
                channels=num_classes,  # Use 'channels' instead of 'num_classes'
            )

    def forward(self, x):
        """
        Forward pass for pixel-wise text mask generation

        Args:
            x: Input tensor of shape (batch_size, 3, height, width)

        Returns:
            Dictionary containing:
            - 'out': Main output tensor of shape (batch_size, num_classes, height, width)
            - 'aux': Auxiliary output (if training mode)
        """
        output = self.deeplabv3(x)

        # Apply sigmoid to get probabilities for text pixels
        if isinstance(output, dict):
            output["out"] = torch.sigmoid(output["out"])
            if "aux" in output:
                output["aux"] = torch.sigmoid(output["aux"])
        else:
            output = torch.sigmoid(output)

        return output

    def predict_mask(self, x, threshold=0.5):
        """
        Generate binary mask predictions

        Args:
            x: Input tensor
            threshold: Threshold for binary classification

        Returns:
            Binary mask tensor (height, width) - single image output
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(x)
            if isinstance(output, dict):
                logits = output["out"]
            else:
                logits = output

            # Get text class probabilities (assuming class 1 is text)
            text_probs = logits[:, 1, :, :] if logits.shape[1] > 1 else logits[:, 0, :, :]
            binary_mask = (text_probs > threshold).float()

            # Return single image mask (H, W)
            return binary_mask.squeeze(0)  # Remove batch dimension
          