import os
import torch
import torch.optim
from enum import Enum
from PIL import ImageDraw
from matplotlib import pyplot as plt


class ModelType(Enum):
    BBOX = 1
    MASK = 2
    INPAINT = 3


class ModelMode(Enum):
    SKIP = 0
    TRAIN = 1
    INFERENCE = 2


def save_checkpoint(state, path="datas/checkpoints/modelX.pth"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer = None,
    path="datas/checkpoints/modelX.pth",
    device="cpu",
):
    if not os.path.exists(path):
        print(f"Checkpoint file not found: {path}")
        return None

    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    if optimizer and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    start_epoch = checkpoint.get("epoch", 0)
    best_loss = checkpoint.get("best_loss", float("inf"))
    print(f"Checkpoint loaded from {path}. Resuming from epoch {start_epoch+1}.")
    return start_epoch, best_loss


# --- Visualization ---
def visualize_model1_output(
    image_pil, target_bboxes, pred_bboxes=None, output_path="model1.png"
):
    """Visualizes model 1 inputs and outputs."""
    draw = ImageDraw.Draw(image_pil)
    for bbox in target_bboxes:  # target_bboxes are (x1,y1,x2,y2)
        if bbox is None or sum(bbox) == 0:
            continue
        draw.rectangle(bbox, outline="green", width=2)

    if pred_bboxes:
        for bbox in pred_bboxes:  # pred_bboxes are (x1,y1,x2,y2)
            if bbox is None or sum(bbox) == 0:
                continue
            draw.rectangle(bbox, outline="red", width=2)

    image_pil.save(output_path)
    print(f"Model 1 visualization saved to {output_path}")


def visualize_model2_output(
    cropped_image_pil, target_mask_pil, pred_mask_pil=None, output_path="model2.png"
):
    """Visualizes model 2 inputs and outputs."""
    num_images = 2 + (1 if pred_mask_pil else 0)
    fig, axes = plt.subplots(1, num_images, figsize=(5 * num_images, 5))

    axes[0].imshow(
        cropped_image_pil, cmap="gray" if cropped_image_pil.mode == "L" else None
    )
    axes[0].set_title("Input Crop")
    axes[0].axis("off")

    axes[1].imshow(target_mask_pil, cmap="gray")
    axes[1].set_title("Target Mask")
    axes[1].axis("off")

    if pred_mask_pil:
        axes[2].imshow(pred_mask_pil, cmap="gray")
        axes[2].set_title("Predicted Mask")
        axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)
    print(f"Model 2 visualization saved to {output_path}")


def visualize_model3_output(
    input_crop_pil,
    input_mask_pil,
    inpainted_result_pil,
    target_clear_crop_pil,
    output_path="model3.png",
):
    """Visualizes model 3 inputs and outputs."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(input_crop_pil, cmap="gray" if input_crop_pil.mode == "L" else None)
    axes[0].set_title("Input Crop (with text)")
    axes[0].axis("off")

    axes[1].imshow(input_mask_pil, cmap="gray")
    axes[1].set_title("Input Mask")
    axes[1].axis("off")

    axes[2].imshow(
        inpainted_result_pil, cmap="gray" if inpainted_result_pil.mode == "L" else None
    )
    axes[2].set_title("Inpainted Result")
    axes[2].axis("off")

    axes[3].imshow(
        target_clear_crop_pil,
        cmap="gray" if target_clear_crop_pil.mode == "L" else None,
    )
    axes[3].set_title("Target Clear Crop")
    axes[3].axis("off")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)
    print(f"Model 3 visualization saved to {output_path}")
