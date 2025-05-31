import os
import re
import torch
from enum import Enum
import numpy as np

class ModelType(Enum):
    BBOX = 1
    MASK = 2
    INPAINT = 3


class ModelMode(Enum):
    SKIP = 0
    TRAIN = 1
    INFERENCE = 2
def tensor_rgb_to_cv2(t: torch.Tensor) -> np.ndarray:
    """
    (C,H,W) 0-1 float RGB tensor ➜ (H,W,3) 0-255 uint8 BGR ndarray
    """
    arr = (t.detach().cpu().numpy() * 255).astype(np.uint8)   # CHW, RGB
    arr = arr.transpose(1, 2, 0)                              # HWC, RGB
    return arr[..., ::-1]                                     # BGR

def save_ckpt(
    ckpt_dir,
    epoch,
    best_acc,
    model,
    optimizer,
    scheduler,
    scaler,
    train_loss_history,
    train_acc_history,
    valid_loss_history,
    valid_acc_history,
):
    checkpoint = {
        "epoch": epoch,
        "best_acc": best_acc,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "train_loss_history": train_loss_history,
        "train_acc_history": train_acc_history,
        "valid_loss_history": valid_loss_history,
        "valid_acc_history": valid_acc_history,
    }
    os.makedirs(ckpt_dir, exist_ok=True)
    filename = f"{model.__class__.__name__}_epoch{epoch}.pth"
    torch.save(checkpoint, ckpt_dir + "/" + filename)


def load_ckpt(
    ckpt_dir,
    model,
    optimizer,
    scheduler,
    scaler,
    device,
):
    model_class_name = model.__class__.__name__
    pattern = re.compile(rf"^{re.escape(model_class_name)}_epoch(\d+)\.pth$")

    latest_epoch = -1
    latest_ckpt_path = None

    for filename in os.listdir(ckpt_dir):
        match = pattern.match(filename)
        if match:
            epoch = int(match.group(1))
            if epoch > latest_epoch:
                latest_epoch = epoch
                latest_ckpt_path = os.path.join(ckpt_dir, filename)

    if latest_ckpt_path is None:
        return model, optimizer, scheduler, scaler, -1, 0.0, [], [], [], []

    checkpoint = torch.load(latest_ckpt_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    scaler.load_state_dict(checkpoint["scaler_state_dict"])

    epoch = checkpoint.get("epoch", -1)
    best_acc = checkpoint.get("best_acc", 0.0)
    train_loss_history = checkpoint.get("train_loss_history", [])
    train_acc_history = checkpoint.get("train_acc_history", [])
    valid_loss_history = checkpoint.get("valid_loss_history", [])
    valid_acc_history = checkpoint.get("valid_acc_history", [])

    return (
        epoch + 1,
        best_acc,
        model,
        optimizer,
        scheduler,
        scaler,
        train_loss_history,
        train_acc_history,
        valid_loss_history,
        valid_acc_history,
    )
