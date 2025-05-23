import torch
import torch.nn as nn
import os


class BaseModel(nn.Module):
    def __init__(self, device=None):
        super().__init__()
        self.device = (
            device
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

    def save_checkpoint(self, model_path, epoch, optimizer_state_dict=None):
        # Ensure directory exists
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.state_dict(),
        }
        if optimizer_state_dict:
            checkpoint["optimizer_state_dict"] = optimizer_state_dict

        try:
            torch.save(checkpoint, model_path)
            print(f"Checkpoint saved to {model_path} at epoch {epoch}")
        except Exception as e:
            print(f"Error saving checkpoint to {model_path}: {e}")

    def load_checkpoint(self, model_path, optimizer=None):
        if not os.path.exists(model_path):
            print(f"No checkpoint found at {model_path}. Starting from scratch.")
            return 0  # Return epoch 0 to indicate training from the beginning

        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.load_state_dict(checkpoint["model_state_dict"])

            # Epochs are 0-indexed during saving, so start_epoch is epoch + 1
            start_epoch = checkpoint.get("epoch", -1) + 1

            if optimizer and "optimizer_state_dict" in checkpoint:
                try:
                    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                    print(
                        f"Loaded checkpoint including optimizer state from {model_path} (resuming from epoch {start_epoch})"
                    )
                except Exception as e:
                    print(
                        f"Error loading optimizer state from {model_path}: {e}. Optimizer state not loaded."
                    )
            else:
                print(
                    f"Loaded model state checkpoint from {model_path} (resuming from epoch {start_epoch})"
                )
                if optimizer and "optimizer_state_dict" not in checkpoint:
                    print(
                        "Optimizer state not found in checkpoint. Optimizer not loaded."
                    )

            return start_epoch
        except (
            FileNotFoundError
        ):  # Should be caught by os.path.exists, but as a safeguard
            print(f"No checkpoint found at {model_path}. Starting from scratch.")
            return 0
        except Exception as e:
            print(
                f"Error loading checkpoint from {model_path}: {e}. Starting from scratch."
            )
            return 0
