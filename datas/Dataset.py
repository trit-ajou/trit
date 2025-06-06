import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset


class MangaDataset1(Dataset):
    def __init__(self):
        super().__init__()

    def __len__(self):
        return len()

    def __getitem__(self, idx: int):
        pass


class MangaDataset2(Dataset):
    def __init__(self, texted_images, transform=False):
        super().__init__()
        self.texted_images = texted_images
        self.transform = transform

        if self.transform:
            self.img_transform = transforms.Compose(
                [
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                    transforms.RandomHorizontalFlip(p=0.3),
                ]
            )

    def __len__(self):
        return len(self.texted_images)

    def __getitem__(self, idx: int):
        texted_image = self.texted_images[idx]

        # Get image and mask
        timg = texted_image.timg  # (C, H, W)
        mask = texted_image.mask  # (1, H, W)

        # Apply transforms if enabled
        if self.transform:
            # Apply same transform to both image and mask
            # Convert to PIL for transforms
            timg_pil = transforms.ToPILImage()(timg)
            mask_pil = transforms.ToPILImage()(mask)

            # Apply transforms
            timg_pil = self.img_transform(timg_pil)

            # Convert back to tensor
            timg = transforms.ToTensor()(timg_pil)
            mask = transforms.ToTensor()(mask_pil)

        # Ensure mask is in correct format for training
        if mask.dim() == 3 and mask.shape[0] == 1:
            mask = mask.squeeze(0)  # (1, H, W) -> (H, W)
        elif mask.dim() == 2:
            pass  # Already (H, W)
        else:
            raise ValueError(f"Unexpected mask dimensions: {mask.shape}")

        return timg, mask


class MangaDataset3(Dataset):
    def __init__(self):
        super().__init__()

    def __len__(self):
        return len()

    def __getitem__(self, idx: int):
        pass
