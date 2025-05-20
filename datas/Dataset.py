from torch.utils.data import Dataset

from .TextedImage import TextedImage


class MangaDataset(Dataset):
    def __init__(self, texted_images: list[TextedImage], input_size: tuple[int, int]):
        super().__init__()
        self.texted_images = texted_images
        self.input_size = input_size

    def __len__(self):
        return len(self.texted_images)

    def __getitem__(self, idx: int):
        texted_image = self.texted_images[idx]
        return texted_image._resize(self.input_size)
