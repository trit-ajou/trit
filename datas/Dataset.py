from torch.utils.data import Dataset, DataLoader, random_split

from .TextedImage import TextedImage
from ..models.Model1 import Model1
from ..Utils import PipelineSetting


class MangaDataset1(Dataset):
    def __init__(self, texted_images: list[TextedImage], setting: PipelineSetting):
        super().__init__()
        self.texted_images = texted_images
        self.input_size = setting.model1_input_size
        self.max_objects = setting.max_objects
        self.device = setting.device

    def __len__(self):
        return len(self.texted_images)

    def __getitem__(self, idx: int):
        texted_image = self.texted_images[idx]
        texted_image._resize(self.input_size)
        target_bboxes, target_scores = Model1.bboxes2tensor(
            texted_image.bboxes, self.input_size, self.max_objects, self.device
        )
        return {
            "timg": texted_image.timg,
            "target_bboxes": target_bboxes,
            "target_scores": target_scores,
        }

    @staticmethod
    def get_dataloader(
        texted_images: list[TextedImage],
        setting: PipelineSetting,
        train_valid_split=1.0,  # 1.0 for valid set only
    ):
        full_set = MangaDataset1(texted_images, setting)
        valid_portion = train_valid_split
        train_portion = 1 - valid_portion
        train_set, valid_set = random_split(full_set, (train_portion, valid_portion))
        train_loader = (
            DataLoader(
                train_set,
                batch_size=setting.batch_size,
                num_workers=setting.num_workers,
                persistent_workers=(setting.num_workers > 0),
                drop_last=True,
            )
            if len(train_set) > 0
            else None
        )
        valid_loader = DataLoader(
            valid_set,
            batch_size=setting.batch_size,
            num_workers=setting.num_workers,
            persistent_workers=(setting.num_workers > 0),
        )
        return train_loader, valid_loader
